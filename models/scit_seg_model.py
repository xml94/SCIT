import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_scit_seg as networks
from torchsummary import summary


class ScitSegModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'style_A', 'style_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'seg_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'seg_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
            self.opt.display_ncols += 1

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.define_G(opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.opt.gpu_ids)
        self.netG_B = networks.define_G(opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.opt.gpu_ids)
        # summary(self.netG_A, input_size=[(3, 256, 256), (1, 256, 256)])

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.ndf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # summary(self.netD_A, input_size=[(3, 256, 256), (1, 256, 256)])
            self.netD_B = networks.define_D(opt.ndf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionStyle = networks.VGGLoss(self.opt.gpu_ids)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.seg_A = input['seg_A'].to(self.device)
        self.seg_B = input['seg_B'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A, self.seg_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B, self.seg_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B, self.seg_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A, self.seg_B)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake, seg_real, seg_fake):
        # Real
        pred_real = netD(real, seg_real)
        loss_D_real = self.criterionGAN(pred_real, True, seg_real)
        # Fake
        pred_fake = netD(fake.detach(), seg_fake)
        loss_D_fake = self.criterionGAN(pred_fake, False, seg_fake)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, self.seg_B, self.seg_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, self.seg_A, self.seg_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B, self.seg_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A, self.seg_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B, self.seg_A), True, self.seg_A)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A, self.seg_B), True, self.seg_B)
        # style loss
        self.loss_style_A = self.criterionStyle(self.fake_B * self.seg_A, self.real_A * self.seg_A) * self.opt.lambda_style
        self.loss_style_B = self.criterionStyle(self.fake_A * self.seg_B, self.real_B * self.seg_B) * self.opt.lambda_style
        # self.loss_style_A = 0
        # self.loss_style_B = 0
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + self.loss_style_A + self.loss_style_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def fuse_real_fake(self, realA, fakeB, segA):
        B, C, H, W = realA.size()
        fuse_img = torch.zeros_like(realA)
        if torch.min(segA) < 0:  # <-1, 1> -> <0, 1>
            segA = (segA + 1) / 2
        for batch in range(B):
            real = realA[batch]
            fake = fakeB[batch]
            seg = segA[batch]
            fuse_img[batch] = real * (1 - seg) + fake * seg
        return fuse_img

    def test(self):
        with torch.no_grad():
            self.forward()
            self.fake_B = self.fuse_real_fake(self.real_A, self.fake_B, self.seg_A)
            self.fake_A = self.fuse_real_fake(self.real_B, self.fake_A, self.seg_B)
            self.compute_visuals()
