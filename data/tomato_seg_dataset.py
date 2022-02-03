import os
import os.path as osp
from data.base_dataset import BaseDataset, get_transform_both
from data.image_folder import make_dataset
from PIL import Image
import random
import re
import torchvision.transforms.functional as F
import numpy as np


class TomatoSegDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        translation_type = opt.translation_type
        class_a, class_b = translation_type.split('2', 1)
        self.class_a, self.class_b = class_a, class_b
        self.phase = self.opt.phase
        if opt.task == 'clf':
            if opt.phase.lower() == 'train':
                class_a = class_a + '_instance_seg_img_split_aug'
                class_b = class_b + '_instance_seg_img_split_aug'
                self.dir_a = osp.join(opt.dataroot, class_a, 'train_seg')
                self.dir_b = osp.join(opt.dataroot, class_b, 'train_seg')
            else:
                class_a = class_a + '_instance_seg_img'
                class_b = class_b + '_instance_seg_img'
                self.dir_a = osp.join(opt.dataroot, class_a, 'seg')
                self.dir_b = osp.join(opt.dataroot, class_b, 'seg')
        elif "ins" in opt.task:
            if opt.phase.lower() == 'train':
                class_a = class_a + '_seg_img_split_aug'
                class_b = class_b + '_seg_img_split_aug'
                self.dir_a = osp.join(opt.dataroot, class_a, 'train_seg')
                self.dir_b = osp.join(opt.dataroot, class_b, 'train_seg')
            else:
                class_a = class_a
                class_b = class_b
                self.dir_a = osp.join(opt.dataroot, class_a, 'mask')
                self.dir_b = osp.join(opt.dataroot, class_b, 'mask')

        self.path_seg_a = sorted(make_dataset(self.dir_a, opt.max_dataset_size))
        self.path_seg_b = sorted(make_dataset(self.dir_b, opt.max_dataset_size))
        self.len_a = len(self.path_seg_a)
        self.len_b = len(self.path_seg_b)

    def __len__(self):
        print(f'{self.class_a} is {self.len_a}')
        print(f'{self.class_b} is {self.len_b}')
        return max(len(self.path_seg_a), len(self.path_seg_b))

    def __getitem__(self, item):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            seg_A (tensor)   -- a mask in the input domain
            seg_B (tensor)
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            seg_A_paths (str)    -- seg_A paths
            seg_B_paths (str)    -- seg_B paths
        """
        seg_A_path = self.path_seg_a[item % self.len_a]
        if self.opt.serial_batches:   # make sure index is within then range
            item_B = item % self.len_b
        else:   # randomize the index for domain B to avoid fixed pairs.
            item_B = random.randint(0, self.len_b - 1)
        seg_B_path = self.path_seg_b[item_B % self.len_b]

        if self.opt.task == 'clf':
            if self.phase.lower() == 'train':
                img_A_path = re.sub('_mask(\d+)', '', seg_A_path).replace('n_seg', 'n_img')
                img_B_path = re.sub('_mask(\d+)', '', seg_B_path).replace('n_seg', 'n_img')
            else:
                # img_A_path = re.sub('_mask(\d+)', '', seg_A_path).replace('/seg', '/img')
                # img_B_path = re.sub('_mask(\d+)', '', seg_B_path).replace('/seg', '/img')
                img_A_path = seg_A_path.replace('/seg', '/img')
                img_B_path = seg_B_path.replace('/seg', '/img')
        elif self.opt.task == 'ins' or self.opt.task == 'obj':
            if self.phase.lower() == 'train':
                img_A_path = re.sub('_mask(\d+)', '', seg_A_path).replace('n_seg', 'n_img')
                img_B_path = re.sub('_mask(\d+)', '', seg_B_path).replace('n_seg', 'n_img')
            else:
                img_A_path = re.sub('_mask(\d+)', '', seg_A_path).replace('/mask', '/img')
                img_B_path = re.sub('_mask(\d+)', '', seg_B_path).replace('/mask', '/img')
        # else:
        #     raise NotIm

        A = Image.open(img_A_path).convert('RGB')
        B = Image.open(img_B_path).convert('RGB')
        seg_A = Image.open(seg_A_path).convert('L')
        seg_B = Image.open(seg_B_path).convert('L')

        # apply transform
        A, seg_A = transform(A, seg_A, self.opt)
        B, seg_B = transform(B, seg_B, self.opt)

        return {
            'A': A,
            'B': B,
            'seg_A': seg_A,
            'seg_B': seg_B,
            'A_paths': seg_A_path,
            'B_paths': seg_B_path,
        }


def transform(img, seg, opt):
    # resize
    img = F.resize(img, (opt.load_size, opt.load_size), Image.BICUBIC)
    seg = F.resize(seg, (opt.load_size, opt.load_size), Image.NEAREST)
    # random crop
    if opt.load_size > opt.crop_size:
        top = np.random.randint(opt.load_size - opt.crop_size)
        left = np.random.randint(opt.load_size - opt.crop_size)
        img = F.crop(img, top, left, opt.crop_size, opt.crop_size)
        seg = F.crop(seg, top, left, opt.crop_size, opt.crop_size)
    # random flip
    if opt.phase.lower() == 'train':
        v_dix = np.random.rand()
        h_idx = np.random.rand()
        if v_dix > 0.5:
            img = F.vflip(img)
            seg = F.vflip(seg)
        if h_idx > 0.5:
            img = F.hflip(img)
            seg = F.hflip(seg)
    # to Tensor and normalize
    img = F.to_tensor(img)
    seg = F.to_tensor(seg)
    img = F.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    seg = F.normalize(seg, [0.5, ], [0.5, ])

    return img, seg
