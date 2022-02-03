"""
do data augmentation for train dataset: img and seg
Input:
    -in_dir <dir>:
        -seg_img_split
            -train_seg <dir>: img.jpg
            -train_img <dir>: imgID_maskID.jpg
Output:
    -in_dir <dir>:
        -seg_img_split_aug
            -train_seg <dir>: imgID_functionID.jpg
            -train_img <dir>: imgID_maskID_functionID.jpg
"""
from tqdm import tqdm
import os
import os.path as osp
import re
import argparse
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import cv2
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_dir(abs_dir):
    if os.path.exists(abs_dir):
        print('%s already exists' % abs_dir)
        raise NotImplementedError('Please check the dir, we use random.')
    else:
        os.makedirs(abs_dir)


parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str)
parser.add_argument('--augment_times', type=int, default=3, help='augment times')
parser = parser.parse_args()

out_dir = parser.in_dir + '/seg_img_split_aug'
in_dir = parser.in_dir + '/seg_img_split'

seg = ['train_seg']
augment_times = parser.augment_times

for suffix in seg:
    seg_dir = osp.join(in_dir, suffix)
    img_dir = osp.join(in_dir, suffix.replace('_seg', '_img'))

    tgt_seg_dir = osp.join(out_dir, suffix)
    tgt_img_dir = osp.join(out_dir, suffix.replace('_seg', '_img'))
    check_dir(tgt_seg_dir)
    check_dir(tgt_img_dir)

    names_seg = os.listdir(seg_dir)
    for name_seg in tqdm(names_seg, desc='Info: doing augmentation for %s' % seg_dir):
        abs_name_seg = osp.join(seg_dir, name_seg)
        name_img = re.sub('_mask\d+', '', name_seg)
        abs_name_img = osp.join(img_dir, name_img)

        source_seg = Image.open(abs_name_seg).convert('L')
        source_img = Image.open(abs_name_img).convert('RGB')

        # copy original image
        new_name_seg = osp.join(tgt_seg_dir, name_seg)
        new_name_img = osp.join(tgt_img_dir, name_img)
        source_seg.save(new_name_seg)
        source_img.save(new_name_img)

        H, W = source_img.size

        # ================
        # adjust contrast
        # ================
        # new_img = ImageOps.autocontrast(source_img)
        # ctrl_suffix = '_ctrst.jpg'
        # new_name_seg = osp.join(tgt_seg_dir, name_seg.replace('.jpg', ctrl_suffix))
        # new_name_img = osp.join(tgt_img_dir, name_img.replace('.jpg', ctrl_suffix))
        # new_img.save(new_name_img)
        # source_seg.save(new_name_seg)


        # ================
        # brighten or darken
        # ================
        median_idx = np.median(np.array(source_img))
        ctrl_idx = np.random.permutation(range(7, 14))[:augment_times] / 10
        for idx in ctrl_idx:
            ctrl_suffix = '_bd%d.jpg' % (idx * 10)
            new_name_seg = osp.join(tgt_seg_dir, name_seg.replace('.jpg', ctrl_suffix))
            new_name_img = osp.join(tgt_img_dir, name_img.replace('.jpg', ctrl_suffix))
            img_bd = source_img.point(lambda p: p * idx)
            img_bd.save(new_name_img)
            source_seg.save(new_name_seg)

        # ================
        # random crop
        # ================
        ctrl_para = 5
        valid_num = 0
        while valid_num < augment_times:
            h = H // ctrl_para
            w = W // ctrl_para
            x, y = np.random.randint(h), np.random.randint(w)

            seg_crop = source_seg.crop((x, y, x + h * (ctrl_para - 1), y + w * (ctrl_para - 1)))
            if np.array(seg_crop).sum() > h * w // 64:
                img_crop = source_img.crop((x, y, x + h * (ctrl_para - 1), y + w * (ctrl_para - 1)))

                ctrl_suffix_crop = '_crop%dx%dy.jpg' % (x, y)
                new_name_seg = osp.join(tgt_seg_dir, name_seg.replace('.jpg', ctrl_suffix_crop))
                new_name_img = osp.join(tgt_img_dir, name_img.replace('.jpg', ctrl_suffix_crop))

                seg_crop.save(new_name_seg)
                img_crop.save(new_name_img)

                valid_num += 1