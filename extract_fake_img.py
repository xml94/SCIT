"""
extract fake images to a directory to compute FID
"""

import os
import shutil
import argparse
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, )
parser.add_argument('--suffix', type=str, default='fake_B', help='use it to recognize fake images')
parser = parser.parse_args()

suffix = parser.suffix
in_dir = osp.join(parser.in_dir, 'images')
out_dir = osp.join(parser.in_dir, 'fake_imgs')
os.makedirs(out_dir, exist_ok=True)

names = os.listdir(in_dir)
for name in names:
    if suffix in name:
        old_name = osp.join(in_dir, name)
        new_name = osp.join(out_dir, name)
        shutil.copyfile(old_name, new_name)
