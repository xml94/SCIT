"""
extract real images and resize to 256 to a directory to compute FID
"""

import os
import shutil
import argparse
import PIL.Image as Image
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, )
parser = parser.parse_args()

in_dir = parser.in_dir
out_dir = in_dir + '_256'
os.makedirs(out_dir, exist_ok=True)

names = os.listdir(in_dir)
for name in names:
    old_name = osp.join(in_dir, name)
    new_name = osp.join(out_dir, name)
    img = Image.open(old_name).convert('RGB')
    img = img.resize((256, 256))
    img.save(new_name)