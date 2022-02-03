import os
import shutil
import argparse

in_root = os.path.join('/home/oem/Mingle/Project/tomato_leaf_translation/results/',
                       'health2powder_seg_scit_style20',
                       'test_latest/images'
                       )
out_root = os.path.join('/home/oem/Mingle/Project/tomato_leaf_translation/results/',
                        'fake_powder_seg_scit',
                        'powder_object_detection_style20'
                        )

os.makedirs(out_root)

names = os.listdir(in_root)
for name in names:
    if 'fake_B' in name:
        in_name = os.path.join(in_root, name)
        out_name = os.path.join(out_root, name)
        shutil.copyfile(in_name, out_name)