## Paper info
[Web in Frontiers in Plant Science](https://www.frontiersin.org/articles/10.3389/fpls.2021.773142/abstract)

## The main code work in the paper
'''
models/scit_seg_model.py
models/networks_scit_seg.py

'''

## Prapare the dataset

* The data level is:
```
├── class_instance_seg_img_split_aug
│   ├── test_img
│   ├── test_seg
│   ├── train_img
    │   ├── 000473_bd10.jpg
    │   ├── 000473_bd11.jpg
│   └── train_seg
        ├── 000473_maskID_bd10.jpg (This is the template, ID is the changing value)
        ├── 000473_mask0_bd10.jpg
        ├── 000473_mask0_bd11.jpg
```
Please notice the alignment of the file names.
Please see the dataset examples for health and powder in datasets/tomato_seg

* If you want to use the data format in CycleGAN, Please change --dataset_mode when your training and testing. Please refer to the original CycleGAN code about the details.

## Train
* must utilze the mask or instance segmentation
* the main file: 
```
scripts_tomato/train.sh
scripts_tomato/train_scit_seg.sh
```
* Examples to train
```
meta code: sh ./scripts_tomato/train.sh class12class2 style_lambda

examples:
sh ./scripts_tomato/train.sh health2powder 0
sh ./scripts_tomato/train.sh health2powder 0.5
sh ./scripts_tomato/train.sh health2powder 1
sh ./scripts_tomato/train.sh health2powder 2
sh ./scripts_tomato/train.sh health2powder 4
```

* visualize your data: IP:port 
  *  such as http://113.198.xx.xx:8090/

## Test
```
export code=./scripts_tomato/test_ins_base.sh
export date=20210719
export type=powder
export style=2
export epoch=latest
export result_dir=./results_ins
sh $code $date $type $style $epoch $dir
```

## Compute FID
```
sh scripts_tomato/fids.sh
```
You can change the values in scripts_tomato/fids.sh and scripts_tomato/compute_fid.sh


## To do data augmentation
you can use data_aug.py

## Reference
* [Pytorch cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
    * The codes are borrowed heavily from pytorch cyclegan
    * You can also search help from pytorch cyclegan
## To cite this paper
