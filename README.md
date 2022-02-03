
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
Please notice the alignment of the file names

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
  *  such as http://113.198.60.xx:8090/

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
You can change the values in fids.sh and compute_fid.sh


## to do data augmentation
you can use data_aug.py

## Reference
* [Pytorch cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
    * The codes are borrowed heavily from pytorch cyclegan
    * You can also search help from pytorch cyclegan
