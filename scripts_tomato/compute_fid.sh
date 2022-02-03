
export gpu=1
export batch=1
export name=$1
export type=$2

# make fake image directory
export path="./results_clf/$name/test_latest"
rm -rf "$path/fake_imgs"
python3 ./extract_fake_img.py --in_dir $path

# make real image directory and resize to 256
export original_real_path=./datasets/tomato_seg/"$type"_instance_seg_img/img
export real_path=./datasets/tomato_seg/"$type"_instance_seg_img/img_256
if [ ! -e $real_path ]
then
  python ./extract_real_img.py --in_dir $original_real_path
fi

export fake_path="./results_clf/$name/test_latest/fake_imgs"
python fid_score.py $real_path $fake_path --batch-size $batch --gpu $gpu