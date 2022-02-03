export date=$1
export type=$2
export style=$3
export epoch=$4
export result_dir=$5

export gpu=0
export batch=4

# tomato_instance_object
python test.py \
--model "scit_seg" \
--dataset_mode 'tomato_seg' \
--name "$date"_health2"$type"_seg_scit_style"$style" \
--dataroot "./datasets/ins_orig_translation_all" \
--gpu_ids $gpu \
--batch_size $batch \
--translation_type health2"$type" \
--epoch $epoch \
--task 'ins' \
--results_dir $result_dir \
--load_size 256 \
--crop_size 256