export type=$1
export style=$2

python train.py \
--model "scit_seg" \
--dataset_mode 'tomato_seg' \
--name "20210719_"$type"_seg_scit_style"$style"" \
--lambda_style $style \
--dataroot "./datasets/tomato_seg" \
--batch_size 6 \
--gpu_ids 0,1,2 \
--translation_type $type \
--display_port 8090
#--n_epochs_decay 100 \
#--epoch_count 100 \
#--continue_train