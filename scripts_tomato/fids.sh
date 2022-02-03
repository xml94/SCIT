export code=./scripts_tomato/compute_fid.sh

for date in 20210618 20210624 20210630 20210706 20210719
do
  for type in powder canker lmold tocv magdef
  do
    for style in 0 0.5 1 2 4
    do
      sh $code "$date"_health2"$type"_seg_scit_style"$style" $type
    done
  done
done
