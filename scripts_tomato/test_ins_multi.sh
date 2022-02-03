export epoch=latest
export code=./scripts_tomato/test_ins_base.sh
export dir=./results_ins
export date=20210719

for type in powder canker lmold tocv magdef
do
  for style in 0 0.5 1 2 4
  do
    sh $code $date $type $style $epoch $dir
  done
done
