#/bin/bash

X=" 0.05 0.3 0.7 1 1.5 2 2.5 10 10000"
Y="0 0.1 1 10 100"
Z="0 0.1 1 10 100"


for i in $X
do
  for j in $Y
  do
   for k in $Z
    do
    #Replace echo with you python command
    python  cross_validation.py  --epsilon $i --reg_param1 $j --reg_param2 $k  --data_name 'recipe'  --data_dir  '/work/amina/RawM_level_datacleaning2017_6_1/model/StabilityModel/Tensorflow_code/11_27_final/Test_on_text/ICDM/realworld_data/dataset/train_test_seperated/'
   done
 done
done


