#/bin/bash

X="0 0.0001 0.001 0.01 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019 0.02 0.025 0.023 0.035 0.04 0.045 0.05 0.055 0.06 0.065 0.070 0.075 1 10 100 10000"
Y="0.01 0.1 0 1 10 100 1000"
Z="0.01 0.1 0 1 10 100 1000 "
for i in $X
do 
 for j in $Y
 do
   for l in $Z  
    do
    #Replace echo with you python command
    python  cross_validation.py  --epsilon $i --reg_param1 $j  --reg_param2 $l
   done
 done
done
