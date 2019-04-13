#/bin/bash

X="0.01 0.016 0.017 0.018 0.019 0.02 0.025 0.023 0.035 0.04 0.045 0.05 0.055 0.06 0.065 0.07  1  10000"
Z="0.0001 0.0005  0.001 0.002 0.003 0.004 0.005 0.06 0.008 0.009 0.01 0.02 0.04 0.06 0.08 0.1 0.2 0.3 0.4"

for i in $X
do 
for l in $Z  
    do
    #Replace echo with you python command
    python   cross_validation_Gaussian_noise.py   --epsilon $i  --sigma $l  --output_dir  './gaussian_noise/'
   done
 done
