#!/bin/bash

objects=( $(tail -n +2 working_cat.csv | cut -d ',' -f1 ) )
FWHM=( $(tail -n +2 working_cat.csv | cut -d ',' -f16 ) )
len=${#objects[@]}

for (( i=471; i<=len; i++ ))
do
    python3 'Re_growth_curves.py' --path='path/to/directory' --target=${objects[$i]} --best_seeing=${FWHM[$i]} --directory='growth_curve' --pixscale=0.1 --output_name=${objects[$i]} --input_name=${objects[$i]} --img_factor=21 &
done
