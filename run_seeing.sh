#!/bin/bash

objects=( $(tail -n +2 moffat_cat_final.csv | cut -d ',' -f1 ) )
len=${#objects[@]}

for (( i=0; i<=2; i++ ))
do
    python3 'calculate_seeing.py' --path='path/to/directory' --target=${objects[$i]} --directory='test' --pixscale=0.205 --output_name=${objects[$i]} --input_name=${objects[$i]} &
done