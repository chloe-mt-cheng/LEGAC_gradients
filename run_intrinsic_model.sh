#!/bin/bash

objects=( $(tail -n +2 intrinsic_model_cat.csv | cut -d ',' -f1 ) )
slope=( $(tail -n +2 intrinsic_model_cat.csv | cut -d ',' -f18 ) )
integrated=( $(tail -n +2 intrinsic_model_cat.csv | cut -d ',' -f20 ) )
core_Re=( $(tail -n +2 intrinsic_model_cat.csv | cut -d ',' -f21 ) )
outskirt_Re=( $(tail -n +2 intrinsic_model_cat.csv | cut -d ',' -f22 ) )
len=${#objects[@]}

for (( i=0; i<=2; i++ ))
do
    python3 'intrinsic_model.py' --path='/LEGAC' --FeH_m=${slope[$i]} --targ=${objects[$i]} --directory='intrinsic_model' --subpix=529 --core_radius=${core_Re[$i]} --outskirt_radius${outskirt_Re[$i]} --grid_size=1000 &
done
