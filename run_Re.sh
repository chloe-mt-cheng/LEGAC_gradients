#!/bin/bash

objects=( $(tail -n +2 working_cat.csv | cut -d ',' -f1 ) )
FWHM=( $(tail -n +2 working_cat.csv | cut -d ',' -f16 ) )
len=${#objects[@]}

python3 '/Users/chloecheng/Documents/LEGAC_resolved/Re_growth_curves.py' --path='/Users/chloecheng/Documents/LEGAC_resolved' --target='M12_97931' --best_seeing=1.65 --directory='growth_curve' --pixscale=0.1 --output_name='M12_97931' --input_name='M12_97931' --img_factor=31 &
python3 '/Users/chloecheng/Documents/LEGAC_resolved/Re_growth_curves.py' --path='/Users/chloecheng/Documents/LEGAC_resolved' --target='M12_97203' --best_seeing=1.63 --directory='growth_curve' --pixscale=0.1 --output_name='M12_97203' --input_name='M12_97203' --img_factor=31 &

#for (( i=471; i<=len; i++ ))
#do
#    python3 '/Users/chloecheng/Documents/LEGAC_resolved/Re_growth_curves.py' --path='/Users/chloecheng/Documents/LEGAC_resolved' --target=${objects[$i]} --best_seeing=${FWHM[$i]} --directory='growth_curve' --pixscale=0.1 --output_name=${objects[$i]} --input_name=${objects[$i]} --img_factor=21 &
#done
