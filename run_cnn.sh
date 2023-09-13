#!/bin/bash
#$ -cwd 
#$ -e ./log
#$ -o ./log
#$ -l h_data=8G,h_rt=12:00:00,highp
#$ -pe shared 3
#$ -M fujy2038
#$ -m bea
#$ -t 1-4:1

source /etc/profile     # so module command is recognized
module load python/3.9.6
cd ./

perc_mask="20 50 75 90"
arr=($perc_mask)
num_perc=$(($SGE_TASK_ID-1))

python3 ./generate_train_test_data_beagle.py -m ${arr[$num_perc]}

python3 ./impute_input.py \
    -r ./beagle_data/processed/train_output_${arr[$num_perc]}_new_100.pkl \
    -t ./beagle_data/processed/train_input_${arr[$num_perc]}_new_100.pkl \
    -c all \
    -p all \
    -i ./beagle_data/processed/test_input_${arr[$num_perc]}_100.pkl \
    -o ./output/test_output_cnn_imputed_${arr[$num_perc]} \
    -d $'\t' \
    -w 20 \
    -g 20 \
    -e 5 \
    -b 32 \
    --lr 1e-3 \
    --kr 0 \
    --drop_perc 0.25 \
    --model_layers 3 \
    --gamma 5 \
    --optimizer adam \
    --hidden_layer_s 32 \
    --to_categorical True \
    --loss_func focal_loss


