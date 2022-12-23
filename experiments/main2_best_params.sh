#!/bin/bash

#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./main2.out
#SBATCH --error=./main2.err
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=carlsonlab

source ~/.bashrc
conda activate bo

python main_2.py \
    --data_path /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
    --dump_path /work/zdc6/wildfire/results/best_params/ \
    --data_dump_path /work/zdc6/wildfire/data/ \
    --model base \
    --tweets_per_sample 1000 \
    --num_samples_per_day 5 \
    --num_components 268 \
    --prior_mean -7.5 \
    --mse_weight 0.08 \
    --batch_size 256 \
    --end_kld 0.027 \
    --lr 0.000086 \
    --b1 0.731666 \
    --b2 0.712416 \
    --wd 0.000216 \
    --use_lds true
    