#!/bin/bash

#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./deep_bayes_opt_lds.out
#SBATCH --error=./deep_bayes_opt_lds.err
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=carlsonlab


source ~/.bashrc
conda activate bo

python optimize_model.py \
    --data_path /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
    --dump_path /work/zdc6/wildfire/results/bo_deep_lds/ \
    --data_dump_path /work/zdc6/wildfire/data/ \
    --save_every 1 \
    --use_lds true \
    --model deep_encoder
