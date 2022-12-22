#!/bin/bash

#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./bayes_opt.out
#SBATCH --error=./bayes_opt.err
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=carlsonlab


mamba activate bo

python optimize_model.py \
    --data_path /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
    --dump_path /work/zdc6/wildfire/results/bo/ \
    --data_dump_path /work/zdc6/wildfire/data/ \
    --save_every 1