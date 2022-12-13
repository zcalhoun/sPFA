#!/bin/bash

#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./main2.out
#SBATCH --error=./main2.err
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=carlsonlab


module load Python-GPU/3.7.6

python3 main_2.py \
    --data_path /hpc/group/carlsonlab/zdc6/data/lemmatized/ \
    --dump_path /work/zdc6/wildfire/results/