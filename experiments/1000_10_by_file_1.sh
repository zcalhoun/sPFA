#!/bin/bash

#SBATCH --output=./slurm.out
#SBATCH --error=./slurm.err
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -p carlsonlab-gpu
#SBATCH --account=carlsonlab

singularity exec -B /hpc/group/carlsonlab/zdc6/ ~/wildfires/wildfire-tweets.sif python3 s_pfa.py \
	--lemmatized_path /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	--data_path /hpc/group/carlsonlab/zdc6/sPFA/data/1000_10_by_file/ \
	--tweet_sample_count 10 \
	--min_df 300 \
	--max_df .01 \
	--train_cities chicago dallas los\ angeles new\ york portland san\ francisco phoenix \
	--test_cities raleigh seattle orange \
	--l1_reg 0.1 \
	--init_kld 0.000001 \
	--klds_epochs 100 \
	--log_level DEBUG \
	--results_path /hpc/group/carlsonlab/zdc6/sPFA/results/exp1/
