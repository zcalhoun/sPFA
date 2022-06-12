#!/bin/bash

#SBATCH --output=./vol_slurm.out
#SBATCH --error=./vol_slurm.err
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=carlsonlab

singularity exec --nv -B /hpc/group/carlsonlab/zdc6/ ~/wildfires/wildfire-tweets.sif python3 s_pfa.py \
	--lemmatized_path /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	--data_path /hpc/group/carlsonlab/zdc6/sPFA/data/1000_10_by_vol/ \
	--tweet_sample_count 10 \
	--sample_method by_volume \
	--min_df 300 \
	--max_df .01 \
	--train_cities chicago dallas los\ angeles new\ york portland san\ francisco phoenix \
	--test_cities raleigh seattle orange \
	--nmf_max_iter 300 \
	--l1_reg 0.1 \
	--init_kld 0.000001 \
	--klds_epochs 100 \
	--log_level DEBUG \
	--results_path /hpc/group/carlsonlab/zdc6/sPFA/results/exp2/
