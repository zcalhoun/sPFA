#!/bin/bash

#SBATCH --output=./slurm28.out
#SBATCH --error=./slurm28.err
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

singularity exec --nv -B /hpc/group/carlsonlab/zdc6/ ~/wildfires/wildfire-tweets.sif python3 s_pfa.py \
	--lemmatized_path /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	--data_path /hpc/group/carlsonlab/zdc6/sPFA/data/1000_10_by_file/ \
	--tweet_sample_count 10 \
	--min_df 300 \
	--max_df .01 \
	--train_cities chicago dallas los\ angeles new\ york portland san\ francisco phoenix \
	--test_cities raleigh seattle orange \
	--nmf_max_iter 300 \
	--num_components 400 \
	--l1_reg 50 \
	--init_kld 0.000001 \
	--klds_epochs 10 \
	--mse_weight 100 \
	--epochs 400 \
	--pretrain_lr 0.00001 \
	--results_path /hpc/group/carlsonlab/zdc6/sPFA/results/exp28/
