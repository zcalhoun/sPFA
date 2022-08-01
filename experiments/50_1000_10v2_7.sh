#!/bin/bash

#SBATCH --output=./slurm50.out
#SBATCH --error=./slurm50.err
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

singularity exec --nv -B /hpc/group/carlsonlab/zdc6/ ~/wildfires/wildfire-tweets.sif python3 main.py \
	--lemmatized_path /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	--data_path /hpc/group/carlsonlab/zdc6/sPFA/data/1000_10_by_file_v2/ \
	--tweet_sample_count 10 \
	--min_df 300 \
	--max_df .01 \
	--train_cities chicago raleigh los\ angeles orange san\ francisco dallas \
	--test_cities phoenix seattle new\ york \
	--nmf_max_iter 300 \
	--num_components 200 \
	--l1_reg 100 \
	--init_kld 0.000001 \
	--klds_epochs 10 \
	--mse_weight 100 \
    --weighted True \
	--nmf_tol 1e-2 \
	--epochs 400 \
	--pretrain_lr 0.0000001 \
	--pred_dropout 0.5 \
	--results_path /hpc/group/carlsonlab/zdc6/sPFA/results/exp50/
