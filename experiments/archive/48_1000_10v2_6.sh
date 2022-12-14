#!/bin/bash

#SBATCH --output=./slurm48.out
#SBATCH --error=./slurm48.err
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=carlsonlab

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
	--l1_reg 1000 \
	--init_kld 0.000001 \
	--klds_epochs 10 \
	--mse_weight 100 \
    --weighted True \
	--epochs 400 \
	--pred_dropout 0.5 \
	--results_path /hpc/group/carlsonlab/zdc6/sPFA/results/exp48/
