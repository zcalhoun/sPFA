#!/bin/bash

#SBATCH --output=./slurm59.out
#SBATCH --error=./slurm59.err
#SBATCH -p carlsonlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=carlsonlab

singularity exec --nv -B /hpc/group/carlsonlab/zdc6/ ~/wildfires/wildfire-tweets.sif python3 main.py \
	--lemmatized_path /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	--data_path /hpc/group/carlsonlab/zdc6/sPFA/data/1000_10_by_file_v2/ \
    --pretrain_checkpoint /hpc/group/carlsonlab/zdc6/sPFA/checkpoints/pretrained/1000_10v2_200comp.pt \
	--tweet_sample_count 10 \
	--min_df 300 \
	--max_df .01 \
	--train_cities chicago raleigh los\ angeles orange san\ francisco dallas \
	--test_cities phoenix seattle new\ york \
	--nmf_max_iter 300 \
    --nmf_tol 0.001 \
	--num_components 200 \
	--l1_reg 1 \
	--init_kld 0.000001 \
    --pretrain_batch_size 32 \
	--klds_epochs 10 \
	--mse_weight 10 \
    --weighted True \
	--epochs 800 \
	--pred_dropout 0.5 \
	--results_path /hpc/group/carlsonlab/zdc6/sPFA/results/exp59/
