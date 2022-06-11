#!bin/bash


#SBATCH --output=../results/test/slurm.out
#SBATCH --error=../results/test/slurm.err

module load Python/3.8.1

python3 s_pfa.py \
	--lemmatized_path /hpc/group/carlsonlab/zdc6/wildfire/data/lemmatized/ \
	--data_path /hpc/group/carlsonlab/zdc6/sPFA/data/1000_1_by_file/ \
	--tweet_sample_count 1 \
	--min_df 300 \
	--max_df .01 \
	--train_cities portland phoenix \
	--test_cities raleigh \
	--num_components 10 \
	--l1_reg 0.1 \
	--nmf_max_iter 10 \
	--nmf_tol 0.01 \
	--pretrain_epochs 2 \
	--epochs 11 \
	--init_kld 0.000001 \
	--end_kld 0.00001 \
	--klds_epochs 2 \
	--log_level DEBUG \
	--results_path /hpc/group/carlsonlab/zdc6/sPFA/results/test/