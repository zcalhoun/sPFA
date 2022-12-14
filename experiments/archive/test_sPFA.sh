#!bin/bash

python s_pfa.py \
	--lemmatized_path ~/research/wildfire/data/lemmatized/ \
	--data_path ./data/ \
	--tweet_sample_count 1 \
	--min_df 3 \
	--max_df .1 \
	--train_cities portland \
	--test_cities portland \
	--num_components 10 \
	--l1_reg 0.1 \
	--nmf_init random \
	--nmf_max_iter 10 \
	--nmf_tol 0.01 \
	--pretrain_epochs 2 \
	--epochs 11 \
	--init_kld 0.000001 \
	--end_kld 0.00001 \
	--klds_epochs 2 \
	--log_level DEBUG \
	--results_path ./results/test/