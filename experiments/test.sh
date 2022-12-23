#!/bin/bash

# mamba activate test

python main_2.py \
    --data_path ~/Desktop/twitter/sPFA/data/raleigh/ \
    --dump_path ~/Desktop/twitter/sPFA/results/ \
    --data_dump_path ~/Desktop/twitter/sPFA/results/ \
    --model base \
    --tweets_per_sample 1000 \
    --num_samples_per_day 5 \
    --num_components 268 \
    --prior_mean -7.5 \
    --mse_weight 0.08 \
    --batch_size 256 \
    --end_kld 0.027 \
    --lr 0.000086 \
    --b1 0.731666 \
    --b2 0.712416 \
    --wd 0.000216 \
    --use_lds true \
    --DEBUG true