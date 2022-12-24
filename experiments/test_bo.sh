#!/bin/bash

# mamba activate bo

python optimize_model.py \
    --data_path ~/Desktop/twitter/sPFA/data/raleigh/ \
    --dump_path ~/Desktop/twitter/sPFA/results/debug/ \
    --data_dump_path ~/Desktop/twitter/sPFA/results/ \
    --save_every 1 \
    --use_lds true \
    --model deep_encoder