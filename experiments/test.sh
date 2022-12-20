#!/bin/bash

# mamba activate test

python main_2.py \
    --data_path ~/Desktop/twitter/sPFA/data/raleigh/ \
    --dump_path ~/Desktop/twitter/sPFA/results/ \
    --data_dump_path ~/Desktop/twitter/sPFA/results/ \
    --model base \
    --prior_mean -1 \
    --DEBUG true \