#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

/home/stu5/anaconda3/envs/lightning/bin/python src/train.py trainer.max_epochs=100

