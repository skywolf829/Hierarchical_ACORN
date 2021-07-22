#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/Hierarchical_ACORN

python3 Code/train.py --train_distributed true --save_name Cat_16GPUs_noresidual --log_img false