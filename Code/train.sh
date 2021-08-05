#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/Hierarchical_ACORN

#python3 Code/train.py --save_name Cat_16GPUs --use_residual false --log_img true --train_distributed true --gpus_per_node 8 --num_nodes 2

python -u Code/train.py --use_residual true --FC_size_exp_start 1 --FC_size_exp_grow 0 \
--octree_depth_end 2 --save_name pluto_2models_size1 --target_signal TrainingData/pluto.h5 \
--num_channels 3 --mode 2D --local_queries_per_iter 1000000

python -u Code/train.py --use_residual true --FC_size_exp_start 1 --FC_size_exp_grow 1 \
--octree_depth_end 2 --save_name pluto_2models_size2 --target_signal TrainingData/pluto.h5 \
--num_channels 3 --mode 2D --local_queries_per_iter 1000000

python -u Code/train.py --use_residual true --FC_size_exp_start 1 --FC_size_exp_grow 2 \
--octree_depth_end 2 --save_name pluto_2models_size3 --target_signal TrainingData/pluto.h5 \
--num_channels 3 --mode 2D --local_queries_per_iter 1000000

python -u Code/train.py --use_residual true --FC_size_exp_start 1 --FC_size_exp_grow 3 \
--octree_depth_end 2 --save_name pluto_2models_size4 --target_signal TrainingData/pluto.h5 \
--num_channels 3 --mode 2D --local_queries_per_iter 1000000