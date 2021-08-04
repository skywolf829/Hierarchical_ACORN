#!/bin/sh
#cd /lus/theta-fs0/projects/DL4VIS/Hierarchical_ACORN

#python3 Code/train.py --save_name Cat_16GPUs --use_residual false --log_img true --train_distributed true --gpus_per_node 8 --num_nodes 2

python -u Code/train.py --use_residual true --FC_size_exp_start 1 --FC_size_exp_grow 0 \
--octree_depth_end 2 --save_name Sinc_2models_fc1 --target_signal TrainingData/3D_toydata.h5 \
--num_channels 1 --mode 3D --error_bound 100

python -u Code/train.py --use_residual true --FC_size_exp_start 2 --FC_size_exp_grow 0 \
--octree_depth_end 2 --save_name Sinc_2models_fc2 --target_signal TrainingData/3D_toydata.h5 \
--num_channels 1 --mode 3D --error_bound 100

python -u Code/train.py --use_residual true --FC_size_exp_start 4 --FC_size_exp_grow 0 \
--octree_depth_end 2 --save_name Sinc_2models_fc4 --target_signal TrainingData/3D_toydata.h5 \
--num_channels 1 --mode 3D --error_bound 100

python -u Code/train.py --use_residual true --FC_size_exp_start 8 --FC_size_exp_grow 0 \
--octree_depth_end 2 --save_name Sinc_2models_fc8 --target_signal TrainingData/3D_toydata.h5 \
--num_channels 1 --mode 3D --error_bound 100