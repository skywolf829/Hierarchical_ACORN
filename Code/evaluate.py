from math import log10
from utility_functions import PSNR, str2bool, ssim
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from models import load_model, save_model
import numpy as np
from octree import OctreeNodeList
from options import *
from models import HierarchicalACORN, PositionalEncoding
import h5py
from pytorch_memlab import LineProfiler, MemReporter, profile
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
import imageio

if __name__ == '__main__':
    
    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    load_from = "Snick_4levels_dynamic_error"

    opt = load_options(os.path.join(save_folder, load_from))
    opt["device"] = "cuda:0"
    opt["save_name"] = load_from

    item = h5py.File(os.path.join(project_folder_path, opt['target_signal']), 'r')['data']
    item = torch.tensor(item).unsqueeze(0).to(opt['device'])

    model = load_model(opt, opt['device'])

    img = model.get_full_img_no_residual()
    octree_blocks = model.octree.get_octree_block_img()

    img *= octree_blocks.to(opt['device'])

    print(img.shape)

    img = img.detach()[0].permute(1, 2, 0).cpu().numpy()
    imageio.imwrite("test.png", img)

