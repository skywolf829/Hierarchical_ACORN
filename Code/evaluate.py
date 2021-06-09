from math import log10
from utility_functions import PSNR, bilinear_interpolate, str2bool, ssim
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
import cv2 as cv

if __name__ == '__main__':
    
    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    load_from = "Cat_ACORN_fixed"#"Snick_5levels_dynamic_error"

    opt = load_options(os.path.join(save_folder, load_from))
    opt["device"] = "cuda:0"
    opt["save_name"] = load_from

    item = h5py.File(os.path.join(project_folder_path, opt['target_signal']), 'r')['data']
    item = torch.tensor(item).unsqueeze(0).to(opt['device'])

    model = load_model(opt, opt['device'])
    with torch.no_grad():
        reconstruction = model.get_full_img_no_residual()    
        imageio.imwrite("recon.png", reconstruction.detach()[0].permute(1, 2, 0).cpu().numpy())

        octree_blocks = model.octree.get_octree_block_img()

        img = reconstruction * octree_blocks.to(opt['device'])

        print(img.shape)

        img = img.detach()[0].permute(1, 2, 0).cpu().numpy()
        imageio.imwrite("test.png", img)

        
        reconstruction = reconstruction.detach()[0].permute(1, 2, 0).cpu().numpy()
        grad_x = cv.Sobel(reconstruction, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(reconstruction, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        #dst = cv.Laplacian(reconstruction, 1, ksize=3)

        print(grad_x.shape)
        print(grad_y.shape)
        imageio.imwrite("x_grad.png", grad_x)
        imageio.imwrite("y_grad.png", grad_y)
        imageio.imwrite("xy_grad.png", grad_x + grad_y)
    
    
    