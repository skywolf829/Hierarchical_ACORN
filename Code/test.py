from math import log10

from numpy.core.shape_base import block
from utility_functions import PSNR, PSNRfromL1, local_to_global, make_coord, ssim3D, str2bool, ssim, PSNRfromMSE
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim as optim
import os
from models import load_model, save_model
import numpy as np
from octree import OctreeNodeList
from options import *
from models import HierarchicalACORN, PositionalEncoding
import argparse
from pytorch_memlab import LineProfiler, MemReporter, profile
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
from torch.multiprocessing import spawn
from torch.distributed import new_group, barrier, group, broadcast
import h5py
import socket
from netCDF4 import Dataset
import imageio

def output_netCDF(model, item, opt):
    
    with torch.no_grad():      
        if('2D' in opt['mode']):
            sample_points = make_coord(item.shape[2:], opt['device'], 
                flatten=False).flatten(0, -2).unsqueeze(0).unsqueeze(0).contiguous()
        else:
            sample_points = make_coord(item.shape[2:], opt['device'], 
                flatten=False).flatten(0, -2).unsqueeze(0).unsqueeze(0).unsqueeze(0).contiguous()
        reconstructed = model.forward_global_positions(sample_points).detach()

        reconstructed = reconstructed.reshape(item.shape[0:2] + tuple(item.shape[2:]))  
        if(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "recon.nc"))):
            os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "recon.nc"))
        if(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "GT.nc"))):
            os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "GT.nc"))
        if(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "tree.nc"))):
            os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "tree.nc"))

        p = PSNR(reconstructed, item, torch.tensor([1.0], dtype=torch.float32, device=opt['device']))
        print("PSNR: %0.04f" % p)

        if(opt['mode'] == '3D'):
            rootgrp = Dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "recon.nc"), "w", format="NETCDF4")
            rootgrp2 = Dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "GT.nc"), "w", format="NETCDF4")
            rootgrp.createDimension("x")
            rootgrp.createDimension("y")
            rootgrp.createDimension("z")
            rootgrp2.createDimension("x")
            rootgrp2.createDimension("y")
            rootgrp2.createDimension("z")
            for chan_num in range(reconstructed.shape[1]):
                dim_i = rootgrp.createVariable('channel_'+str(chan_num), np.float32, ("x","y","z"))
                dim_i2 = rootgrp2.createVariable('channel_'+str(chan_num), np.float32, ("x","y","z"))
                dim_i[:] = reconstructed[0,chan_num].cpu().numpy()
                dim_i2[:] = item[0,chan_num].cpu().numpy()

        else:
            rootgrp = Dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "recon.nc"), "w", format="NETCDF4")
            rootgrp2 = Dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "GT.nc"), "w", format="NETCDF4")
            rootgrp.createDimension("x")
            rootgrp.createDimension("y")
            rootgrp2.createDimension("x")
            rootgrp2.createDimension("y")
            for chan_num in range(reconstructed.shape[1]):
                dim_i = rootgrp.createVariable('channel_'+str(chan_num), np.float32, ("x","y"))
                dim_i2 = rootgrp2.createVariable('channel_'+str(chan_num), np.float32, ("x","y"))
                dim_i[:] = reconstructed[0,chan_num].cpu().numpy()
                dim_i2[:] = item[0,chan_num].cpu().numpy()
            rootgrp

        octree_blocks = model.octree.get_octree_block_img(opt['num_channels'], opt['device'])
                                        
        if('3D' in opt['mode']):
            octree_grp = Dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "tree.nc"), "w", format="NETCDF4")
            octree_grp.createDimension("x")
            octree_grp.createDimension("y")
            octree_grp.createDimension("z")
            dim_i = octree_grp.createVariable('blocks'+str(chan_num), np.float32, ("x","y","z"))
            dim_i[:] = octree_blocks[0,0].cpu().numpy()
        else:
            octree_grp = Dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "tree.nc"), "w", format="NETCDF4")
            octree_grp.createDimension("x")
            octree_grp.createDimension("y")
            dim_i = octree_grp.createVariable('blocks'+str(chan_num), np.float32, ("x","y"))
            dim_i[:] = octree_blocks[0,0].cpu().numpy()

def output_img(model, item, opt):
    
    with torch.no_grad():      
        if('2D' in opt['mode']):
            sample_points = make_coord(item.shape[2:], opt['device'], 
                flatten=False).flatten(0, -2).unsqueeze(0).unsqueeze(0).contiguous()
        else:
            sample_points = make_coord(item.shape[2:], opt['device'], 
                flatten=False).flatten(0, -2).unsqueeze(0).unsqueeze(0).unsqueeze(0).contiguous()

        t = time.time()
        reconstructed = model.forward_global_positions(sample_points).detach()
        print("Reconstruction time: %0.02f" % (time.time() - t))
        reconstructed = reconstructed.reshape(item.shape[0:2] + tuple(item.shape[2:])).clamp_(0, 1)
        
        p = PSNR(reconstructed, item, torch.tensor([1.0], dtype=torch.float32, device=opt['device']))
        print("PSNR: %0.04f" % p)

        if(opt['mode'] == '3D'):
            rootgrp = Dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "recon.nc"), "w", format="NETCDF4")
            rootgrp2 = Dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "GT.nc"), "w", format="NETCDF4")
            rootgrp.createDimension("x")
            rootgrp.createDimension("y")
            rootgrp.createDimension("z")
            rootgrp2.createDimension("x")
            rootgrp2.createDimension("y")
            rootgrp2.createDimension("z")
            for chan_num in range(reconstructed.shape[1]):
                dim_i = rootgrp.createVariable('channel_'+str(chan_num), np.float32, ("x","y","z"))
                dim_i2 = rootgrp2.createVariable('channel_'+str(chan_num), np.float32, ("x","y","z"))
                dim_i[:] = reconstructed[0,chan_num].cpu().numpy()
                dim_i2[:] = item[0,chan_num].cpu().numpy()

        else:
            img = reconstructed.cpu()[0].permute(1, 2, 0).numpy()
            imageio.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "reconstructed.jpg"), img)

        octree_blocks = model.octree.get_octree_block_img(opt['num_channels'], opt['device'])
                                        
        if('3D' in opt['mode']):
            octree_grp = Dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "tree.nc"), "w", format="NETCDF4")
            octree_grp.createDimension("x")
            octree_grp.createDimension("y")
            octree_grp.createDimension("z")
            dim_i = octree_grp.createVariable('blocks'+str(chan_num), np.float32, ("x","y","z"))
            dim_i[:] = octree_blocks[0,0].cpu().numpy()
        else:
            img = (reconstructed*octree_blocks).cpu()[0].permute(1, 2, 0).numpy()
            imageio.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'SavedModels',opt['save_name'], "reconstructed_blocks.jpg"), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create netCDF output from a trained model')

    parser.add_argument('--device',default="cuda:0",type=str,help='Device to use')
    parser.add_argument('--load_from',default="Temp", type=str,help='Load a model to continue training')

    args = vars(parser.parse_args())

    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    opt = load_options(os.path.join(save_folder, args['load_from']))
    opt['device'] = args['device']
    model = load_model(opt, args['device'])
    for i in range(len(model.models)):
        model.models[i] = model.models[i].to(opt['device'])
    item = h5py.File(os.path.join(project_folder_path, opt['target_signal']), 'r')['data']
    item = torch.tensor(item).unsqueeze(0).to(opt['device'])

    #output_netCDF(model, item, opt)
    output_img(model, item, opt)