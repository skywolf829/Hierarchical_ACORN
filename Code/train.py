from math import log10

from numpy.core.shape_base import block
from utility_functions import PSNR, str2bool, ssim, PSNRfromMSE
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
from models import save_model
import numpy as np
from octree import OctreeNodeList
from options import *
from models import HierarchicalACORN, PositionalEncoding
import argparse
from pytorch_memlab import LineProfiler, MemReporter, profile
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
import h5py

class Trainer():
    def __init__(self, opt):
        self.opt = opt
        torch.manual_seed(0b10101010101010101010101010101010)

    def train(self, rank, model, item):
        if(self.opt['train_distributed']):
            self.opt['device'] = "cuda:" + str(rank)
            dist.init_process_group(                                   
                backend='nccl',                                         
                init_method='env://',                                   
                world_size = self.opt['num_nodes'] * self.opt['gpus_per_node'],                              
                rank=rank                                               
            )
            model = model.to(rank)
            model = DDP(model, device_ids=[rank])
            if(rank == 0): 
                print("Training in parallel")
        else:
            print("Training on " + self.opt['device'])
            model = model.to(self.opt['device'])

        model_optim = optim.Adam(model.models[-1].parameters(), lr=self.opt["lr"], 
            betas=(self.opt["beta_1"],self.opt["beta_2"]))

        #optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
        #    milestones=[self.opt['epochs']/5, 
        #    2*self.opt['epochs']/5, 
        #    3*self.opt['epochs']/5, 
        #    4*self.opt['epochs']/5],gamma=self.opt['gamma'])

        if(not self.opt['train_distributed'] or rank == 0):
            writer = SummaryWriter(os.path.join('tensorboard',self.opt['save_name']))
        start_time = time.time()

        loss = nn.MSELoss().to(self.opt["device"])
        step = 1
        item = item.to(self.opt['device'])

        
        target_PSNR = self.opt['error_bound']
        MSE_limit = 10 ** ((-1*target_PSNR + 20*log10(1.0))/10)
        
        model.init_octree(item.shape)
        model.residual = torch.zeros_like(item, device=self.opt['device']).detach()

        pe = PositionalEncoding(self.opt)
            
        for model_num in range(self.opt['octree_depth_end'] - self.opt['octree_depth_start']):
            print("Model %i, total parameter count: %i" % (model_num, model.count_parameters()))
            blocks, block_positions = model.octree.depth_to_blocks_and_block_positions(
                        model.octree.max_depth(), 
                        rank, 
                        self.opt['gpus_per_node']*self.opt['num_nodes'] if self.opt['train_distributed'] else 1)
            block_positions = torch.tensor(block_positions, 
                    device=self.opt['device'])

            for epoch in range(self.opt['epoch'], self.opt['epochs']):
                self.opt["epoch"] = epoch            
                model.zero_grad()           

                block_error_sum = 0                
                
                b = 0
                while b < len(blocks):
                    blocks_this_iter = min(self.opt['max_blocks_per_iter'], len(blocks)-b)

                    if('2D' in self.opt['mode']):
                        local_positions = torch.rand([blocks_this_iter,self.opt['local_queries_per_block'], 
                            self.opt['local_queries_per_block'], 2], device=self.opt['device']) * 2 - 1
                        #print("Local positions shape: " + str(local_positions.shape))
                        '''
                        feat_grids = model.models[-1].feature_encoder(pe(block_positions[b:b+blocks_this_iter]))
                        model.models[-1].feat_grid_shape[0] = feat_grids.shape[0]
                        feat_grids = feat_grids.reshape(model.models[-1].feat_grid_shape)
                        feats = F.grid_sample(feat_grids[b:b+blocks_this_iter], local_positions, mode="bilinear", align_corners=False)
                        feats = feats.permute(0, 2, 3, 1)
                        block_output = model.models[-1].feature_decoder(feats)                       
                        res = F.grid_sample(model.residual.expand([blocks_this_iter, -1, -1, -1]), global_positions, mode="bilinear", align_corners=False)
                        block_output = block_output.permute(0, 3, 1, 2) + res
                        '''
                        block_output = model(blocks, block_positions, local_positions)
                        #print("block_output shape: " + str(block_output.shape))
                        shapes = torch.tensor([block.shape for block in blocks[b:b+blocks_this_iter]], device=self.opt['device']).unsqueeze(1).unsqueeze(1)
                        poses = torch.tensor([block.pos for block in blocks[b:b+blocks_this_iter]], device=self.opt['device']).unsqueeze(1).unsqueeze(1)

                        global_positions = (local_positions.clone() + 1) / 2
                        global_positions[...,0] *= (shapes[...,2]/item.shape[2])
                        global_positions[...,0] += (poses[...,1]/item.shape[2])
                        global_positions[...,0] *= 2
                        global_positions[...,0] -= 1
                        
                        global_positions[...,1] *= (shapes[...,3]/item.shape[3])
                        global_positions[...,1] += (poses[...,0]/item.shape[3])
                        global_positions[...,1] *= 2
                        global_positions[...,1] -= 1
                        block_item = F.grid_sample(item.expand([blocks_this_iter, -1, -1, -1]), global_positions, mode='bilinear', align_corners=False)
                        #print("block_item shape: " + str(block_item.shape))
                    
                    block_error = loss(block_output,block_item) * (blocks_this_iter/len(blocks))
                    blocks[b].last_loss = block_error.detach().item() 
                    block_error.backward(retain_graph=True)
                    block_error_sum += block_error.detach()

                    b += blocks_this_iter
                    

                #block_error_sum *= (1/len(blocks))
                #block_error_sum.backward()
                model_optim.step()
                #optim_scheduler.step()
                
                if(step % self.opt['log_every'] == 0 and (not self.opt['train_distributed'] or rank == 0)):
                    with torch.no_grad():    
                        reconstructed = model.get_full_img()    
                        octree_blocks = model.octree.get_octree_block_img(self.opt['device'])            
                        psnr = PSNR(reconstructed, item, torch.tensor(1.0))
                        s = ssim(reconstructed, item)
                        print("Iteration %i, MSE: %0.06f, PSNR (dB): %0.02f, SSIM: %0.03f" % \
                            (epoch, block_error_sum.item(), psnr.item(), s.item()))
                        writer.add_scalar('Training PSNR', PSNRfromMSE(block_error_sum, torch.tensor(1.0, device=self.opt['device'])), step)
                        writer.add_scalar('PSNR', psnr.item(), step)
                        writer.add_scalar('SSIM', s.item(), step)         
                        if(len(model.models) > 1):
                            writer.add_image("Network"+str(len(model.models)-1)+"residual", 
                                ((reconstructed-model.residual)[0]+0.5).clamp_(0, 1), step)
                        writer.add_image("reconstruction", reconstructed[0].clamp_(0, 1), step)
                        writer.add_image("reconstruction_blocks", reconstructed[0].clamp_(0, 1)*octree_blocks[0], step)

                elif(step % 5 == 0 and (not self.opt['train_distributed'] or rank == 0)):
                    print("Iteration %i, MSE: %0.06f" % \
                            (epoch, block_error_sum.item()))
                    writer.add_scalar('Training PSNR', PSNRfromMSE(block_error_sum, torch.tensor(1.0, device=self.opt['device'])), step)
                step += 1
            
                if(epoch % self.opt['save_every'] == 0 and (not self.opt['train_distributed'] or rank == 0)):
                    save_model(model, self.opt)
                    print("Saved model and octree")
                    
            if(not self.opt['train_distributed'] or rank == 0):
                with torch.no_grad():    
                    reconstructed = model.get_full_img()    
                    octree_blocks = model.octree.get_octree_block_img(self.opt['device'])          
                    psnr = PSNR(reconstructed, item, torch.tensor(1.0))
                    s = ssim(reconstructed, item)
                    print("Iteration %i, MSE: %0.06f, PSNR (dB): %0.02f, SSIM: %0.02f" % \
                        (epoch, block_error_sum.item(), psnr.item(), s.item()))
                    writer.add_scalar('Training PSNR', PSNRfromMSE(block_error_sum, torch.tensor(1.0, device=self.opt['device'])), step)
                    writer.add_scalar('PSNR', psnr.item(), step)
                    writer.add_scalar('SSIM', s.item(), step)         
                    if(len(model.models) > 1):
                        writer.add_image("Network"+str(len(model.models)-1)+"residual", 
                            ((reconstructed-model.residual)[0]+0.5).clamp_(0, 1), step)
                    writer.add_image("reconstruction_blocks", reconstructed[0].clamp_(0, 1)*octree_blocks[0], step)
                    writer.add_image("reconstruction", reconstructed[0].clamp_(0, 1), step)


            if((not self.opt['train_distributed'] or rank == 0) and \
                model_num < self.opt['octree_depth_end'] - self.opt['octree_depth_start']-1):
                print("Total parameter count: %i" % model.count_parameters())   
                print("Adding higher-resolution model")   
                with torch.no_grad():                                    
                    model.residual = model.get_full_img().detach()
                    model.calculate_block_errors(loss, item)
                model.add_model(torch.tensor([1.0], dtype=torch.float32, device=self.opt['device']))
                model.to(self.opt['device'])

                print("Last error: " + str(block_error_sum.item()))
                #model.errors.append(block_error_sum.item()**0.5)

                if(self.opt['error_bound_split']):
                    #model.octree.split_from_error_max_depth(MSE_limit)
                    model.octree.split_from_error_max_depth(item, model.residual, loss, MSE_limit)
                    
                else:
                    model.octree.split_all_at_depth(model.octree.max_depth())

                model_optim = optim.Adam(model.models[-1].parameters(), lr=self.opt["lr"], 
                    betas=(self.opt["beta_1"],self.opt["beta_2"]))
                #optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
                #    milestones=[self.opt['epochs']/5, 
                #    2*self.opt['epochs']/5, 
                #    3*self.opt['epochs']/5, 
                #    4*self.opt['epochs']/5],gamma=self.opt['gamma'])
                for param in model.models[-2].parameters():
                    param.requires_grad = False
                self.opt['epoch'] = 0
                torch.cuda.empty_cache()
        
        if(not self.opt['train_distributed'] or rank == 0):
            print("Total parameter count: %i" % model.count_parameters())   
            end_time = time.time()
            total_time = end_time - start_time
            print("Time to train: %0.01f minutes" % (total_time/60))
            save_model(model, self.opt)
            print("Saved model")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on an input that is 2D')

    parser.add_argument('--mode',default=None,type=str,help='The type of input - 2D, 3D')
    parser.add_argument('--target_signal',default=None,type=str,help='File to train on')
    parser.add_argument('--save_folder',default=None,type=str,help='The folder to save the models folder into')
    parser.add_argument('--save_name',default=None,type=str,help='The name for the folder to save the model')
    parser.add_argument('--device',default=None,type=str,help='Device to use')
    
    parser.add_argument('--num_channels',default=None,type=int,help='Number of channels in the data')
    parser.add_argument('--feat_grid_channels',default=None,type=int,help='Channels in the feature grid')
    parser.add_argument('--feat_grid_x',default=None,type=int,help='X resolution of feature grid')
    parser.add_argument('--feat_grid_y',default=None,type=int,help='Y resolution of feature grid')
    parser.add_argument('--feat_grid_z',default=None,type=int,help='Z resolution of feature grid (if 3D)')
    parser.add_argument('--local_queries_per_block',default=None,type=int,help='num queries per block while training')
    parser.add_argument('--max_blocks_per_iter',default=None,type=int,help='max blocks in a batch per iter')
    parser.add_argument('--num_positional_encoding_terms',default=None,type=int,help='Number of positional encoding terms')
    parser.add_argument('--FC_size_exp_start',default=None,type=float,help='How large the FC layers start')
    parser.add_argument('--FC_size_exp_grow',default=None,type=float,help='How much the FC layers grow deeper in the octree')
    
    parser.add_argument('--octree_depth_start',default=None,type=int,help='How deep to start the octree, inclusive')    
    parser.add_argument('--octree_depth_end',default=None,type=int,help='How deep to end the octree, exclusive')
    parser.add_argument('--error_bound_split',default=None,type=str2bool,help='Whether to split based on error')
    parser.add_argument('--error_bound',default=None,type=float,help='The target PSNR error')

    parser.add_argument('--train_distributed',type=str2bool,default=None, help='Use distributed training')
    parser.add_argument('--gpus_per_node',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--num_nodes',default=None, type=int,help='Whether or not to save discriminators')

    parser.add_argument('--epochs',default=None, type=int,help='Number of epochs to use')
    parser.add_argument('--lr',default=None, type=float,help='Learning rate for the generator')    
    parser.add_argument('--beta_1',default=None, type=float,help='')
    parser.add_argument('--beta_2',default=None, type=float,help='')

    parser.add_argument('--load_from',default=None, type=str,help='Load a model to continue training')
    parser.add_argument('--save_every',default=None, type=int,help='How often to save during training')
    parser.add_argument('--log_every',default=None, type=int,help='How often to log during training')

    args = vars(parser.parse_args())

    file_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(file_folder_path, "..")

    input_folder = os.path.join(project_folder_path, "TrainingData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    prof = LineProfiler()
    prof.enable()

    if(args['load_from'] is None):
        opt = Options.get_default()
        for k in args.keys():
            if args[k] is not None:
                opt[k] = args[k]
        item = h5py.File(os.path.join(project_folder_path, opt['target_signal']), 'r')['data']
        item = torch.tensor(item).unsqueeze(0)
        model = HierarchicalACORN(opt)

    trainer = Trainer(opt)

    if(not opt['train_distributed']):
        trainer.train(0, model, item)
    else:
        #to be implemented
        print("not implemented yet")
    print(prof.display())
    prof.disable()