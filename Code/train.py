from math import log10
from utility_functions import PSNR, str2bool, ssim
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

    #@profile
    def train(self, model, item):
        
        print("Training on " + self.opt['device'])

        model = model.to(self.opt['device'])

        model_optim = optim.Adam(model.models[-1].parameters(), lr=self.opt["lr"], 
            betas=(self.opt["beta_1"],self.opt["beta_2"]))

        #optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
        #    milestones=[self.opt['epochs']/5, 
        #    2*self.opt['epochs']/5, 
        #    3*self.opt['epochs']/5, 
        #    4*self.opt['epochs']/5],gamma=self.opt['gamma'])

        writer = SummaryWriter(os.path.join('tensorboard',self.opt['save_name']))
        
        start_time = time.time()

        loss = nn.MSELoss().to(self.opt["device"])
        step = 0
        item = item.to(self.opt['device'])

        
        target_PSNR = 40
        MSE_limit = 10 ** ((-1*target_PSNR + 20*log10(1.0))/10)
        
        model.init_octree(item.shape)
        model.residual = torch.zeros_like(item, device=self.opt['device']).detach()

        pe = PositionalEncoding(opt)
            
        for model_num in range(opt['octree_depth_end'] - opt['octree_depth_start']):
            for epoch in range(self.opt['epoch'], self.opt['epochs']):
                self.opt["epoch"] = epoch            
                model.zero_grad()            

                block_error_sum = 0

                blocks, block_positions = model.octree.depth_to_blocks_and_block_positions(
                        model.octree.max_depth())
                block_positions = torch.tensor(block_positions, 
                        device=self.opt['device'])
                block_positions = pe(block_positions)

                feat_grids = model.models[-1].feature_encoder(block_positions)
                #feat_grids = checkpoint_sequential(model.models[-1].feature_encoder, 8, block_positions)  

                model.models[-1].feat_grid_shape[0] = feat_grids.shape[0]
                feat_grids = feat_grids.reshape(model.models[-1].feat_grid_shape)
                
                for b in range(len(blocks)):
                    #print("Block %i/%i" % (b+1, len(blocks)))
                    block_output = F.interpolate(feat_grids[b:b+1], size=blocks[b].shape[2:], 
                        mode='bilinear' if "2D" in self.opt['mode'] else "trilinear", 
                        align_corners=False)
                    #block_output = checkpoint_sequential(model.models[-1].feature_decoder, 1, block_output)                    
                    block_output = model.models[-1].feature_decoder(block_output)

                    if('2D' in opt['mode']):
                        block_output += model.residual[:,:,
                            blocks[b].pos[0]:blocks[b].pos[0]+block_output.shape[2],
                            blocks[b].pos[1]:blocks[b].pos[1]+block_output.shape[3]].detach()
                        block_item = item[:,:,
                            blocks[b].pos[0]:blocks[b].pos[0]+block_output.shape[2],
                            blocks[b].pos[1]:blocks[b].pos[1]+block_output.shape[3]]
                    else:
                        block_output += model.residual[:,:,
                            blocks[b].pos[0]:blocks[b].pos[0]+block_output.shape[2],
                            blocks[b].pos[1]:blocks[b].pos[1]+block_output.shape[3],
                            blocks[b].pos[2]:blocks[b].pos[2]+block_output.shape[4]].detach()
                        block_item = item[:,:,
                            blocks[b].pos[0]:blocks[b].pos[0]+block_output.shape[2],
                            blocks[b].pos[1]:blocks[b].pos[1]+block_output.shape[3],
                            blocks[b].pos[2]:blocks[b].pos[2]+block_output.shape[4]]

                    block_error = loss(block_output,block_item) * (1/len(blocks))
                    block_error.backward(retain_graph=True)
                    block_error_sum += block_error.detach()
                    

                #block_error_sum *= (1/len(blocks))
                #block_error_sum.backward()
                model_optim.step()
                #optim_scheduler.step()
                
                if(step % 100 == 0):
                    with torch.no_grad():    
                        reconstructed = model.get_full_img()                    
                        psnr = PSNR(reconstructed, item, torch.tensor(1.0))
                        s = ssim(reconstructed, item)
                        print("Iteration %i, MSE: %0.04f, PSNR (dB): %0.02f, SSIM: %0.02f" % \
                            (epoch, block_error_sum.item(), psnr.item(), s.item()))
                        writer.add_scalar('MSE', block_error_sum.item(), step)
                        writer.add_scalar('PSNR', psnr.item(), step)
                        writer.add_scalar('SSIM', s.item(), step)             
                        if(len(model.models) > 1):
                            writer.add_image("Network"+str(len(model.models)-1)+"residual", 
                                ((reconstructed-model.residual)[0]+0.5).clamp_(0, 1), step)
                        writer.add_image("reconstruction", reconstructed[0].clamp_(0, 1), step)
                elif(step % 5 == 0):
                    print("Iteration %i, MSE: %0.04f" % \
                            (epoch, block_error_sum.item()))
                step += 1
            
                if(epoch % self.opt['save_every'] == 0):
                    save_model(model, self.opt)
                    print("Saved model and octree")

            if(model_num < opt['octree_depth_end'] - opt['octree_depth_start']-1):
                print("Adding higher-resolution model")   
                with torch.no_grad():                                    
                    model.residual = model.get_full_img().detach()
                    model.calculate_block_errors(loss, item)
                model.add_model(opt)
                model.to(opt['device'])
                print("Last error: " + str(block_error_sum.item()))
                #model.errors.append(block_error_sum.item()**0.5)
                model.errors.append(1.0)
                #octree.next_depth_level()
                model.octree.split_from_error_max_depth(MSE_limit)
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
    parser.add_argument('--num_positional_encoding_terms',default=None,type=int,help='Number of positional encoding terms')
    
    parser.add_argument('--octree_depth_start',default=None,type=int,help='How deep to start the octree, inclusive')    
    parser.add_argument('--octree_depth_end',default=None,type=int,help='How deep to end the octree, inclusive')

    parser.add_argument('--train_distributed',type=str2bool,default=None, help='Use distributed training')
    parser.add_argument('--gpus_per_node',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--num_nodes',default=None, type=int,help='Whether or not to save discriminators')

    parser.add_argument('--epochs',default=None, type=int,help='Number of epochs to use')
    parser.add_argument('--lr',default=None, type=float,help='Learning rate for the generator')    
    parser.add_argument('--beta_1',default=None, type=float,help='')
    parser.add_argument('--beta_2',default=None, type=float,help='')

    parser.add_argument('--load_from',default=None, type=str,help='Load a model to continue training')
    parser.add_argument('--save_every',default=None, type=int,help='How often to save during training')

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
    trainer.train(model, item)

    print(prof.display())
    prof.disable()