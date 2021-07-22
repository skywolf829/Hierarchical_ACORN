from math import log10

from numpy.core.shape_base import block
from utility_functions import PSNR, local_to_global, make_coord, str2bool, ssim, PSNRfromMSE
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
from torch.multiprocessing import spawn
from torch.distributed import new_group, barrier, group, broadcast
import h5py
import socket

class Trainer():
    def __init__(self, opt):
        self.opt = opt

    #@profile
    def train(self, rank, model, item):
        torch.manual_seed(0b10101010101010101010101010101010)
        if(self.opt['train_distributed']):
            node_name = socket.gethostname()
            with open(os.environ['COBALT_NODEFILE'], 'r') as file:
                nodes = file.read().replace('\n', ',')
            nodes = nodes[:len(nodes)-1]
            nodes = nodes.split(',')
            self.opt['node_num'] = nodes.index(node_name)
            rank = rank + self.opt['gpus_per_node']*self.opt['node_num']
            self.opt['device'] = "cuda:" + str(rank)
            model.opt = self.opt
            dist.init_process_group(                                   
                backend='nccl',                                         
                #init_method='env://',                                 
                init_method='file://'+os.getcwd()+"/DistTemp",                                    
                world_size = self.opt['num_nodes'] * self.opt['gpus_per_node'],                              
                rank=rank                                               
            )
            model = model.to(self.opt['device'])
            model.pe = PositionalEncoding(self.opt)
            #model = DDP(model, device_ids=[rank])
            #print("Training in parallel, node " + str(self.opt['node_num']) + " device cuda:" + str(rank))
            # Synchronize all models
            for model_num in range(len(model.models)):
                for param in model.models[model_num].parameters():
                    broadcast(param, 0)
        else:
            print("Training on " + self.opt['device'])
            model = model.to(self.opt['device'])

        

        #optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
        #    milestones=[self.opt['epochs']/5, 
        #    2*self.opt['epochs']/5, 
        #    3*self.opt['epochs']/5, 
        #    4*self.opt['epochs']/5],gamma=self.opt['gamma'])

        if(rank == 0 or not self.opt['train_distributed']):
            writer = SummaryWriter(os.path.join('tensorboard',self.opt['save_name']))
        start_time = time.time()

        loss = nn.MSELoss().to(self.opt["device"])
        step = 0
        item = item.to(self.opt['device'])

        
        target_PSNR = self.opt['error_bound']
        MSE_limit = 10 ** ((-1*target_PSNR + 20*log10(1.0))/10)
        
        model.init_octree(item.shape)
            
        for model_num in range(self.opt['octree_depth_end'] - self.opt['octree_depth_start']):
            if(self.opt['train_distributed']):
                barrier()
            for m_num in range(self.opt['octree_depth_end'] - self.opt['octree_depth_start']):
                for param in model.models[m_num].parameters():
                    param.requires_grad = model_num == m_num

            model_optim = optim.Adam(model.models[model_num].parameters(), lr=self.opt["lr"], 
                betas=(self.opt["beta_1"],self.opt["beta_2"]))

            blocks, block_positions = model.octree.depth_to_blocks_and_block_positions(
                        model.octree.max_depth())
            block_positions = torch.tensor(block_positions, 
                    device=self.opt['device'])
                    
            if(rank == 0 or not self.opt['train_distributed']):
                print("Model %i, total parameter count: %i, num blocks: %i" % 
                    (model_num, model.count_parameters(), len(blocks)))
            if(self.opt['train_distributed']):
                num_blocks = len(blocks)
                if(rank == 0):
                    print("Blocks: " + str(num_blocks))
                if(num_blocks < 
                    self.opt['num_nodes'] * self.opt['gpus_per_node']):
                    g = new_group(list(range(num_blocks)), backend='nccl')
                    stride = num_blocks
                else:
                    g = group.WORLD
                    stride = self.opt['num_nodes'] * self.opt['gpus_per_node']
            else:
                stride = 1    
            
            model_caches = {}
            
            block_error_sum = torch.tensor(0, dtype=torch.float32, device=self.opt['device']) 
            if(rank < len(blocks) or not self.opt['train_distributed']):
                best_MSE = 1.0
                best_MSE_epoch = 0
                early_stop = False
                epoch = self.opt['epoch']
                while epoch < self.opt['epochs'] and not early_stop:
                    self.opt["epoch"] = epoch            
                    model.zero_grad()          
                    
                    block_error_sum = torch.tensor(0, dtype=torch.float32, device=self.opt['device'])
                    b = rank * int(len(blocks)/stride)
                    b_stop = min((rank+1) * int(len(blocks)/stride), len(blocks))
                    if((rank == 0 or not self.opt['train_distributed']) and epoch == 0):
                        writer.add_scalar("num_nodes", len(blocks), model_num)
                    queries = max(int(self.opt['local_queries_per_iter'] / len(blocks)),
                        self.opt['min_queries_per_block'])
                    total_queries = torch.tensor(0, dtype=torch.int, device=self.opt['device'])
                    while b < b_stop:
                        #blocks_this_iter = min(self.opt['max_blocks_per_iter'], b_stop-b)
                        blocks_this_iter = b_stop-b

                        if('2D' in self.opt['mode']):
                            local_positions = torch.rand([blocks_this_iter, 1, 
                                queries, 2], device=self.opt['device']) * 2 - 1
                                
                            shapes = torch.tensor([block.shape for block in blocks[b:b+blocks_this_iter]], device=self.opt['device']).unsqueeze(1).unsqueeze(1)
                            poses = torch.tensor([block.pos for block in blocks[b:b+blocks_this_iter]], device=self.opt['device']).unsqueeze(1).unsqueeze(1)
                            
                            global_positions = local_to_global(local_positions.clone(), shapes, poses, item.shape)                            
                            global_positions = global_positions.flatten(0, -2).unsqueeze(0).unsqueeze(0).contiguous()
                            
                            if((b, blocks_this_iter) not in model_caches.keys()):
                                model_caches[(b, blocks_this_iter)] = model.block_index_to_global_indices_mapping(global_positions)

                            block_output = model.forward_global_positions(global_positions, 
                                index_to_global_positions_indices=model_caches[(b, blocks_this_iter)],
                                local_positions=local_positions, block_start=b)
                            block_item = F.grid_sample(item.expand([-1, -1, -1, -1]), 
                                global_positions.flip(-1), mode='bilinear', align_corners=False)
                        
                        block_error = loss(block_output,block_item) * queries * blocks_this_iter #* (blocks_this_iter/len(blocks))
                        block_error.backward(retain_graph=True)
                        block_error_sum += block_error.detach()
                        total_queries += queries * blocks_this_iter

                        b += blocks_this_iter
                        

                    if self.opt['train_distributed']:
                        # Grad averaging for dist training
                        dist.all_reduce(block_error_sum, op=dist.ReduceOp.SUM, group=g)
                        dist.all_reduce(total_queries, op=dist.ReduceOp.SUM, group=g)
                        for param in model.models[model_num].parameters():
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=g)
                            param.grad.data *= (1/total_queries)
                    else:
                        for param in model.models[model_num].parameters():
                            param.grad.data *= (1/total_queries)
                    block_error_sum /= total_queries
                    model_optim.step()
                    #optim_scheduler.step()
                    
                    if(block_error_sum > best_MSE and best_MSE_epoch < epoch - 100):
                        early_stop = True
                        if(rank == 0 or not self.opt['train_distributed']):
                            print("Stopping early")
                    else:
                        best_MSE = block_error_sum
                        best_MSE_epoch = epoch

                    if(step % self.opt['log_every'] == 0 and 
                        (not self.opt['train_distributed'] or rank == 0)
                        and self.opt['log_img']):
                        self.log_with_image(model, item, block_error_sum, writer, step)

                    elif(step % 5 == 0 and (not self.opt['train_distributed'] or rank == 0)):
                        print("Iteration %i, MSE: %0.06f" % \
                                (epoch, block_error_sum.item()))
                        writer.add_scalar('Training PSNR', PSNRfromMSE(block_error_sum, torch.tensor(1.0, device=self.opt['device'])), step)
                    step += 1
                
                    if(epoch % self.opt['save_every'] == 0 and (not self.opt['train_distributed'] or rank == 0)):
                        save_model(model, self.opt)
                        print("Saved model and octree")
                    epoch += 1
            
            if(self.opt['train_distributed']):
                barrier()  
                # Synchronize all models, whether they were training or not
                for param in model.models[model_num].parameters():
                    broadcast(param, 0)

            if((rank == 0 or not self.opt['train_distributed']) and self.opt['log_img']):
                self.log_with_image(model, item, block_error_sum, writer, step)

            if(model_num < self.opt['octree_depth_end'] - self.opt['octree_depth_start']-1):

                model = model.to(self.opt['device'])
                model.pe = PositionalEncoding(self.opt)

                if(self.opt['use_residual']):
                    with torch.no_grad():   
                        sample_points = make_coord(item.shape[2:], self.opt['device'], 
                                flatten=False).flatten(0, -2).unsqueeze(0).unsqueeze(0).contiguous()       
                        reconstructed = model.forward_global_positions(sample_points)    
                        reconstructed = reconstructed.reshape(item.shape) 
                        model.residual = reconstructed.detach()  
                        model.octree.split_from_error_max_depth(reconstructed, item, loss, MSE_limit)    
                        del reconstructed

                elif(self.opt['error_bound_split']):
                    with torch.no_grad(): 
                        model.octree.split_from_error_max_depth_blockwise(model, item, loss, MSE_limit, self.opt)
                            
                elif(not self.opt['error_bound_split']):
                    model.octree.split_all_at_depth(model.octree.max_depth())


                #optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=model_optim,
                #    milestones=[self.opt['epochs']/5, 
                #    2*self.opt['epochs']/5, 
                #    3*self.opt['epochs']/5, 
                #    4*self.opt['epochs']/5],gamma=self.opt['gamma'])
                self.opt['epoch'] = 0

        if(rank == 0 or not self.opt['train_distributed']):
            print("Total parameter count: %i" % model.count_parameters())   
            end_time = time.time()
            total_time = end_time - start_time
            print("Time to train: %0.01f minutes" % (total_time/60))
            save_model(model, self.opt)
            print("Saved model")

    #@profile
    def log_with_image(self, model, item, block_error_sum, writer, step, img_size = [512, 512]):
        with torch.no_grad():  
            if(self.opt['use_residual']):
                temp_residual = model.residual
                model.residual = None  
            sample_points = make_coord(img_size, self.opt['device'], 
                flatten=False).flatten(0, -2).unsqueeze(0).unsqueeze(0).contiguous()            
            reconstructed = model.forward_global_positions(sample_points).detach()
            
            reconstructed = reconstructed.reshape(item.shape[0:2] + tuple(img_size))     

            if(len(model.models) > 1):
                res = model.forward_global_positions(sample_points, depth_end=model.octree.max_depth())    
                res = res.reshape(item.shape[0:2] + tuple(img_size))
                writer.add_image("Network"+str(len(model.models)-1)+"residual", 
                    ((reconstructed-res)[0]+0.5).clamp(0, 1), step)
            writer.add_image("reconstruction", reconstructed[0].clamp(0, 1), step)
            octree_blocks = F.interpolate(model.octree.get_octree_block_img(self.opt['device']),
                size=img_size, mode="bilinear", align_corners=False)
            writer.add_image("reconstruction_blocks", reconstructed[0].clamp(0, 1)*octree_blocks[0], step)

            if(self.opt['log_psnr']):
                psnr = PSNR(reconstructed, F.interpolate(item, size=img_size, mode='bilinear', align_corners=False), 
                    torch.tensor(1.0)).item()
            else:
                psnr = 0            
            if(self.opt['log_ssim']):
                s = ssim(reconstructed, F.interpolate(item, size=img_size, mode='bilinear', align_corners=False)).item()
            else:
                s = 0

            print("Iteration %i, MSE: %0.06f, PSNR (dB): %0.02f, SSIM: %0.03f" % \
                (step, block_error_sum.item(), psnr, s))
            writer.add_scalar('Training PSNR', PSNRfromMSE(block_error_sum, torch.tensor(1.0, device=self.opt['device'])), step)
            writer.add_scalar('PSNR', psnr, step)
            writer.add_scalar('SSIM', s, step)         
            
            if(self.opt['use_residual']):
                model.residual = temp_residual


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
    parser.add_argument('--local_queries_per_iter',default=None,type=int,help='num queries per iteration while training')
    parser.add_argument('--min_queries_per_block',default=None,type=int,help='min queries per block while training')
    parser.add_argument('--max_blocks_per_iter',default=None,type=int,help='max blocks in a batch per iter')
    parser.add_argument('--num_positional_encoding_terms',default=None,type=int,help='Number of positional encoding terms')
    parser.add_argument('--FC_size_exp_start',default=None,type=float,help='How large the FC layers start')
    parser.add_argument('--FC_size_exp_grow',default=None,type=float,help='How much the FC layers grow deeper in the octree')
    parser.add_argument('--use_residual',default=None,type=str2bool,help='Use a cached residual to accelerate training')
    
    parser.add_argument('--octree_depth_start',default=None,type=int,help='How deep to start the octree, inclusive')    
    parser.add_argument('--octree_depth_end',default=None,type=int,help='How deep to end the octree, exclusive')
    parser.add_argument('--error_bound_split',default=None,type=str2bool,help='Whether to split based on error')
    parser.add_argument('--error_bound',default=None,type=float,help='The target PSNR error')

    parser.add_argument('--train_distributed',type=str2bool,default=None, help='Use distributed training')
    parser.add_argument('--gpus_per_node',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--num_nodes',default=None, type=int,help='Whether or not to save discriminators')
    parser.add_argument('--node_num',default=None, type=int,help='This nodes ID')

    parser.add_argument('--epochs',default=None, type=int,help='Number of epochs to use')
    parser.add_argument('--lr',default=None, type=float,help='Learning rate for the generator')    
    parser.add_argument('--beta_1',default=None, type=float,help='')
    parser.add_argument('--beta_2',default=None, type=float,help='')

    parser.add_argument('--load_from',default=None, type=str,help='Load a model to continue training')
    parser.add_argument('--save_every',default=None, type=int,help='How often to save during training')
    parser.add_argument('--log_every',default=None, type=int,help='How often to log during training')
    parser.add_argument('--log_img',default=None, type=str2bool,help='Log img during training')
    parser.add_argument('--log_ssim',default=None, type=str2bool,help='Log ssim during training')

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
        for model_num in range(opt['octree_depth_end'] - opt['octree_depth_start']):
            model.add_model(torch.tensor([1.0], dtype=torch.float32, device=opt['device']))

    trainer = Trainer(opt)

    if(not opt['train_distributed']):
        trainer.train(0, model, item)
    else:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        spawn(trainer.train, args=(model, item), nprocs=opt['gpus_per_node'])
    print(prof.display())
    prof.disable()