from octree import OctreeNodeList
from matplotlib.pyplot import xcorr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.activation import LeakyReLU, SiLU
from utility_functions import  create_batchnorm_layer, create_conv_layer, weights_init, make_coord, \
    bilinear_interpolate, trilinear_interpolate
import os
from options import save_options
from einops.layers.torch import Rearrange
from math import pi
from pytorch_memlab import LineProfiler, MemReporter, profile, profile_every
from torch.utils.checkpoint import checkpoint_sequential, checkpoint

file_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(file_folder_path, "..")

input_folder = os.path.join(project_folder_path, "TrainingData")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")


def save_model(model, opt):
    folder_to_save_in = os.path.join(save_folder, opt['save_name'])
    if(not os.path.exists(folder_to_save_in)):
        os.makedirs(folder_to_save_in)
    print("Saving model to %s" % (folder_to_save_in))
    
    model_state = model.state_dict()
    torch.save(model_state, os.path.join(folder_to_save_in, "model.ckpt"))

    save_options(opt, folder_to_save_in)

def load_model(opt, device):
    model = HierarchicalACORN(opt)
    folder_to_load_from = os.path.join(save_folder, opt['save_name'])

    if not os.path.exists(folder_to_load_from):
        print("%s doesn't exist, load failed" % folder_to_load_from)
        return

    from collections import OrderedDict
    if os.path.exists(os.path.join(folder_to_load_from, "model.ckpt")):
        model_params = torch.load(os.path.join(folder_to_load_from, "model.ckpt"),
            map_location=device)
        model.load_state_dict(model_params)
        print("Successfully loaded model")
    else:
        print("Warning: model.ckpt doesn't exists - can't load these model parameters")
   
    return model

class PositionalEncoding(nn.Module):
    def __init__(self, opt):
        super(PositionalEncoding, self).__init__()        
        self.opt = opt
        self.n_dims = 2 if "2D" in opt['mode'] else 3
        self.L = opt['num_positional_encoding_terms']
        self.L_terms = torch.arange(0, opt['num_positional_encoding_terms'], 
            device=opt['device'], dtype=torch.float32).repeat_interleave(2*self.n_dims)
        self.L_terms = torch.pow(2, self.L_terms) * pi
        #self.phase_shift = torch.rand(self.L_terms.shape, 
            #device=opt['device'])

    def forward(self, locations):
        repeats = len(list(locations.shape)) * [1]
        repeats[-1] = self.L*2
        locations = locations.repeat(repeats)
        
        locations = locations * self.L_terms# + self.phase_shift
        if(self.n_dims == 2):
            locations[..., 0::4] = torch.sin(locations[..., 0::4])
            locations[..., 1::4] = torch.sin(locations[..., 1::4])
            locations[..., 2::4] = torch.cos(locations[..., 2::4])
            locations[..., 3::4] = torch.cos(locations[..., 3::4])
        else:
            locations[..., 0::6] = torch.sin(locations[..., 0::6])
            locations[..., 1::6] = torch.sin(locations[..., 1::6])
            locations[..., 2::6] = torch.sin(locations[..., 2::6])
            locations[..., 3::6] = torch.cos(locations[..., 3::6])
            locations[..., 4::6] = torch.cos(locations[..., 4::6])
            locations[..., 5::6] = torch.cos(locations[..., 5::6])
        return locations

class VolumeToFC(nn.Module):
    def __init__(self, opt):
        super(VolumeToFC, self).__init__()
        self.mode = opt['mode']

    def forward(self, x):
        if "2D" in self.mode:
            x = x.permute(0, 2, 3, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
        return x

class FCToVolume(nn.Module):
    def __init__(self, opt):
        super(FCToVolume, self).__init__()
        self.mode = opt['mode'] 

    def forward(self, x):
        if "2D" in self.mode:
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.permute(0, 4, 1, 2, 3)
        return x

class Reshaper(nn.Module):
    def __init__(self, s):
        super(Reshaper, self).__init__()
        self.s = s

    def forward(self, x):
        return x.reshape(self.s)

class ACORN(nn.Module):
    def __init__(self, nodes_per_layer, opt):
        super(ACORN, self).__init__()        
        self.opt = opt

        self.n_dims = 2 if "2D" in opt['mode'] else 3
        self.last_layer_output = opt['feat_grid_channels']*opt['feat_grid_x']*opt['feat_grid_y']
        if "3D" in opt['mode']:
            self.last_layer_output *= opt['feat_grid_z']
        
        self.feat_grid_shape = [1, self.opt['feat_grid_channels'], 
            self.opt['feat_grid_x'], self.opt['feat_grid_y']]        
        if "3D" in self.opt['mode']:
            self.feat_grid_shape.append(self.opt['feat_grid_z'])

        self.positional_encoder = PositionalEncoding(opt)

        self.feature_encoder = nn.Sequential(
            #nn.Linear(self.n_dims, nodes_per_layer),
            nn.Linear(2*self.n_dims*opt['num_positional_encoding_terms'], nodes_per_layer),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.ReLU(),
            nn.Linear(nodes_per_layer, self.last_layer_output)
        )
        
        self.feature_decoder = nn.Sequential(
            VolumeToFC(opt),
            nn.Linear(opt['feat_grid_channels'], 64),
            nn.ReLU(),
            nn.Linear(64, opt['num_channels']),
            FCToVolume(opt)
        )

        self.apply(weights_init)
    
    def forward(self, global_coordinates, block_shape, mult = 1.0):
        # Todo
        return global_coordinates
        
    def forward_batch(self, global_coordinates, blocks, mult = 1.0):
        global_coordinates = self.positional_encoder(global_coordinates)
        # global coordinates will be B x n_dims
        all_block_vals = []
        shape = [global_coordinates.shape[0], self.opt['feat_grid_channels'], 
            self.opt['feat_grid_x'], self.opt['feat_grid_y']]
        if "3D" in self.opt['mode']:
            shape.append(self.opt['feat_grid_z'])

        vals = self.feature_encoder(global_coordinates)
        vals = vals.reshape(shape)
        for i in range(global_coordinates.shape[0]):

            b_vals = F.interpolate(vals[i:i+1], size=blocks[i].shape[2:], 
                mode='bilinear' if "2D" in self.opt['mode'] else "trilinear", 
                align_corners=False)

            b_vals = self.feature_decoder(b_vals)

            all_block_vals.append(b_vals * mult)
        
        return all_block_vals

class HierarchicalACORN(nn.Module):
    def __init__(self, opt):
        super(HierarchicalACORN, self).__init__()        
        self.opt = opt
        self.models = nn.ModuleList([ACORN(int(2**(6.54)), opt)])
        self.errors = [1]
        self.residual = None
    
    def init_octree(self, data_shape):
        self.octree = OctreeNodeList(data_shape)
        
    def add_model(self, opt):
        self.models.append(ACORN(int(2**(len(self.models)*0.5+6)), opt))

    def forward(self, block, block_position, model_no):        
        block_output = self.models[model_no](block_position, block, self.errors[-1])
        return block_output

    def calculate_block_errors(self, octree, error_func):
        blocks, block_positions = octree.depth_to_blocks_and_block_positions(
                octree.max_depth)
        block_positions = torch.tensor(block_positions, 
                device=self.opt['device'])
        block_outputs = self.models[-1].forward_batch(block_positions, blocks, self.errors[-1])
        for b in range(len(blocks)):
            if('2D' in self.opt['mode']):
                block_outputs[b] += self.residual[:,:,
                    blocks[b].start_position[0]:\
                        blocks[b].start_position[0]+block_outputs[b].shape[2],
                    blocks[b].start_position[1]:\
                        blocks[b].start_position[1]+block_outputs[b].shape[3]]
            else:
               block_outputs[b] +=  self.residual[:,:,
                    blocks[b].start_position[0]:\
                        blocks[b].start_position[0]+block_outputs[b].shape[2],
                    blocks[b].start_position[1]:\
                        blocks[b].start_position[1]+block_outputs[b].shape[3],                    
                    blocks[b].start_position[2]:\
                        blocks[b].start_position[2]+block_outputs[b].shape[4]]
            error = error_func(block_outputs[b], blocks[b].data())
            blocks[b].error = error.item()

    def get_full_img(self, octree):
        blocks, block_positions = octree.depth_to_blocks_and_block_positions(
                octree.max_depth)
        block_positions = torch.tensor(block_positions, 
                device=self.opt['device'])
        out = torch.zeros_like(octree.data, device=self.opt['device'])
        out += self.residual
        block_outputs = self.models[-1].forward_batch(block_positions, blocks, self.errors[-1])
        for b in range(len(blocks)):
            if('2D' in self.opt['mode']):
                out[:,:,
                    blocks[b].start_position[0]:\
                        blocks[b].start_position[0]+block_outputs[b].shape[2],
                    blocks[b].start_position[1]:\
                        blocks[b].start_position[1]+block_outputs[b].shape[3]] += block_outputs[b]
            else:
               out[:,:,
                    blocks[b].start_position[0]:\
                        blocks[b].start_position[0]+block_outputs[b].shape[2],
                    blocks[b].start_position[1]:\
                        blocks[b].start_position[1]+block_outputs[b].shape[3],                    
                    blocks[b].start_position[2]:\
                        blocks[b].start_position[2]+block_outputs[b].shape[4]] += block_outputs[b]
        return out

    def get_full_img_no_residual(self, octree):
        blocks, block_positions = octree.depth_to_blocks_and_block_positions(
                octree.max_depth)
        block_positions = torch.tensor(block_positions, 
                device=self.opt['device'])
        out = torch.zeros_like(octree.data, device=self.opt['device'])
        out += self.residual
        block_outputs = self.models[-1].forward_batch(block_positions, blocks, self.errors[-1])
        for b in range(len(blocks)):
            if('2D' in self.opt['mode']):
                out[:,:,
                    blocks[b].start_position[0]:\
                        blocks[b].start_position[0]+block_outputs[b].shape[2],
                    blocks[b].start_position[1]:\
                        blocks[b].start_position[1]+block_outputs[b].shape[3]] += block_outputs[b]
            else:
               out[:,:,
                    blocks[b].start_position[0]:\
                        blocks[b].start_position[0]+block_outputs[b].shape[2],
                    blocks[b].start_position[1]:\
                        blocks[b].start_position[1]+block_outputs[b].shape[3],                    
                    blocks[b].start_position[2]:\
                        blocks[b].start_position[2]+block_outputs[b].shape[4]] += block_outputs[b]
        return out