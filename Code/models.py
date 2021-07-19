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
import time
from torch.multiprocessing import Pool, spawn

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
    model.octree.save(opt)
    save_options(opt, folder_to_save_in)

def load_model(opt, device):
    model = HierarchicalACORN(opt)
    
    folder_to_load_from = os.path.join(save_folder, opt['save_name'])
    if not os.path.exists(folder_to_load_from):
        print("%s doesn't exist, load failed" % folder_to_load_from)
        return

    
    if os.path.exists(os.path.join(folder_to_load_from, "octree.data")):
        model.octree = OctreeNodeList.load(folder_to_load_from)
        print("Successfully loaded octree")
    else:
        print("Warning: no octree associated with model")

    for _ in range(model.octree.max_depth() - model.octree.min_depth()):
        model.add_model(torch.tensor([1.0], dtype=torch.float32, device=opt['device']))
    
    model.to(device)

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
        self.last_layer_output = opt['feat_grid_channels']*opt['feat_grid_y']*opt['feat_grid_x']
        if "3D" in opt['mode']:
            self.last_layer_output *= opt['feat_grid_z']
        
        self.feat_grid_shape = [1, self.opt['feat_grid_channels'], 
            self.opt['feat_grid_y'], self.opt['feat_grid_x']]        
        if "3D" in self.opt['mode']:
            self.feat_grid_shape.insert(2, self.opt['feat_grid_z'])

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
            nn.Linear(opt['feat_grid_channels'], 64),
            nn.ReLU(),
            nn.Linear(64, opt['num_channels'])
        )

        self.vol2FC = VolumeToFC(opt)
        self.FC2vol = FCToVolume(opt)
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
            b_vals = self.vol2FC(b_vals)
            b_val_shape = list(b_vals.shape)
            b_vals = b_vals.flatten(0, -2)
            batch_vals = []
            batch_start = 0
            while(batch_start < b_vals.shape[0]):
                batch_size = min(self.opt['local_queries_per_block']*self.opt['local_queries_per_block']*self.opt['max_blocks_per_iter'], 
                    b_vals.shape[0]-batch_start)
                batch_vals.append(self.feature_decoder(b_vals[batch_start:batch_start+batch_size]))
                batch_start += batch_size

            b_vals = torch.cat(batch_vals)
            b_val_shape[-1] = b_vals.shape[-1]
            b_vals = b_vals.reshape(b_val_shape)
            b_vals = self.FC2vol(b_vals)

            all_block_vals.append(b_vals * mult)
        
        return all_block_vals

class HierarchicalACORN(nn.Module):
    def __init__(self, opt):
        super(HierarchicalACORN, self).__init__()        
        self.opt = opt
        self.models = nn.ModuleList([ACORN(int(2**(opt['FC_size_exp_start'])), opt)])
        self.register_buffer("RMSE", torch.tensor([1], dtype=torch.float32, device=opt['device']))
        self.residual = None
        self.octree : OctreeNodeList = None
        self.pe = PositionalEncoding(opt)
        
    def init_octree(self, data_shape):
        self.octree = OctreeNodeList(data_shape)
        for i in range(self.opt['octree_depth_start']):
            self.octree.split_all_at_depth(i)
            self.octree.delete_depth_level(i)

    def add_model(self, error):
        self.models.append(ACORN(int(2**(len(self.models)*self.opt['FC_size_exp_grow']+self.opt['FC_size_exp_start'])), 
            self.opt))
        self.RMSE = torch.cat([self.RMSE, error])

    def forward(self, blocks, block_positions, local_positions, depth=None, parent_mapping=None, out=None):
        if(depth is None):
            depth = self.octree.max_depth()
        model_no = depth - self.opt['octree_depth_start']
        print("model_no " + str(model_no))
        print("Global block positions: " + str(block_positions))
        encoded_positions = self.pe(block_positions)
        feat_grids = self.models[model_no].feature_encoder(encoded_positions)

        print("feat_grids 1 shape: " + str(feat_grids.shape))
        self.models[model_no].feat_grid_shape[0] = feat_grids.shape[0]
        feat_grids = feat_grids.reshape(self.models[model_no].feat_grid_shape)
        print("feat_grids 2 shape: " + str(feat_grids.shape))

        # If the local positions are a list, that means each block has an unequal
        #   number of query locations and it can't be batched.
        if(isinstance(local_positions, list)):
            feats = []
            for i in range(len(blocks)):
                print("local_positions[i]: "+str(local_positions[i].shape))
                feat = F.grid_sample(feat_grids[i:i+1], local_positions[i], mode='bilinear', align_corners=False)
                feat = self.models[model_no].vol2FC(feat)
                # Guaranteed that we will already have an "out" variable populated
                for child_ind in parent_mapping[blocks[i].index]:
                    out[child_ind:child_ind+1] += self.models[model_no].FC2vol(
                        self.models[model_no].feature_decoder(
                            feat[:,:,parent_mapping[blocks[i].index][child_ind][0]:parent_mapping[blocks[i].index][child_ind][1],:]
                            )
                        )
        else:
            # Batch computation of the local queries            
            if(out is None):
                print("local_positions: "+str(local_positions.shape))
                feats = F.grid_sample(feat_grids, local_positions, mode="bilinear", align_corners=False)
                feats = self.models[model_no].vol2FC(feats)            
                out = self.models[model_no].FC2vol(self.models[model_no].feature_decoder(feats))
            else:
                print(parent_mapping)
                for i in range(len(blocks)):
                    print("local_positions[i:i+1]: "+str(local_positions[i:i+1].shape))
                    feat = F.grid_sample(feat_grids[i:i+1], local_positions[i:i+1], mode='bilinear', align_corners=False)
                    feat = self.models[model_no].vol2FC(feat)
                    print("feat shape: " + str(feat.shape))
                    # Guaranteed that we will already have an "out" variable populated
                    for child_ind in parent_mapping[blocks[i].index].keys():
                        out2 = self.models[model_no].FC2vol(
                            self.models[model_no].feature_decoder(
                                feat[:,:,parent_mapping[blocks[i].index][child_ind][0]:parent_mapping[blocks[i].index][child_ind][1],:]
                                )
                            )   
                        print("out2 shape: " + str(out2.shape))
                        out[child_ind:child_ind+1] += out2
        
        print("out shape: " + str(out.shape))

        if(model_no > 0):
            # Parent block and parent block local positions, index -> block and index -> tensor of local positions
            parent_blocks = {}                
            parent_block_local_positions = {}

            # Mapping so that parent block queries can be added the the correct block
            #   maps parent_ind -> out_ind -> output_points[start:end]
            if(parent_mapping is None):
                parent_mapping = {}
            

            for i in range(len(blocks)):
                # Get quadrand/octant and the parent index for the current block
                magic_num = 4 if '2D' in self.opt['mode'] else 8
                nodes_above = sum([magic_num**k for k in range(0, blocks[i].depth)])
                quad = (blocks[i].index - nodes_above) % magic_num
                parent_index = int((blocks[i].index - nodes_above) / magic_num) + sum([magic_num**k for k in range(0, blocks[i].depth-1)])
                
                #print("block ind: " + str(blocks[i].index))
                #print("block depth: " + str(blocks[i].depth))
                #print("quad: " + str(quad))
                #print("parent_index: " + str(parent_index))
                # Add parent index to parent blocks if necessary
                if(parent_index not in parent_blocks):
                    parent_blocks[parent_index] = self.octree.depth_to_nodes[depth-1][parent_index]

                # Copy the locations for this block and shift it to the correct local position for
                #   the parent
                parent_block_local_position = local_positions[i:i+1].clone() * 0.5
                print("parent_block_local_position shape: " + str(parent_block_local_position.shape))
                if('2D' in self.opt['mode']):
                    parent_block_local_position[...,1] += ((quad % 2) - 0.5)                        
                    parent_block_local_position[...,0] += (int(quad / 2) - 0.5)
                else:               
                    parent_block_local_position[...,0] += (int(quad / 4) - 0.5)
                    parent_block_local_position[...,1] += (int((quad % 4) / 2) - 0.5)                        
                    parent_block_local_position[...,2] += (quad % 2 - 0.5)  

                # Add the local positions to the local position dict by directly adding it,
                #   or concatenating it with the local positions that already exist to query

                if(parent_index in parent_block_local_positions.keys()):
                    
                    if(blocks[i].index in parent_mapping.keys()):
                        start = parent_block_local_positions[parent_index].shape[2]

                        parent_block_local_positions[parent_index] = \
                            torch.cat([parent_block_local_positions[parent_index], parent_block_local_position], 2)
                            
                        for out_ind in parent_mapping[blocks[i].index]:
                            offset = parent_mapping[blocks[i].index][out_ind]
                            parent_mapping[parent_index][out_ind] = [start + offset[0],
                                start + offset[1]]
                    else:
                        parent_mapping[parent_index][blocks[i].index] = [parent_block_local_positions[parent_index].shape[2], 
                            parent_block_local_positions[parent_index].shape[2] + parent_block_local_position.shape[2]]
                        parent_block_local_positions[parent_index] = \
                            torch.cat([parent_block_local_positions[parent_index], parent_block_local_position], 2)
                else:
                    if(blocks[i].index in parent_mapping.keys()):
                        parent_block_local_positions[parent_index] = parent_block_local_position
                        
                        parent_mapping[parent_index] = {}
                        start = 0
                        for out_ind in parent_mapping[blocks[i].index]:
                            offset = parent_mapping[blocks[i].index][out_ind]
                            parent_mapping[parent_index][out_ind] = [start, start + offset[1]-offset[0]]
                            start += offset[1]-offset[0]
                    else:
                        parent_block_local_positions[parent_index] = parent_block_local_position
                        parent_mapping[parent_index] = {}
                        parent_mapping[parent_index][blocks[i].index] = [0, parent_block_local_position.shape[2]]
            

            
            # Keep track of if all blocks have the same # local queries for efficiency down the road
            same_shape = True
            s = None
            
            parent_block_local_positions = list(parent_block_local_positions.values())
            for l_p in parent_block_local_positions:
                # Update the shape if it is not set
                if(s is None):
                    s = list(l_p.shape)
                # Check if they are the same shape
                same_shape = same_shape and s == list(l_p.shape)
            
            # If they are the same shape (i.e. 10000 queries per parent block), we can 
            #   combine them into a single tensor for batch computation. Otherwise,
            #   we have to iterate over each block for values.
            if(same_shape):
                parent_block_local_positions = torch.cat(parent_block_local_positions, 0)
                print("All blocks had same # query points")
                print("parent_block_local_positions shape" + str(parent_block_local_positions.shape))
            else:
                print("Some blocks were unequal with query points")

            # Update values for the next depth level of the ACORN hierarchy.
            parent_blocks = list(parent_blocks.values())
            parent_block_positions = torch.tensor(self.octree.blocks_to_positions(parent_blocks), device=self.opt['device'])                
            print("parent_block_positions shape: " + str(parent_block_positions.shape))               
            
            out = self.forward(parent_blocks, parent_block_positions, parent_block_local_positions, depth-1, parent_mapping, out)    

        return out

    def calculate_block_errors(self, reconstructed, item, error_func):
        blocks, block_positions = self.octree.depth_to_blocks_and_block_positions(
                self.octree.max_depth())
        block_positions = torch.tensor(block_positions, 
                device=self.opt['device'])
        for block in blocks:
            error = error_func(block.data(reconstructed), block.data(item))
            block.error = error.item()
            print("Block error: " + str(block.error))

    def count_parameters(self): 
        num = 0
        for i in range(len(self.models)):
            num_this_model = sum(p.numel() for p in self.models[i].parameters())
            num += num_this_model
        return num

    def get_full_img(self):
        blocks, block_positions = self.octree.depth_to_blocks_and_block_positions(
                self.octree.max_depth())
        block_positions = torch.tensor(block_positions, 
                device=self.opt['device'])
        out = torch.zeros(self.octree.full_shape, dtype=torch.float32, device=self.opt['device'])
        out += self.residual
        block_outputs = self.models[-1].forward_batch(block_positions, blocks, self.RMSE[-1])
        for b in range(len(blocks)):
            if('2D' in self.opt['mode']):
                out[:,:,
                    blocks[b].pos[0]:\
                        blocks[b].pos[0]+block_outputs[b].shape[2],
                    blocks[b].pos[1]:\
                        blocks[b].pos[1]+block_outputs[b].shape[3]] += block_outputs[b]
            else:
                out[:,:,
                    blocks[b].pos[0]:\
                        blocks[b].pos[0]+block_outputs[b].shape[2],
                    blocks[b].pos[1]:\
                        blocks[b].pos[1]+block_outputs[b].shape[3],                    
                    blocks[b].pos[2]:\
                        blocks[b].pos[2]+block_outputs[b].shape[4]] += block_outputs[b]
        return out

    def block_index_to_global_indices_mapping(self, global_positions):
        print(global_positions.shape)
        index_to_global_positions_indices = {}
        for depth in range(self.octree.min_depth(), self.octree.max_depth()+1):
            for block in self.octree.depth_to_nodes[depth].values():
                block_bbox = block.bounding_box
                mask = global_positions[0,0,...,0] >= block_bbox[0]
                mask = torch.logical_and(mask,global_positions[0,0,...,0] < block_bbox[1])
                for i in range(1, global_positions.shape[-1]):                    
                    mask = torch.logical_and(mask,global_positions[0,0,...,i] >= block_bbox[i*2])
                    mask = torch.logical_and(mask,global_positions[0,0,...,i] <  block_bbox[i*2+1])
                index_to_global_positions_indices[block.index] = mask

        return index_to_global_positions_indices
    
    
    #@profile
    def forward_global_positions(self, global_positions, index_to_global_positions_indices=None, 
    depth_start=None, depth_end=None, local_positions=None, block_start=None):
        if depth_start is None:
            depth_start = self.opt['octree_depth_start']
        if self.opt['use_residual'] and self.residual is not None:
            depth_start = self.octree.max_depth()
        if depth_end is None:
            depth_end = self.octree.max_depth()+1
            
        pre_global_index_mapping = time.time()
        if(index_to_global_positions_indices is None):
            index_to_global_positions_indices = self.block_index_to_global_indices_mapping(global_positions)
        #print("Global index mapping time %f" % (time.time()-pre_global_index_mapping))

        if('2D' in self.opt['mode']):
            out_shape = [global_positions.shape[0], self.opt['num_channels'], 1, global_positions.shape[-2]]
        else:            
            out_shape = [global_positions.shape[0], self.opt['num_channels'], 1, 1, global_positions.shape[-2]]
        
        out = torch.zeros(out_shape, device=self.opt['device'])
        if(self.opt['use_residual'] and self.residual is not None):
            out += F.grid_sample(self.residual, 
                    global_positions.flip(-1), mode='bilinear', align_corners=False).detach()


        start_time = time.time()
        for depth in range(depth_start, depth_end):
            model_start_time = time.time()
            model_no = depth - self.opt['octree_depth_start']

            blocks, block_positions = self.octree.depth_to_blocks_and_block_positions(depth)

            block_positions = torch.tensor(block_positions, 
                    device=self.opt['device'])

            encoded_positions = self.pe(block_positions)
            feat_grids = self.models[model_no].feature_encoder(encoded_positions)
            self.models[model_no].feat_grid_shape[0] = feat_grids.shape[0]
            feat_grids = feat_grids.reshape(self.models[model_no].feat_grid_shape)
            
            if(depth == depth_end-1 and local_positions is not None):
                feat = F.grid_sample(feat_grids[block_start:block_start+local_positions.shape[0]], 
                    local_positions, mode='bilinear', align_corners=False)
                feat = self.models[model_no].vol2FC(feat)
                out_temp = self.models[model_no].feature_decoder(feat)
                out_temp = out_temp.flatten(0, -2).unsqueeze(0).unsqueeze(0)
                out_temp = self.models[model_no].FC2vol(out_temp)                
                out += out_temp
            else:  
                local_positions_at_depth = self.octree.global_to_local_batch(
                    global_positions, depth)
                for i in range(len(blocks)):
                    local_positions_in_block = local_positions_at_depth[...,index_to_global_positions_indices[blocks[i].index],:]
                    if(local_positions_in_block.shape[-2] > 0):
                        feat = F.grid_sample(feat_grids[i:i+1], local_positions_in_block, mode='bilinear', align_corners=False)
                        feat = self.models[model_no].vol2FC(feat)
                        out[...,index_to_global_positions_indices[blocks[i].index]] += self.models[model_no].FC2vol(self.models[model_no].feature_decoder(feat))
                
            #print("Model at depth %i took %f seconds" % (depth, time.time()-model_start_time))
        
        #print("Feedforward for %i points took %f seconds" % (global_positions.shape[-2], time.time()-start_time))
        return out
                
    def get_full_img_no_residual(self):  
        out = torch.zeros(self.octree.full_shape, dtype=torch.float32, device=self.opt['device'])
        model_no = 0
        for depth in range(self.octree.max_depth(), self.octree.max_depth()+1):
            blocks, block_positions = self.octree.depth_to_blocks_and_block_positions(
                depth)
            block_positions = torch.tensor(block_positions, 
                    device=self.opt['device'])
            block_outputs = self.models[model_no].forward_batch(block_positions, blocks, self.RMSE[model_no])
            for b in range(len(blocks)):
                if('2D' in self.opt['mode']):
                    out[:,:,
                        blocks[b].pos[0]:\
                            blocks[b].pos[0]+block_outputs[b].shape[2],
                        blocks[b].pos[1]:\
                            blocks[b].pos[1]+block_outputs[b].shape[3]] += block_outputs[b]
                else:
                    out[:,:,
                        blocks[b].pos[0]:\
                            blocks[b].pos[0]+block_outputs[b].shape[2],
                        blocks[b].pos[1]:\
                            blocks[b].pos[1]+block_outputs[b].shape[3],                    
                        blocks[b].pos[2]:\
                            blocks[b].pos[2]+block_outputs[b].shape[4]] += block_outputs[b]
            model_no += 1
        return out