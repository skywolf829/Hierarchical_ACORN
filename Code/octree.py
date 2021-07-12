import torch
from typing import List, Tuple
import os

file_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(file_folder_path, "..")

input_folder = os.path.join(project_folder_path, "TrainingData")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

class OctreeNode:
    def __init__(self, node_list, 
    shape : List[int],
    pos : List[int],
    depth : int, index : int):
        self.node_list = node_list
        self.shape = shape
        self.pos : List[int] = pos
        self.depth : int = depth
        self.index : int = index
        self.last_loss : float = 0
        self.parent = None
        self.children = []
        self.bounding_box = []
        for i in range(len(pos)):
            dim_start = pos[i]
            dim_end = pos[i] + shape[2+i]
            dim_start *= (2/node_list.full_shape[2+i])
            dim_end *= (2/node_list.full_shape[2+i])
            dim_start -= 1
            dim_end -= 1
            self.bounding_box.append(dim_start)
            self.bounding_box.append(dim_end)

    def __str__(self) -> str:
        return "{ data_shape: " + str(self.shape) + ", " + \
        "depth: " + str(self.depth) + ", " + \
        "index: " + str(self.index) + "}" 

    def split(self):
        nodes = []
        nodes_above = sum([(2**(len(self.shape)-2))**i for i in range(0, self.depth+1)])
        nodes_here = sum([(2**(len(self.shape)-2))**i for i in range(0, self.depth)])
        k = 0
        for dim_1_start, dim_1_size in [(0, int(self.shape[2]/2)), (int(self.shape[2]/2), self.shape[2] - int(self.shape[2]/2))]:
            for dim_2_start, dim_2_size in [(0, int(self.shape[3]/2)), (int(self.shape[3]/2), self.shape[3] - int(self.shape[3]/2))]:
                if(len(self.shape) == 5):
                    for dim_3_start, dim_3_size in [(0, int(self.shape[4]/2)), (int(self.shape[4]/2), self.shape[4] - int(self.shape[4]/2))]:
                        n_quad = OctreeNode(
                            self.node_list,
                            [
                                self.shape[0], self.shape[1], 
                                dim_1_size, dim_2_size, dim_3_size
                            ],
                            [
                                self.pos[0]+dim_1_start, 
                                self.pos[1]+dim_2_start,
                                self.pos[2]+dim_3_start
                            ],
                            self.depth+1,
                            nodes_above + (self.index-nodes_here)*(2**(len(self.shape)-2)) + k
                        )
                        nodes.append(n_quad)
                        k += 1     
                else:
                    n_quad = OctreeNode(
                        self.node_list,
                        [
                            self.shape[0], self.shape[1], 
                            dim_1_size, dim_2_size
                        ],
                        [
                            self.pos[0]+dim_1_start, 
                            self.pos[1]+dim_2_start,
                        ],
                        self.depth+1,
                        nodes_above + (self.index-nodes_here)*(2**(len(self.shape)-2)) + k
                    )
                    nodes.append(n_quad)
                    k += 1       
        return nodes

    def data(self, data):
        if(len(self.shape) == 4):
            return data[:,:,
                self.pos[0]:self.pos[0]+self.shape[2],
                self.pos[1]:self.pos[1]+self.shape[3]]
        else:
            return data[:,:,
                self.pos[0]:self.pos[0]+self.shape[2],
                self.pos[1]:self.pos[1]+self.shape[3],
                self.pos[2]:self.pos[2]+self.shape[4]]

    def size(self) -> float:
        return (self.data.element_size() * self.data.numel()) / 1024.0

class OctreeNodeList:
    def __init__(self, full_shape):
        self.full_shape = full_shape

        root = OctreeNode(self, 
            full_shape, [0, 0] if len(full_shape) == 4 else [0, 0, 0], 0, 0)

        self.depth_to_nodes : dict[int, List] = {0: { 0: root } }

    def append(self, n : OctreeNode):
        if(n.depth not in self.depth_to_nodes):
            self.depth_to_nodes[n.depth] = {}
        self.depth_to_nodes[n.depth][n.index] = n        

    def remove(self, n : OctreeNode) -> bool:
        found : bool = False
        if(n.depth in self.depth_to_nodes):
            i : int = 0
            if(n.index in self.depth_to_nodes[n.depth]):
                del self.depth_to_nodes[n.depth][n.index]
                found = True
        return found
   
    def get_octree_block_img(self, device="cpu"):
        base = torch.ones(self.full_shape, dtype=torch.float32, device=device)
        color_to_fill = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(2)
        for k in self.depth_to_nodes.keys():
            for block in self.depth_to_nodes[k].values():
                base[:,:,block.pos[0]:block.pos[0]+block.shape[2],
                    block.pos[1]] = color_to_fill

                base[:,:,block.pos[0],
                    block.pos[1]:block.pos[1]+block.shape[3]] = color_to_fill

                base[:,:,block.pos[0]:block.pos[0]+block.shape[2],
                    block.pos[1]+block.shape[3]-1] = color_to_fill

                base[:,:,block.pos[0]+block.shape[2]-1,
                    block.pos[1]:block.pos[1]+block.shape[3]] = color_to_fill
        return base

    def split_all_at_depth(self, d):
        for block in self.depth_to_nodes[d].values():            
            split_nodes = block.split()
            for j in range(len(split_nodes)):
                split_nodes[j].parent = block
                block.children.append(split_nodes[j])
                self.append(split_nodes[j])

    def split_from_error_max_depth(self, reconstructed, item, error_func, max_error):
        max_depth = self.max_depth()
        '''
        for i in range(len(self.depth_to_nodes[max_depth])):
            if self.depth_to_nodes[max_depth][i].error > max_error:
                split_nodes = self.depth_to_nodes[max_depth][i].split()
                for j in range(len(split_nodes)):
                    self.append(split_nodes[j])
        '''
        for block in self.depth_to_nodes[max_depth].values():
            split_nodes = block.split()
            for b in split_nodes:
                if error_func(b.data(reconstructed), b.data(item)) > max_error:
                    b.parent = block
                    block.children.append(b)
                    self.append(b)

    def blocks_to_positions(self, blocks):
        block_positions = []
        for i in range(len(blocks)):
            p = []
            for j in range(len(blocks[i].pos)):
                p_i = blocks[i].pos[j] / self.full_shape[2+j]
                p_i = p_i - 0.5
                p_i = p_i * 2
                p_i = p_i + (1/(2**blocks[i].depth))
                p.append(p_i)
            block_positions.append(p)
        return block_positions

    def depth_to_blocks_and_block_positions(self, depth_level, rank=0, num_splits=1):
        blocks = list(self.depth_to_nodes[depth_level].values())[rank::num_splits]
        if(rank > len(blocks)):
            blocks = []
        block_positions = self.blocks_to_positions(blocks)
        
        return blocks, block_positions

    '''
    def index_to_bounding_box(self, index, depth):
        bounding_box = []
        box_pos = self.depth_to_nodes[depth][index].pos
        box_shape = list(self.depth_to_nodes[depth][index].shape[2:])
        for i in range(len(box_pos)):
            dim_start = box_pos[i]
            dim_end = box_pos[i] + box_shape[i]
            dim_start *= (2/self.full_shape[2+i])
            dim_end *= (2/self.full_shape[2+i])
            dim_start -= 1
            dim_end -= 1
            bounding_box.append(dim_start)
            bounding_box.append(dim_end)
            
        return bounding_box
    '''

    def global_to_local(self, global_coords, index, depth):
        local_coords = global_coords.clone()
        bounding_box = self.depth_to_nodes[depth][index].bounding_box
        for i in range(0, len(bounding_box), 2):
            dim_width = bounding_box[i+1] - bounding_box[i]
            local_coords[...,int(i/2)] -= bounding_box[i]
            local_coords[...,int(i/2)] *= (2/dim_width)
            local_coords[...,int(i/2)] -= 1
        return local_coords

    def global_to_local_batch(self, global_coords, depth):
        local_coords = global_coords.clone()

        # shift to [0, 2]
        local_coords += 1
        # mod by block relative size
        local_coords %= 2**(1-depth)
        # scale each block between 0-2
        local_coords *= 2**depth
        # translate block back to [-1, 1]
        local_coords -= 1

        return local_coords

    def delete_depth_level(self, depth_level):
        del self.depth_to_nodes[depth_level]

    def max_depth(self):
        return max(list(self.depth_to_nodes.keys()))

    def min_depth(self):
        return min(list(self.depth_to_nodes.keys()))

    def save(self, opt):
        folder_to_save_in = os.path.join(save_folder, opt['save_name'])
        if(not os.path.exists(folder_to_save_in)):
            os.makedirs(folder_to_save_in)
        print("Saving octree to %s" % (folder_to_save_in))

        torch.save(self, os.path.join(folder_to_save_in, "octree.data"))
    
    def load(location):
        return torch.load(os.path.join(location, "octree.data"))