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

    def __str__(self) -> str:
        return "{ data_shape: " + str(self.shape) + ", " + \
        "depth: " + str(self.depth) + ", " + \
        "index: " + str(self.index) + "}" 

    def split(self):
        nodes = []
        k = 0
        for x_quad_start in range(0, self.shape[2], int(self.shape[2]/2)):
            if(x_quad_start == 0):
                x_size = int(self.shape[2]/2)
            else:
                x_size = self.shape[2] - int(self.shape[2]/2)
            for y_quad_start in range(0, self.shape[3], int(self.shape[3]/2)):
                if(y_quad_start == 0):
                    y_size = int(self.shape[3]/2)
                else:
                    y_size = self.shape[3] - int(self.shape[3]/2)
                if(len(self.shape) == 5):
                    for z_quad_start in range(0, self.shape[4], int(self.shape[4]/2)):
                        if(z_quad_start == 0):
                            z_size = int(self.shape[4]/2)
                        else:
                            z_size = self.shape[4] - int(self.shape[4]/2)
                        n_quad = OctreeNode(
                            self.node_list,
                            [
                                self.shape[0], self.shape[1], 
                                x_size, y_size, z_size
                            ],
                            [
                                self.pos[0]+x_quad_start, 
                                self.pos[1]+y_quad_start,
                                self.pos[2]+z_quad_start
                            ],
                            self.depth+1,
                            self.index*(2**(len(self.shape)-2)) + k
                        )
                        nodes.append(n_quad)
                        k += 1     
                else:
                    n_quad = OctreeNode(
                        self.node_list,
                        [
                            self.shape[0], self.shape[1], 
                            x_size, y_size
                        ],
                        [
                            self.pos[0]+x_quad_start, 
                            self.pos[1]+y_quad_start,
                        ],
                        self.depth+1,
                        self.index*(2**(len(self.shape)-2)) + k
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

        self.depth_to_nodes : dict[int, List] = {0: [root]}

    def append(self, n : OctreeNode):
        if(n.depth not in self.depth_to_nodes):
            self.depth_to_nodes[n.depth] = []
        self.depth_to_nodes[n.depth].append(n)        

    def remove(self, n : OctreeNode) -> bool:
        found : bool = False
        if(n.depth in self.depth_to_nodes):
            i : int = 0
            while(i < len(self.depth_to_nodes[n.depth]) and not found):
                if(self.depth_to_nodes[n.depth][i] is n):
                    self.depth_to_nodes[n.depth].pop(i)
                    found = True
                i += 1
        return found
    
    def get_octree_block_img(self, device="cpu"):
        base = torch.ones(self.full_shape, dtype=torch.float32, device=device)
        color_to_fill = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(2)
        for k in self.depth_to_nodes.keys():
            for block in self.depth_to_nodes[k]:
                base[:,:,block.pos[0]:block.pos[0]+block.shape[2],block.pos[1]] = color_to_fill
                base[:,:,block.pos[0],block.pos[1]:block.pos[1]+block.shape[3]] = color_to_fill
                base[:,:,block.pos[0]:block.pos[0]+block.shape[2],block.pos[1]+block.shape[3]-1] = color_to_fill
                base[:,:,block.pos[0]+block.shape[2]-1,block.pos[1]:block.pos[1]+block.shape[3]] = color_to_fill
        return base

    def split_all_at_depth(self, d):
        for i in range(len(self.depth_to_nodes[d])):            
            split_nodes = self.depth_to_nodes[d][i].split()
            for j in range(len(split_nodes)):
                self.append(split_nodes[j])

    def split_from_error_max_depth(self, max_error):
        max_depth = self.max_depth()
        
        for i in range(len(self.depth_to_nodes[max_depth])):
            if self.depth_to_nodes[max_depth][i].error > max_error:
                split_nodes = self.depth_to_nodes[max_depth][i].split()
                for j in range(len(split_nodes)):
                    self.append(split_nodes[j])

    def depth_to_blocks_and_block_positions(self, depth_level, rank=0, num_splits=1):
        blocks = self.depth_to_nodes[depth_level]
        block_positions = []
        for i in range(len(blocks)):
            p = []
            for j in range(len(blocks[i].pos)):
                p_i = blocks[i].pos[j] / self.full_shape[2+j]
                p_i = p_i - 0.5
                p_i = p_i * 2
                p_i = p_i + (1/2**depth_level)
                p.append(p_i)
            block_positions.append(p)
        
        return blocks[rank::num_splits], block_positions[rank::num_splits]

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