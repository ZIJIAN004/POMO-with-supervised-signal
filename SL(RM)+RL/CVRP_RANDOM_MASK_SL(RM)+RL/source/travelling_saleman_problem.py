"""
The MIT License

Copyright (c) 2020 Yeong-Dae Kwon
Copyright (c) 2026 LIU ZIJIAN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import torch
import numpy as np

# For debugging
from IPython.core.debugger import set_trace

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
####################################
# PROJECT VARIABLES
####################################
from HYPER_PARAMS import *
from TORCH_OBJECTS import *

####################################
# INTERNAL LIBRARY
####################################

####################################
# DATA
####################################
def TSP_DATA_LOADER__OFFLINE(filepath, batch_size, num_sample=None):
    """
    filepath: .pkl 文件的路径
    batch_size: 批次大小
    num_sample: (可选) 如果只想加载前 N 条数据用于测试，可以填这个参数
    """
    dataset = TSP_Dataset__Offline(filepath=filepath, num_sample=num_sample)
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0, 
                             collate_fn=TSP_collate_fn)
    return data_loader

class TSP_Dataset__Offline(Dataset):
    def __init__(self, filepath, num_sample=None):
        self.filepath = filepath
        with open(filepath, 'rb') as f:
            self.dataset_list = pickle.load(f)

        if num_sample is not None:
            self.dataset_list = self.dataset_list[:num_sample]


    def __getitem__(self, index):
        item = self.dataset_list[index]
        
        node_xy_data = item['data']  # shape: (100, 2), numpy array
        reference_route = item['solution'] # list: [0, 5, 2, ..., 0]
        
        return node_xy_data, reference_route

    def __len__(self):
        return len(self.dataset_list)

def TSP_collate_fn(batch):
    data_list = [item[0] for item in batch]
    ref_list = [item[1] for item in batch]

    data_tensor = torch.from_numpy(np.stack(data_list)).float()
    lengths = [len(ref) for ref in ref_list]
    max_len = max(lengths)
    min_len_val = min(lengths)
    
    padded_ref_list = []
    for ref in ref_list:
        pad_size = max_len - len(ref)
        padded_ref = list(ref) + [0] * pad_size
        padded_ref_list.append(padded_ref)
    
    ref_tensor = torch.tensor(padded_ref_list).long()
    min_len_tensor = torch.tensor(min_len_val).long()
    
    return data_tensor, ref_tensor, min_len_tensor


####################################
# STATE
####################################
class GROUP_STATE:
    def __init__(self, group_size, data, random_back_mask=None):
        
        self.batch_s = data.size(0)
        self.group_s = group_size 
        self.node_num = data.size(1) 
        self.data = data
        self.device = data.device
        # shape = (batch, node_num)
        self.demands = data[:, :, 2]
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_s, self.group_s, 0), dtype=torch.long, device=self.device)

        # shape = (batch, group)
        self.remaining_capacity = torch.ones((self.batch_s, self.group_s), device=self.device)

        # Visited Mask
        # shape = (batch, group, node_num)
        self.visited_mask = torch.zeros((self.batch_s, self.group_s, self.node_num), dtype=torch.bool, device=self.device)

        # shape = (batch, group, node_num)
        self.ninf_mask = torch.zeros((self.batch_s, self.group_s, self.node_num), device=self.device)
        self.finished = torch.zeros((self.batch_s, self.group_s), dtype=torch.bool, device=self.device)
        self.back_mask_bool = torch.zeros((self.batch_s, self.group_s, self.node_num), dtype=torch.bool, device=self.device)
        
        if random_back_mask is not None:
            back_mask = random_back_mask.to(self.device)
            expanded_indices = back_mask.unsqueeze(1).expand(self.batch_s, self.group_s, -1)
            
            self.back_mask_bool.scatter_(dim=2, index=expanded_indices, value=True)
            
            self.ninf_mask[self.back_mask_bool] = -float('inf')
                
    def move_to(self, selected_idx_mat):
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)

        current_node_demand = self.demands.gather(dim=1, index=selected_idx_mat)
        is_depot = (selected_idx_mat == 0)

        self.remaining_capacity = torch.where(
            is_depot,
            torch.ones_like(self.remaining_capacity),
            self.remaining_capacity - current_node_demand
        )

        scatter_index = selected_idx_mat[:, :, None]
        self.visited_mask.scatter_(dim=2, index=scatter_index, value=True)
        self.visited_mask[:, :, 0] = False

        new_mask = self.visited_mask.clone()

        capacity_mask = self.demands[:, None, :] > self.remaining_capacity[:, :, None]
        new_mask = new_mask | capacity_mask
        
        new_mask = new_mask | self.back_mask_bool
        new_mask[:, :, 0] = is_depot

        self.ninf_mask[:] = 0
        self.ninf_mask[new_mask] = -float('inf')

        current_step_finished = (self.ninf_mask == -float('inf')).all(dim=2)
        self.finished = self.finished | current_step_finished
        if self.finished.any():
            self.ninf_mask[:, :, 0][self.finished] = 0.0


####################################
# ENVIRONMENT
####################################

class GROUP_ENVIRONMENT:

    def __init__(self, data):
        # seq.shape = (batch, TSP_SIZE, 2)

        self.data = data
        self.batch_s = data.size(0)
        self.group_s = None
        self.group_state = None

    def reset(self, group_size, random_back_mask=None):
        self.group_s = group_size
        self.group_state = GROUP_STATE(group_size=group_size, data=self.data, random_back_mask=random_back_mask)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = self.group_state.finished.all()
        if done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_group_travel_distance(self):
            node_index = self.group_state.selected_node_list
            seq_len = node_index.size(2)

            # shape = (batch, group, seq_len, 2)
            gathering_index = node_index.unsqueeze(3).expand(self.batch_s, -1, seq_len, 2)

            seq_expanded = self.data[:, None, :, :2].expand(self.batch_s, self.group_s, -1, 2)

            ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)

            rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
            
            segment_lengths = ((ordered_seq - rolled_seq)**2).sum(3).sqrt()
            # size = (batch, group, seq_len)

            group_travel_distances = segment_lengths.sum(2)
            # size = (batch, group)
            
            return group_travel_distances








