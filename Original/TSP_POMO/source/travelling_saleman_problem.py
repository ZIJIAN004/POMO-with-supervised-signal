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


####################################
# EXTERNAL LIBRARY
####################################
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
        
        node_xy_data = item['coords']
        reference_route = item['solution']
        
        return node_xy_data, reference_route

    def __len__(self):
        return len(self.dataset_list)

def TSP_collate_fn(batch):

    data_list = [item[0] for item in batch]
    ref_list = [item[1] for item in batch]
    
    data_tensor = torch.from_numpy(np.stack(data_list)).float()
    ref_tensor = torch.tensor(ref_list).long()
    
    return data_tensor, ref_tensor


####################################
# STATE
####################################
class GROUP_STATE:
    def __init__(self, group_size, data):
        # data.shape = (batch, group, 2)
        self.batch_s = data.size(0)
        self.group_s = group_size
        self.data = data
        self.device = data.device

        # History
        ####################################
        self.selected_count = 0
        self.current_node = None

        # shape = (batch, group, 0)
        self.selected_node_list = torch.zeros((self.batch_s, group_size, 0), dtype=torch.long, device=self.device)

        # Status
        ####################################
        self.ninf_mask = torch.zeros((self.batch_s, group_size, TSP_SIZE), device=self.device)


    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # History
        ####################################
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)

        # Status
        ####################################
        batch_idx_mat = torch.arange(self.batch_s, device=self.device)[:, None].expand(self.batch_s, self.group_s)
        group_idx_mat = torch.arange(self.group_s, device=self.device)[None, :].expand(self.batch_s, self.group_s)
        self.ninf_mask[batch_idx_mat, group_idx_mat, selected_idx_mat] = -float('inf')


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

    def reset(self, group_size):
        self.group_s = group_size
        self.group_state = GROUP_STATE(group_size=group_size, data=self.data)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = (self.group_state.selected_count == TSP_SIZE)
        if done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_group_travel_distance(self):
        gathering_index = self.group_state.selected_node_list.unsqueeze(3).expand(self.batch_s, -1, TSP_SIZE, 2)
        # shape = (batch, group, TSP_SIZE, 2)
        seq_expanded = self.data[:, None, :, :].expand(self.batch_s, self.group_s, TSP_SIZE, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape = (batch, group, TSP_SIZE, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # size = (batch, group, TSP_SIZE)

        group_travel_distances = segment_lengths.sum(2)
        # size = (batch, group)
        return group_travel_distances






