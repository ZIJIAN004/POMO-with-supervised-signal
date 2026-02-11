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
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from ncps.wirings import AutoNCP

import numpy as np

# For debugging
from IPython.core.debugger import set_trace

# Hyper Parameters
from HYPER_PARAMS import *
from TORCH_OBJECTS import *

########################################
# ACTOR
########################################

class ACTOR(nn.Module):

    def __init__(self):
        super().__init__()
        self.box_select_probabilities = None
        # shape = (batch, group, TSP_SIZE)

        self.encoder = Encoder()
        self.node_prob_calculator = Next_Node_Probability_Calculator_for_group()

        self.batch_s = None
        self.encoded_nodes = None

    def reset(self, group_state):
        self.batch_s = group_state.data.size(0)
        self.encoded_nodes = self.encoder(group_state.data)
        # shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        self.node_prob_calculator.reset(self.encoded_nodes, group_ninf_mask=group_state.ninf_mask)

    def soft_reset(self, group_state):
        self.node_prob_calculator.reset(self.encoded_nodes, group_ninf_mask=group_state.ninf_mask)

    def update(self, group_state):
        encoded_LAST_NODES = pick_nodes_for_each_group(self.encoded_nodes, group_state.current_node)
        # shape = (batch_s, group, EMBEDDING_DIM)

        probs = self.node_prob_calculator(encoded_LAST_NODES)
        # shape = (batch_s, group, TSP_SIZE)
        self.box_select_probabilities = probs

    def get_action_probabilities(self):
        return self.box_select_probabilities


########################################
# ACTOR_SUB_NN : ENCODER
########################################

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(2, EMBEDDING_DIM)
        self.layers = nn.ModuleList([Encoder_Layer() for _ in range(ENCODER_LAYER_NUM)])

    def forward(self, data):
        # data.shape = (batch_s, TSP_SIZE, 2)

        embedded_input = self.embedding(data)
        # shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class Encoder_Layer(nn.Module):
    def __init__(self):
        super().__init__()

        self.Wq = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wk = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wv = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.multi_head_combine = nn.Linear(HEAD_NUM * KEY_DIM, EMBEDDING_DIM)

        self.addAndNormalization1 = Add_And_Normalization_Module()
        self.feedForward = Feed_Forward_Module()
        self.addAndNormalization2 = Add_And_Normalization_Module()

    def forward(self, input1):
        # input.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        q = reshape_by_heads(self.Wq(input1), head_num=HEAD_NUM)
        k = reshape_by_heads(self.Wk(input1), head_num=HEAD_NUM)
        v = reshape_by_heads(self.Wv(input1), head_num=HEAD_NUM)
        # q shape = (batch_s, HEAD_NUM, TSP_SIZE, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape = (batch_s, TSP_SIZE, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3


########################################
# ACTOR_SUB_NN : Next_Node_Probability_Calculator
########################################

class Next_Node_Probability_Calculator_for_group(nn.Module):
    def __init__(self):
        super().__init__()

        self.brain_size = LNN_GRAPH_DIM + 64        
        self.readout_size = LNN_GRAPH_DIM 
        
        wiring = AutoNCP(units=self.brain_size, 
                         output_size=self.readout_size, 
                         sparsity_level=0.5)
        
        self.lnn_sparse_encoder = CfC(input_size=EMBEDDING_DIM, 
                                      units=wiring, 
                                      batch_first=True,
                                      return_sequences=False) 

        self.Wq_fused = nn.Linear(LNN_GRAPH_DIM + EMBEDDING_DIM + EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)

        self.Wk = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wv = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.multi_head_combine = nn.Linear(HEAD_NUM * KEY_DIM, EMBEDDING_DIM)

        self.global_graph_feature = None 
        self.encoded_first_node = None   
        
        self.k = None  
        self.v = None  
        self.single_head_key = None  
        self.group_ninf_mask = None  

    def reset(self, encoded_nodes, group_ninf_mask):
        # encoded_nodes.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)
        self.global_graph_feature, _ = self.lnn_sparse_encoder(encoded_nodes)
        
        self.encoded_first_node = None 
        
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=HEAD_NUM)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=HEAD_NUM)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        self.group_ninf_mask = group_ninf_mask

    def forward(self, encoded_LAST_NODE):
        # encoded_LAST_NODE.shape = (batch_s, group, EMBEDDING_DIM)
        batch_s, group_s, _ = encoded_LAST_NODE.size()

        if self.encoded_first_node is None:
            self.encoded_first_node = encoded_LAST_NODE

        # Global Feature: (batch, embed) -> (batch, group, embed)
        graph_feat = self.global_graph_feature.unsqueeze(1).expand(batch_s, group_s, -1)
        
        # First Node: (batch, group, embed)
        first_feat = self.encoded_first_node

        # Last Node: (batch, group, embed)
        last_feat = encoded_LAST_NODE

        cat_input = torch.cat([graph_feat, first_feat, last_feat], dim=-1)

        q_flat = self.Wq_fused(cat_input)

        q = reshape_by_heads(q_flat, head_num=HEAD_NUM)

        out_concat = multi_head_attention(q, self.k, self.v, group_ninf_mask=self.group_ninf_mask)
        
        mh_atten_out = self.multi_head_combine(out_concat)
        
        score = torch.matmul(mh_atten_out, self.single_head_key)
        score_scaled = score / np.sqrt(EMBEDDING_DIM)
        score_clipped = LOGIT_CLIPPING * torch.tanh(score_scaled)
        score_masked = score_clipped + self.group_ninf_mask.clone()
        probs = F.softmax(score_masked, dim=2)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def pick_nodes_for_each_group(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)
    # node_index_to_pick.shape = (batch_s, group_s)
    batch_s = node_index_to_pick.size(0)
    group_s = node_index_to_pick.size(1)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_s, group_s, EMBEDDING_DIM)
    # shape = (batch_s, group, EMBEDDING_DIM)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape = (batch_s, group, EMBEDDING_DIM)

    return picked_nodes


def reshape_by_heads(qkv, head_num):
    # q.shape = (batch, C, head_num*key_dim)

    batch_s = qkv.size(0)
    C = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, C, head_num, -1)
    # shape = (batch, C, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape = (batch, head_num, C, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, ninf_mask=None, group_ninf_mask=None):
    # q shape = (batch_s, head_num, n, key_dim)   : n can be either 1 or TSP_SIZE
    # k,v shape = (batch_s, head_num, TSP_SIZE, key_dim)
    # ninf_mask.shape = (batch_s, TSP_SIZE)
    # group_ninf_mask.shape = (batch_s, group, TSP_SIZE)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape = (batch_s, head_num, n, TSP_SIZE)

    score_scaled = score / np.sqrt(key_dim)
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, None, :].expand(batch_s, head_num, n, TSP_SIZE)
    if group_ninf_mask is not None:
        score_scaled = score_scaled + group_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, TSP_SIZE)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch_s, head_num, n, TSP_SIZE)

    out = torch.matmul(weights, v)
    # shape = (batch_s, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch_s, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch_s, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm_by_EMB = nn.BatchNorm1d(EMBEDDING_DIM, affine=True)

    def forward(self, input1, input2):
        # input.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        batch_s = input1.size(0)
        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * TSP_SIZE, EMBEDDING_DIM))

        return normalized.reshape(batch_s, TSP_SIZE, EMBEDDING_DIM)


class Feed_Forward_Module(nn.Module):
    def __init__(self):
        super().__init__()

        self.W1 = nn.Linear(EMBEDDING_DIM, FF_HIDDEN_DIM)
        self.W2 = nn.Linear(FF_HIDDEN_DIM, EMBEDDING_DIM)

    def forward(self, input1):
        # input.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

        return self.W2(F.relu(self.W1(input1)))
