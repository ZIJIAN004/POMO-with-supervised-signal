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

import random
import numpy as np
import time
import math
import torch
import torch.nn as nn
# For debugging
from IPython.core.debugger import set_trace

# Hyper Parameters
from HYPER_PARAMS import *
from TORCH_OBJECTS import *

from source.utilities import Average_Meter
from source.travelling_saleman_problem import TSP_DATA_LOADER__OFFLINE, GROUP_ENVIRONMENT


def generate_consecutive_list():
    length = random.randint(1, 99) 

    max_start_val = 101 - length
    
    start_val = random.randint(2, max_start_val)
    
    result_list = list(range(start_val, start_val + length))
    
    return result_list

########################################
# TRAIN
########################################

def TRAIN(actor_group, epoch, timer_start, logger):

    actor_group.train()
    
    sft_loss_AM = Average_Meter()
    
    train_loader = TSP_DATA_LOADER__OFFLINE(filepath=DATA_FILEPATH, batch_size=TRAIN_BATCH_SIZE)
    episode = 0
    logger_start = time.time()

    for data, ref in train_loader:
        data = data.to(device)
        ref = ref.to(device)
        batch_s = data.size(0)
        episode = episode + batch_s

        random_mask = generate_consecutive_list()
        
        mask_start_step = random_mask[0]     # e.g., 2
        mask_end_step   = random_mask[-1]    # e.g., 4
        
        start_idx = mask_start_step - 1      # e.g., index 1
        end_idx   = mask_end_step - 1        # e.g., index 3

        if mask_end_step != 100:
            back_true_mask = list(range(mask_end_step + 1, 101))
            mask_indices = torch.tensor(back_true_mask, device=ref.device, dtype=torch.long)
            target_indices = mask_indices - 1
            sub_ref = ref[:, target_indices]
        else:
            sub_ref = None

        actor_group.optimizer.zero_grad()

        # Shared Environment Setup
        env = GROUP_ENVIRONMENT(data)
        group_s_p = 1 

        group_state, _, _ = env.reset(group_size=group_s_p, random_back_mask=sub_ref)
        actor_group.reset(group_state)

        if start_idx > 0:
            pre_actions = ref[:, :start_idx]
            
            for t in range(start_idx):

                action = pre_actions[:, t].unsqueeze(1) # (Batch, 1)
                group_state, _, _ = env.step(action)
                
                if t == 0:
                    actor_group.update(group_state) # First Move Context

        
        total_loss = 0
        steps_trained = 0
        
        for t in range(start_idx, end_idx + 1):
            actor_group.update(group_state)

            action_probs = actor_group.get_action_probabilities() # (Batch, 1, N)
            
            true_node = ref[:, t] 
            true_action = true_node.unsqueeze(1) # (Batch, 1)

            log_probs = (action_probs + 1e-10).log()
            
            gathered_log_prob = log_probs.gather(2, true_action.unsqueeze(2)).squeeze() # (Batch)
            
            # Loss = Negative Log Likelihood
            step_loss = -gathered_log_prob.mean()
            
            total_loss += step_loss
            steps_trained += 1
            
            group_state, _, _ = env.step(true_action)
        
        if steps_trained > 0:
            avg_loss = total_loss / steps_trained
            avg_loss.backward(retain_graph=True)
            sft_loss_AM.push(avg_loss.detach().reshape(-1))

        env = GROUP_ENVIRONMENT(data)
        group_s = TSP_SIZE
        group_state, reward, done = env.reset(group_size=group_s)
        actor_group.reset(group_state)

        # First Move is given
        first_action = torch.arange(group_s,device=device)[None, :].expand(batch_s, group_s)
        group_state, reward, done = env.step(first_action)

        group_prob_list = torch.zeros((batch_s, group_s, 0), device=device)
        while not done:
            actor_group.update(group_state)
            action_probs = actor_group.get_action_probabilities()
            # shape = (batch, group, TSP_SIZE)
            action = action_probs.reshape(batch_s*group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s)
            # shape = (batch, group)
            group_state, reward, done = env.step(action)

            batch_idx_mat = torch.arange(batch_s,device=device)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s,device=device)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
            # shape = (batch, group)
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

        # LEARNING - Actor
        ###############################################
        group_reward = reward
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)

        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()
        loss.backward()
        actor_group.optimizer.step()
        # LOGGING
        ###############################################
        if (time.time()-logger_start > LOG_PERIOD_SEC) or (episode == TRAIN_DATASET_SIZE):
            timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
            log_str = 'Ep:{:03d}-{:07d}({:5.1f}%)  T:{:s}  SFT_Loss:{:5f}' \
                .format(epoch, episode, episode/TRAIN_DATASET_SIZE*100,
                        timestr,
                        sft_loss_AM.result())
            logger.info(log_str)
            logger_start = time.time()
            sft_loss_AM = Average_Meter()
            
    # LR STEP
    actor_group.lr_stepper.step()









