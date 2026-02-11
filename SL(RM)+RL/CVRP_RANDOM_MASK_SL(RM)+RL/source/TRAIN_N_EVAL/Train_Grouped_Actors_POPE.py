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
import torch

from source.utilities import Average_Meter
from source.travelling_saleman_problem import TSP_DATA_LOADER__OFFLINE, GROUP_ENVIRONMENT
from HYPER_PARAMS import *
from TORCH_OBJECTS import *

def generate_consecutive_list(min_len_val):
    if min_len_val < 2:
        return []

    max_end_idx = min_len_val - 1
    max_mask_len = max_end_idx 
    
    length = random.randint(1, max_mask_len)
    
    max_start_val = max_end_idx - length + 1
    start_val = random.randint(1, max_start_val)
    
    result_list = list(range(start_val, start_val + length))
    
    return result_list

def TRAIN(actor_group, epoch, timer_start, logger):

    actor_group.train()
    
    sft_loss_AM = Average_Meter()
    rl_loss_AM = Average_Meter()
    
    train_loader = TSP_DATA_LOADER__OFFLINE(filepath=DATA_FILEPATH, batch_size=TRAIN_BATCH_SIZE)
    episode = 0
    logger_start = time.time()

    for data, ref, min_len_tensor in train_loader:
        data = data.to(device)
        ref = ref.to(device)
        batch_s = data.size(0)
        episode = episode + batch_s
        
        actor_group.optimizer.zero_grad()
        
        min_len_val = min_len_tensor.item()
        random_mask = generate_consecutive_list(min_len_val)
        
        if random_mask:
            mask_start_step = random_mask[0]
            mask_end_step   = random_mask[-1]
            start_idx = mask_start_step
            end_idx   = mask_end_step

            total_max_len = ref.size(1) 
            if end_idx < total_max_len - 1:
                future_steps = torch.arange(end_idx + 1, total_max_len, device=device)
                sub_ref = ref[:, future_steps]
            else:
                sub_ref = None

            env = GROUP_ENVIRONMENT(data)
            group_s_p = 1 
            
            group_state, _, _ = env.reset(group_size=group_s_p, random_back_mask=sub_ref)
            actor_group.reset(group_state)

            if start_idx > 0:
                pre_actions = ref[:, :start_idx] 
                for t in range(start_idx):
                    action = pre_actions[:, t].unsqueeze(1)
                    group_state, _, _ = env.step(action)
                    if t == 0:
                        actor_group.update(group_state)

            sft_total_loss = 0
            steps_trained = 0
            
            for t in random_mask:
                # Update Context
                actor_group.update(group_state)
                # Get Probabilities
                action_probs = actor_group.get_action_probabilities() # (Batch, 1, N)
                
                # Get Ground Truth
                true_node = ref[:, t] 
                true_action = true_node.unsqueeze(1) 

                # Calculate Cross Entropy Loss
                log_probs = (action_probs + 1e-10).log()
                gathered_log_prob = log_probs.gather(2, true_action.unsqueeze(2)).squeeze()
                step_loss = -gathered_log_prob.mean()
                
                sft_total_loss += step_loss
                steps_trained += 1
                
                # Teacher Forcing Step
                group_state, _, _ = env.step(true_action)

            # 1.5 SFT Backward (Gradient Accumulation)
            if steps_trained > 0:
                sft_avg_loss = sft_total_loss / steps_trained
                
                sft_avg_loss.backward()
                
                sft_loss_AM.push(sft_avg_loss.detach(), n_for_rank_0_tensor=1)

        env = GROUP_ENVIRONMENT(data)
        pomo_group_s = data.size(1) - 1
        
        group_state, _, done = env.reset(group_size=pomo_group_s)
        actor_group.reset(group_state)

        start_node_ref = ref[:, 0] # (Batch,) -> Depot (Node 0)
            
        start_action = start_node_ref.unsqueeze(1).expand(batch_s, pomo_group_s) 
            
        group_state, reward, done = env.step(start_action)
        
        actor_group.update(group_state)

        # arange(20) -> 0..19; +1 -> 1..20
        pivot_indices = torch.arange(pomo_group_s, device=device) + 1
        first_action = pivot_indices[None, :].expand(batch_s, pomo_group_s)
        
        group_state, _, done = env.step(first_action)
        actor_group.update(group_state)

        group_prob_list = torch.zeros((batch_s, pomo_group_s, 0), device=device)
        
        while not done:
            action_probs = actor_group.get_action_probabilities()
            # shape = (batch, group, TSP_SIZE)
            
            # Sampling
            action = action_probs.reshape(batch_s * pomo_group_s, -1).multinomial(1)\
                .squeeze(dim=1).reshape(batch_s, pomo_group_s)
            
            # Step
            group_state, reward, done = env.step(action)
            actor_group.update(group_state)

            # Record Log Prob
            batch_idx_mat = torch.arange(batch_s, device=device)[:, None].expand(batch_s, pomo_group_s)
            group_idx_mat = torch.arange(pomo_group_s, device=device)[None, :].expand(batch_s, pomo_group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, pomo_group_s)
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

        rl_raw_distance = env._get_group_travel_distance()
        group_reward = -rl_raw_distance # (Batch, Group)

        # Log Prob Sum
        group_log_prob = group_prob_list.log().sum(dim=2)

        # Advantage (POMO Baseline)
        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        # Loss
        group_loss = -group_advantage * group_log_prob
        rl_loss = group_loss.mean()
        
        rl_loss.backward()
        
        rl_loss_AM.push(rl_loss.detach(), n_for_rank_0_tensor=1)

        actor_group.optimizer.step()


        # LOGGING
        if (time.time()-logger_start > LOG_PERIOD_SEC) or (episode >= TRAIN_DATASET_SIZE):
            timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
            log_str = 'Ep:{:03d}-{:07d}({:5.1f}%)  T:{:s}  SFT_L:{:5f}  RL_L:{:5f}' \
                .format(epoch, episode, episode/TRAIN_DATASET_SIZE*100,
                        timestr,
                        sft_loss_AM.result(),
                        rl_loss_AM.result())
            logger.info(log_str)
            logger_start = time.time()
            sft_loss_AM = Average_Meter()
            rl_loss_AM = Average_Meter()

    actor_group.lr_stepper.step()







