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

import numpy as np
import time

# For debugging
from IPython.core.debugger import set_trace

# Hyper Parameters
from HYPER_PARAMS import *
from TORCH_OBJECTS import *

from source.utilities import Average_Meter
from source.travelling_saleman_problem import TSP_DATA_LOADER__OFFLINE, GROUP_ENVIRONMENT


########################################
# TRAIN
########################################

def TRAIN(actor_group, epoch, timer_start, logger):

    actor_group.train()
    #reset the counting logic
    distance_AM_whole = Average_Meter()
    actor_loss_AM_whole = Average_Meter()
    distance_AM_partial = Average_Meter()
    actor_loss_AM_partial = Average_Meter()
    
    train_loader = TSP_DATA_LOADER__OFFLINE(filepath=DATA_FILEPATH, batch_size=TRAIN_BATCH_SIZE)
    episode=0
    logger_start = time.time()
    for data,ref in train_loader:
        # data.shape = (batch_s, TSP_SIZE, 2)
        # ref.shape= (batch_s, TSP_SIZE+1)
        data=data.to(device)
        ref=ref.to(device)
        
        batch_s = data.size(0)
        episode = episode + batch_s

        # Actor Group Move                              
        ###############################################
        env = GROUP_ENVIRONMENT(data)                  
        group_s = TSP_SIZE
        group_s_p=TSP_SIZE-(ref.size(1)//2)                        

        #Training the model using the partial best solution
        ###############################################
        group_state_p, reward_p, done_p = env.reset(group_size=group_s_p)  
        actor_group.reset(group_state_p)     
        #Push the model to the state of the partial solution
        for i in range(ref.size(1)//2):
            col = ref[:, i] 
            col_unsqueezed = col.unsqueeze(1)
            pre_action = col_unsqueezed.expand(-1, group_s_p)
            group_state_p, reward_p, done_p = env.step(pre_action)
            if i == 0:
                actor_group.update(group_state_p)
        # First Move is given
        first_action = ref[:, (ref.size(1)//2) : -1] 
        group_state_p, reward_p, done_p = env.step(first_action)

        group_prob_list = torch.zeros((batch_s, group_s_p, 0), device=device)
        while not done_p:
            actor_group.update(group_state_p)
            action_probs = actor_group.get_action_probabilities()
            # shape = (batch, group, TSP_SIZE)
            action = action_probs.reshape(batch_s*group_s_p, -1).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s_p)
            # shape = (batch, group)
            group_state_p, reward_p, done_p = env.step(action)

            batch_idx_mat = torch.arange(batch_s,device=device)[:, None].expand(batch_s, group_s_p)
            group_idx_mat = torch.arange(group_s_p,device=device)[None, :].expand(batch_s, group_s_p)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s_p)
            # shape = (batch, group)
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

        # LEARNING - Actor
        ###############################################
        group_reward = reward_p
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)

        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()

        actor_group.optimizer.zero_grad()
        loss.backward()
        actor_group.optimizer.step()
        
        # RECORDING
        ###############################################
        max_reward, _ = group_reward.max(dim=1)
        distance_AM_partial.push(-max_reward)  # reward was given as negative dist
        actor_loss_AM_partial.push(group_loss.detach().reshape(-1))
        
        
        #Train the model using the whole RL
        ###############################################
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

        actor_group.optimizer.zero_grad()
        loss.backward()
        actor_group.optimizer.step()

        # RECORDING
        ###############################################
        max_reward, _ = group_reward.max(dim=1)
        distance_AM_whole.push(-max_reward)  # reward was given as negative dist
        actor_loss_AM_whole.push(group_loss.detach().reshape(-1))


        # LOGGING
        ###############################################
        if (time.time()-logger_start > LOG_PERIOD_SEC) or (episode == TRAIN_DATASET_SIZE):
            timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
            log_str = 'Ep:{:03d}-{:07d}({:5.1f}%)  T:{:s}  ALoss_whole:{:+5f}  CLoss:{:5f}  Avg.dist_whole:{:5f} ALoss_partial:{:+5f} Avg.dist_partial:{:5f}' \
                .format(epoch, episode, episode/TRAIN_DATASET_SIZE*100,
                        timestr, actor_loss_AM_whole.result(), 0,
                        distance_AM_whole.result(),actor_loss_AM_partial.result(),
                        distance_AM_partial.result())
            logger.info(log_str)
            logger_start = time.time()
            distance_AM_whole = Average_Meter()
            actor_loss_AM_whole = Average_Meter()
            distance_AM_partial = Average_Meter()
            actor_loss_AM_partial = Average_Meter()
    # LR STEP, after each epoch
    actor_group.lr_stepper.step()










