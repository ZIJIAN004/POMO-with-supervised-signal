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

SAVE_FOLDER_NAME = "TRAIN_00_LNN_POPE"
print(SAVE_FOLDER_NAME)

####################################
# EXTERNAL LIBRARY
####################################
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 

import os
import shutil
import time
import numpy as np


####################################
# INTERNAL LIBRARY
####################################
from source.utilities import Get_Logger#这个函数是干嘛的



####################################
# PROJECT VARIABLES
####################################
from HYPER_PARAMS import *
from TORCH_OBJECTS import *



####################################
# PROJECT MODULES (to swap as needed)
####################################
if USE_LNN:
    import source.MODEL__Actor.grouped_actors_lnn as A_Module
else:
    import source.MODEL__Actor.grouped_actors as A_Module

if USE_POPE:
    import source.TRAIN_N_EVAL.Train_Grouped_Actors_POPE as T_Module
else:
    import source.TRAIN_N_EVAL.Train_Grouped_Actors as T_Module
    
import source.TRAIN_N_EVAL.Evaluate_Grouped_Actors as E_Module


# Make Log File
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)


# Save used HYPER_PARAMS
hyper_param_filepath = './HYPER_PARAMS.py'
hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(result_folder_path) 
shutil.copy(hyper_param_filepath, hyper_param_save_path)


############################################################################################################
############################################################################################################



# Objects to Use
actor = A_Module.ACTOR().to(device)
actor.optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE, weight_decay=ACTOR_WEIGHT_DECAY)
actor.lr_stepper = lr_scheduler.StepLR(actor.optimizer, step_size=LR_DECAY_EPOCH, gamma=LR_DECAY_GAMMA)


    
    
# GO
timer_start = time.time()
for epoch in range(1, TOTAL_EPOCH+1):
    
    log_package = {
        'epoch': epoch,
        'timer_start': timer_start,
        'logger': logger     
    }


    #  TRAIN
    #######################################################
    T_Module.TRAIN(actor, **log_package)
    

    #  EVAL
    #######################################################
    E_Module.EVAL(actor, result_folder_path=result_folder_path, **log_package)


    #  CHECKPOINT
    #######################################################
    checkpoint_epochs = np.arange(1, TOTAL_EPOCH+1, 50)
    if epoch in checkpoint_epochs:
        checkpoint_folder_path = '{}/CheckPoint_ep{:05d}'.format(result_folder_path, epoch)
        os.mkdir(checkpoint_folder_path)

        model_save_path = '{}/ACTOR_state_dic.pt'.format(checkpoint_folder_path)
        torch.save(actor.state_dict(), model_save_path)
        optimizer_save_path = '{}/OPTIM_state_dic.pt'.format(checkpoint_folder_path)
        torch.save(actor.optimizer.state_dict(), optimizer_save_path)
        lr_stepper_save_path = '{}/LRSTEP_state_dic.pt'.format(checkpoint_folder_path)
        torch.save(actor.lr_stepper.state_dict(), lr_stepper_save_path)
            

from source.utilities import Extract_from_LogFile

exec_command_str = Extract_from_LogFile(result_folder_path, 'eval_result')
print(exec_command_str)
exec(exec_command_str)

from matplotlib import pyplot as plt
plt.plot(0,0)
plt.show()

plt.plot(eval_result)
plt.grid(True)


plt.savefig('{}/eval_result.jpg'.format(result_folder_path))
