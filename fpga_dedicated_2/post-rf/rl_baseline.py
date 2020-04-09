
import numpy as np
import test_for_eyeriss as simnas
import time
from itertools import combinations,permutations
import copy
from  multiprocessing import Queue
import multiprocessing
import math
from ev_util import *
from ev_dict_object import *
#for saving np to matlab 
import scipy.io as sio
from rl_controller import *
import torch
import random
import logging
import sys
logger = logging.getLogger(__name__)

def tiling_translation(tiling_scheme,tiling_pool,alloc_slots,pe_array,space_partition):
    tiling_pool=copy.deepcopy(tiling_pool[alloc_slots[pe_array]:alloc_slots[pe_array+1]])
    index=0
    for i in range(len(tiling_scheme)):
        tmp=tiling_scheme[i]
        for j in range(i+1,len(tiling_scheme)):
            tmp*=space_partition[pe_array][j]
        index+=tmp
    #print('abs index: ',index)
    tiling_string=tiling_pool[index]
    return tiling_string[0]



def dsp_check(tiling_scheme_string,dsp_limit):
    dsp_consumption=1    
    for i in tiling_scheme_string:
        if 'noc' in i:
            dsp_consumption*=tiling_scheme_string[i]
    return dsp_consumption<dsp_limit
input_dnn=[\
# [1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[128,0],'ch_in':[64,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[128,0],'ch_in':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[512,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
]


#fpga dedicated 706
tmp_hw_spec={\
    'gb_vol':2*1024*1024, \
    'rf_vol':512, \
    'num_pe':144, \
    'num_rf':144
}


tiling1=fpga_tiling_generator(input_dnn,tmp_hw_spec)


controller_params = {
    #"sw_space": ([3],[3],[16],[1],[1],[2],[4]),
    # dataflow 1, dataflow 2, PE for d1, BW for d1
    "hw_space": [],
    'max_episodes': 40,
    "num_children_per_episode": 1,
    "num_hw_per_child": 10,
    'hidden_units': 35,
}
controller_params_pool=[]

for pe_array in range(4):
    for pe_array_dim_choices in range(10):
        controller_params_pool.append(copy.deepcopy(controller_params))
        tiling_space_1=tiling1.tiling_space_partition(pe_array,0,pe_array_dim_choices)
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
        controller_params_pool[-1]["hw_space"].append(list(range(7)))
    
        controller_params_pool[-1]["hw_space"].append(list(range(tiling_space_1[0])))
        controller_params_pool[-1]["hw_space"].append(list(range(tiling_space_1[1])))
        controller_params_pool[-1]["hw_space"].append(list(range(tiling_space_1[2])))
        controller_params_pool[-1]["hw_space"].append(list(range(tiling_space_1[3])))
        controller_params_pool[-1]["hw_space"].append(list(range(tiling_space_1[4])))
        controller_params_pool[-1]["hw_space"].append(list(range(tiling_space_1[5])))
        controller_params_pool[-1]["hw_space"].append(list(range(tiling_space_1[6])))
        
        controller_params_pool[-1]["hw_space"].append(list(range(2)))
        controller_params_pool[-1]["hw_space"].append(list(range(2)))
        controller_params_pool[-1]["hw_space"].append(list(range(2)))
        controller_params_pool[-1]["hw_space"].append(list(range(2)))
        controller_params_pool[-1]["hw_space"].append(list(range(2)))
        controller_params_pool[-1]["hw_space"].append(list(range(2)))
        controller_params_pool[-1]["hw_space"].append(list(range(2)))
        
sample_num=0        
# print(len(controller_params_pool))
# print(len(controller_params_pool[-1]["hw_space"]))
# exit()
best=np.inf
best_design=None
# for pe_array in range(4):
    # print(sample_num)
    # print(best)
    # for pe_array_dim_choices in range(10):
        # controller_params=controller_params_pool[pe_array*10+pe_array_dim_choices]
        # seed = 0
        # torch.manual_seed(seed)
        # random.seed(seed)
        # logging.basicConfig(stream=sys.stdout,
                            # level=logging.DEBUG,
                            # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

        # print("Begin")
        # controller = Controller(tiling1,controller_params,pe_array,pe_array_dim_choices,tmp_hw_spec)
        # controller.global_train()
        # sample_num+=controller.sample_num
        # if controller.current_best<best:
            # best=controller.current_best
            # best_design=controller.current_best_design
# print(sample_num)
# print(best)





trial=[]
trial_design=[]
trial_best_ts=[]
for _ in range(10):   
    design_history={}
    for i in range(40):
        design_history[i]=[]
    sample_num=0  
    best=np.inf
    best_design=None
    best_ts=None
    for _ in range(25):
        print("="*10)
        print('new run')
        print(sample_num)
        print(best)      
        print(best_design) 
        pe_array=random.randint(0,3)
        pe_array_dim_choices=random.randint(0,9)
        controller_params=controller_params_pool[pe_array*10+pe_array_dim_choices]
        #seed = 0
        #torch.manual_seed(seed)
        #random.seed(seed)
        logging.basicConfig(stream=sys.stdout,
                            level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

        print("Begin")
        controller = Controller(tiling1,controller_params,pe_array,pe_array_dim_choices,tmp_hw_spec,initial_input=design_history[pe_array*10+pe_array_dim_choices])
        controller.global_train()
        sample_num+=controller.sample_num
        design_history[pe_array*10+pe_array_dim_choices]=controller.current_best_design
        if controller.current_best<best:
            best=controller.current_best
            best_design=(pe_array, pe_array_dim_choices,controller.current_best_design)
            best_ts=sample_num        
    print(sample_num)
    print(best)      
    print(best_design)
    trial.append((sample_num,best))
    trial_design.append(best_design)
    trial_best_ts.append(best_ts)
print(trial)
print(trial_best_ts)
print(trial_design)                
