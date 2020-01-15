from random import random, randint
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


def random_life(df_order,dnn,num_samples,stride_list,init_multiplier,hw_spec,n=200,return_best_dict=False):
    #after smapling a loop-order, routine to optimize tiling factors to get the energy feedback


    #df_order: input loop-order
    #dnn: user input DNN specs
    #num_samples: max_number of population during tiling factor optimization
    #stride_list: stride numbers for DNN layers
    #init_multiplier: initial population multiplier in tiling factor optimization
    #hw_spec: input hw_spec
    #n: number of iteration to go through in tiling factor optimization 
    #return_best_dict: wether to return the detail results, otherwise only score(penalty) returned

    df_order=copy.deepcopy(df_order)
    dnn=copy.deepcopy(dnn)
    layer_wise=(type(df_order[0])==list)
    if layer_wise:
        #generate reference df_order
        ref_df_order=[]
        for j in range(len(dnn)):
            ref_df_order.append([])
            for i in df_order[j]:
                if 'ref' not in i:
                    ref_df_order[j].append(i)
        #generate net_arch
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #right now because df_order for each layer has the same amount of components
        #they are different in the order of the components
        #we use value 1 to deem one component is not in looporder
        net_arch=gen_net_arch(ref_df_order[0],dnn)
        
    else:
        #generate reference df_order
        ref_df_order=[]
        for i in df_order:
            if 'ref' not in i:
                ref_df_order.append(i)
        #generate net_arch
        net_arch=gen_net_arch(ref_df_order,dnn)    
    #initial max_num pop
    ev_dict1=ev_dict(stride_list,net_arch,ref_df_order,max_pop=num_samples,true_df_order=df_order,hw_spec=hw_spec)
#    print(len(ev_dict1.arch_factor_list[0]['batch']))
#    print(len(ev_dict1.arch_factor_list[0]['ch_in']))
#    print(len(ev_dict1.arch_factor_list[0]['ch_out']))
#    print(len(ev_dict1.arch_factor_list[0]['col_out']))
#    print(len(ev_dict1.arch_factor_list[0]['row_out']))
#    print(len(ev_dict1.arch_factor_list[0]['col_kernel']))
#    print(len(ev_dict1.arch_factor_list[0]['row_kernel']))
    #print(ev_dict1.space_size)
    #ev_dict1.exhuastive_search()
    
    #exit()
    #optimize for n cycles
    ev_dict1.search(n=n,init_multiplier=init_multiplier)       #TODO: add search for n cycles or search for convergence?
    #return the score
    score=ev_dict1.best_score
    if return_best_dict:
        return score,ev_dict1.best_dict,ev_dict1.best_layer_breakdown
    else:
        return score



sample_num=320

tmp_hw_spec={ \
    'gb_vol':108*1024*8, \
    'rf_vol':512*8, \
    'num_pe':168, \
    'num_rf':168
}

dnn=[\
[4, {'ch_out':[96,0],'ch_in':[3,0],'batch':[4,0],'col_out':[55,0],'row_out':[55,0],'row_kernel':[11,0],'col_kernel':[11,0]}],\

#[1,{'ch_out':[256,0],'ch_in':[48,0],'batch':[4,0],'col_out':[27,0],'row_out':[27,0],'row_kernel':[5,0],'col_kernel':[5,0]}],\
]

stride_list=[4]

child= [['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf', 'ref_rf_we',\
                'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc',\
                'ref_gb_we', 'ch_out_gb', 'ref_gb_in',  'ch_in_gb', 'ref_gb_out', \
                'col_out_dram', 'ch_out_dram', 'batch_dram'\
       ]]


for _ in range(1):
    start=time.time()
    print(random_life(child,dnn,sample_num,stride_list,1,tmp_hw_spec,n=int(210)),',',end='')
    #print('TIME: ', time.time()-start)



    score=-99999
    start=time.time()
    for i in range(30): 
        tmp=random_life(child,dnn,sample_num,stride_list,1,tmp_hw_spec,n=int(1))
        if tmp>score:
            score=tmp
    print(score)    
    #print('TIME: ', time.time()-start)



 
