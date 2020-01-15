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

def eval_func(hw_spec):
    eval_val=hw_spec['gb_vol']+hw_spec['num_pe']*hw_spec['rf_vol']
    return eval_val

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
    #optimize for n cycles
    ev_dict1.search(n=n,init_multiplier=init_multiplier)       #TODO: add search for n cycles or search for convergence?
    #return the score
    score=ev_dict1.best_score
    if return_best_dict:
        return score,ev_dict1.best_dict,ev_dict1.best_layer_breakdown
    else:
        return score


def fine_tune(best_lp_set,input_dnn,input_rf,input_stride_list,hw_spec,n=200):
    #a more refined version of random_life, aimed to get the best tiling factors, in turn the best performance, for a loop-order set

    #best_lp_set: loop-orders for each layer
    #input_dnn: user input dnn specs
    #input_rf: rf-noc-template to be used
    #input_stride_list: ...
    #hw_spec: ...
    #n: iteration to optimize tiling factors



    sum_score=0
    dnn=copy.deepcopy(input_dnn)
    stride_list=copy.deepcopy(input_stride_list)
    best_layer_breakdown=[]
    best_dict=[]
    for layer in range(len(dnn)):
        #fine tune loop order based on memory accumulation
        try:
            bscore=random_life(arch_sample_results_df(len(dnn),best_lp_set,input_rf)[layer],[dnn[layer]],320,[stride_list[layer]],3,hw_spec,n=n,return_best_dict=True) #change back  200
        except:
            print(len(dnn))
            print(best_lp_set)
            print(input_rf)
            print(layer)
            print(arch_sample_results_df(len(dnn),best_lp_set,input_rf)[layer])
            print("DATTAFLOW FORMAT ERROR !!!")
            exit()        
        best_dict+=bscore[1]
        best_layer_breakdown+=bscore[2]
        print(bscore[0])
        sum_score+=bscore[0]
    return sum_score,best_dict,best_layer_breakdown


def baseline_gen(best_lp_set,input_dnn,input_stride_list,hw_spec,n=200):
    #legacy function not used; but leave it
    sum_score=0
    dnn=copy.deepcopy(input_dnn)
    stride_list=copy.deepcopy(input_stride_list)
    best_layer_breakdown=[]
    best_dict=[]
    for layer in range(len(dnn)):
        #fine tune loop order based on memory accumulation
        bscore=random_life(best_lp_set,[dnn[layer]],320,[stride_list[layer]],3,hw_spec,n=n,return_best_dict=True) #change back  200
     
        best_dict+=bscore[1]
        best_layer_breakdown+=bscore[2]
        print(bscore[0])
        sum_score+=bscore[0]
    print(best_dict)
    return sum_score,best_dict,best_layer_breakdown


start_time=time.time()
input_stride_list=[4,1,1,1,1]
#inception
input_stride_list=[2,1,1,1,1,1,1,1,1]
input_stride_list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
input_stride_list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#vgg16
input_stride_list=[1,1,1,1,1,1,1,1,1,1,1,1,1]
#input_stride_list=[7,1,1]
prop_m=0.5                                                              #mutation probability
max_pop=200               #change back  200
#number of samples for each df_order  has to be multiple of cores in the machine
#the below is not recommended for changed
sample_num=320             #change back  200
dying_rate=0.2                                                          #the dying_rate is aimed to allowing only elites to have children
k=10                 #change back    10                                                   #top k looporder

#inceptionv1
input_dnn=[\
[2,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[7,0],'col_kernel':[7,0]}],\
[1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[192,0],'ch_in':[64,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[64,0],'ch_in':[192,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[96,0],'ch_in':[192,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[128,0],'ch_in':[96,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[16,0],'ch_in':[192,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[32,0],'ch_in':[16,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[32,0],'ch_in':[192,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[1,0],'col_kernel':[1,0]}]
#3c
]
input_dnn=[\
#4b
[1,{'ch_out':[192,0],'ch_in':[480,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[96,0],'ch_in':[480,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[208,0],'ch_in':[96,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[16,0],'ch_in':[480,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[48,0],'ch_in':[16,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[64,0],'ch_in':[480,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
#4c
[1,{'ch_out':[160,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[112,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[224,0],'ch_in':[112,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[24,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[64,0],'ch_in':[24,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[64,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
#4d
[1,{'ch_out':[128,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[128,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[24,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[64,0],'ch_in':[24,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[64,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
#4e
[1,{'ch_out':[112,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[144,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[288,0],'ch_in':[144,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[32,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[64,0],'ch_in':[32,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[64,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}]
]


input_dnn=[\
#4f
[1,{'ch_out':[256,0],'ch_in':[528,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[160,0],'ch_in':[528,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[320,0],'ch_in':[160,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[32,0],'ch_in':[528,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[128,0],'ch_in':[32,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[128,0],'ch_in':[528,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
#5b
[1,{'ch_out':[256,0],'ch_in':[832,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[160,0],'ch_in':[832,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[320,0],'ch_in':[160,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[32,0],'ch_in':[832,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[128,0],'ch_in':[32,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[128,0],'ch_in':[832,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
#5c
[1,{'ch_out':[384,0],'ch_in':[832,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[192,0],'ch_in':[832,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[384,0],'ch_in':[192,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[48,0],'ch_in':[832,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
[1,{'ch_out':[128,0],'ch_in':[48,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[128,0],'ch_in':[832,0],'batch':[1,0],'col_out':[7,0],'row_out':[7,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
#fc
[1,{'ch_out':[1001,0],'ch_in':[1024,0],'batch':[1,0],'col_out':[1,0],'row_out':[1,0],'row_kernel':[1,0],'col_kernel':[1,0]}]

]


#vgg16
#conv
#input_dnn=[\
#[1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[128,0],'ch_in':[64,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[128,0],'ch_in':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}] \
#]
#fc
#input_dnn=[\
#[7,{'ch_out':[4096,0],'ch_in':[7,0],'batch':[1,0],'col_out':[1,0],'row_out':[1,0],'row_kernel':[7,0],'col_kernel':[7,0]}],\
#[1,{'ch_out':[4096,0],'ch_in':[4096,0],'batch':[1,0],'col_out':[1,0],'row_out':[1,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
#[1,{'ch_out':[4096,0],'ch_in':[1001,0],'batch':[1,0],'col_out':[1,0],'row_out':[1,0],'row_kernel':[1,0],'col_kernel':[1,0]}] \
#]



#lenet 5
# input_dnn=[\
# [1, {'ch_out':[6,0],'ch_in':[1,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[5,0],'col_kernel':[5,0]}],\
# [1, {'ch_out':[16,0],'ch_in':[6,0],'batch':[1,0],'col_out':[10,0],'row_out':[10,0],'row_kernel':[5,0],'col_kernel':[5,0]}],\
# [1, {'ch_out':[120,0],'ch_in':[11,0],'batch':[1,0],'col_out':[1,0],'row_out':[1,0],'row_kernel':[5,0],'col_kernel':[5,0]}],\
# #fc
# [1, {'ch_out':[84,0],'ch_in':[120,0],'batch':[1,0],'col_out':[1,0],'row_out':[1,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
# [1, {'ch_out':[10,0],'ch_in':[84,0],'batch':[1,0],'col_out':[1,0],'row_out':[1,0],'row_kernel':[1,0],'col_kernel':[1,0]}],\
# ]
#alexnet
input_dnn=[\
[4, {'ch_out':[96,0],'ch_in':[3,0],'batch':[4,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[11,0],'col_kernel':[11,0]}],\

[1,{'ch_out':[256,0],'ch_in':[48,0],'batch':[4,0],'col_out':[27,0],'row_out':[27,0],'row_kernel':[5,0],'col_kernel':[5,0]}],\

[1,{'ch_out':[384,0],'ch_in':[256,0],'batch':[4,0],'col_out':[13,0],'row_out':[13,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\

[1,{'ch_out':[384,0],'ch_in':[192,0],'batch':[4,0],'col_out':[13,0],'row_out':[13,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\

[1,{'ch_out':[256,0],'ch_in':[192,0],'batch':[4,0],'col_out':[13,0],'row_out':[13,0],'row_kernel':[3,0],'col_kernel':[3,0]}]\
]




test_child=[
            #{'ch_out_rf': 2, 'ch_in_rf': 1,'row_kernel_rf': 11, 'row_out_rf': 55,'batch_rf': 2,\
            # 'col_kernel_noc': 11,'ch_in_noc': 1,'col_out_noc': 7, 'ch_out_noc': 2, \
            # 'ch_out_gb': 12,'ch_in_gb': 3,\
            # 'col_out_dram': 8,'ch_out_dram': 2, 'batch_dram': 2,    },\
            {'ch_out_rf':16, 'ch_in_rf':1, 'row_kernel_rf':11, 'ref_rf_we':64, 'row_out_rf':56, 'ref_rf_in':16, 'batch_rf':1,\
            'ref_rf_out':64, 'col_kernel_noc':11, 'ch_in_noc':1, 'col_out_noc':7, 'ch_out_noc':2,\
            'ref_gb_we':64, 'ch_out_gb':2, 'ref_gb_in':64, 'ch_in_gb':3,\
            'ref_gb_out':64, 'col_out_dram':8, 'ch_out_dram':1, 'batch_dram':4,\
            },\
            {'ch_out_rf':16, 'ch_in_rf':2, 'row_kernel_rf':5, 'ref_rf_we':64, 'row_out_rf':27, 'ref_rf_in':16, 'batch_rf':1,\
            'ref_rf_out':64, 'col_kernel_noc':5, 'ch_in_noc':1, 'col_out_noc':27, 'ch_out_noc':1,\
            'ref_gb_we':64, 'ch_out_gb':4, 'ref_gb_in':64, 'ch_in_gb':24,\
            'ref_gb_out':64, 'col_out_dram':1, 'ch_out_dram':4, 'batch_dram':4,\
            },\
            {'ch_out_rf':16, 'ch_in_rf':6, 'row_kernel_rf':3, 'ref_rf_out':64, 'row_out_rf':13, 'ref_rf_in':16, 'batch_rf':4,\
            'ref_rf_we':64, 'col_kernel_noc':3, 'ch_in_noc':1, 'col_out_noc':13, 'ch_out_noc':4,\
            'ref_gb_we':64, 'ch_out_gb':1, 'ref_gb_in':64, 'ch_in_gb':43,\
            'ref_gb_out':64, 'col_out_dram':1, 'ch_out_dram':6, 'batch_dram':1,\
            },\
            {'ch_out_rf':16, 'ch_in_rf':3, 'row_kernel_rf':3, 'ref_rf_out':64, 'row_out_rf':13, 'ref_rf_in':16, 'batch_rf':4,\
             'ref_rf_we':64, 'col_kernel_noc':3, 'ch_in_noc':2, 'col_out_noc':13, 'ch_out_noc':2,\
             'ref_gb_we':64, 'ch_out_gb':2, 'ref_gb_in':64, 'ch_in_gb':32,\
             'ref_gb_out':64, 'col_out_dram':1, 'ch_out_dram':6, 'batch_dram':1,\
            },\
            {'ch_out_rf':16, 'ch_in_rf':3, 'row_kernel_rf':3, 'ref_rf_out':64, 'row_out_rf':13, 'ref_rf_in':16, 'batch_rf':4,\
            'ref_rf_we':64, 'col_kernel_noc':3, 'ch_in_noc':2, 'col_out_noc':13, 'ch_out_noc':2,\
            'ref_gb_we':64, 'ch_out_gb':2, 'ref_gb_in':64, 'ch_in_gb':32,\
            'ref_gb_out':64, 'col_out_dram':1, 'ch_out_dram':4, 'batch_dram':1
            }
            ]
test_looporder=['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf', 'ref_rf_we',\
                'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc',\
                'ref_gb_we', 'ch_out_gb', 'ref_gb_in',  'ch_in_gb', 'ref_gb_out', \
                'col_out_dram', 'ch_out_dram', 'batch_dram'\
               ]

##TPU
#test_child=[{'row_out_rf':55, 'col_out_rf':55,'batch_rf':1,\
#             'ch_in_noc':1,'ch_out_noc':16,'col_kernel_noc':11, 'row_kernel_noc':11,
#             'ch_in_gb':3,'ch_out_gb':2,  \
#             'ch_in_dram':1, 'ch_out_dram':3, 'batch_dram':1\
#            },\
#            # {'ch_out_rf':16, 'ch_in_rf':2, 'row_kernel_rf':5, 'ref_rf_we':64, 'row_out_rf':27, 'ref_rf_in':16, 'batch_rf':1,\
#            # 'ref_rf_out':64, 'col_kernel_noc':5, 'ch_in_noc':1, 'col_out_noc':27, 'ch_out_noc':1,\
#            # 'ref_gb_we':64, 'ch_out_gb':4, 'ref_gb_in':64, 'ch_in_gb':24,\
#            # 'ref_gb_out':64, 'col_out_dram':1, 'ch_out_dram':4, 'batch_dram':4,\
#            # },\
#            # {'ch_out_rf':16, 'ch_in_rf':6, 'row_kernel_rf':3, 'ref_rf_out':64, 'row_out_rf':13, 'ref_rf_in':16, 'batch_rf':4,\
#            # 'ref_rf_we':64, 'col_kernel_noc':3, 'ch_in_noc':1, 'col_out_noc':13, 'ch_out_noc':4,\
#            # 'ref_gb_we':64, 'ch_out_gb':1, 'ref_gb_in':64, 'ch_in_gb':43,\
#            # 'ref_gb_out':64, 'col_out_dram':1, 'ch_out_dram':6, 'batch_dram':1,\
#            # },\
#            # {'ch_out_rf':16, 'ch_in_rf':3, 'row_kernel_rf':3, 'ref_rf_out':64, 'row_out_rf':13, 'ref_rf_in':16, 'batch_rf':4,\
#             # 'ref_rf_we':64, 'col_kernel_noc':3, 'ch_in_noc':2, 'col_out_noc':13, 'ch_out_noc':2,\
#             # 'ref_gb_we':64, 'ch_out_gb':2, 'ref_gb_in':64, 'ch_in_gb':32,\
#             # 'ref_gb_out':64, 'col_out_dram':1, 'ch_out_dram':6, 'batch_dram':1,\
#            # },\
#            # {'ch_out_rf':16, 'ch_in_rf':3, 'row_kernel_rf':3, 'ref_rf_out':64, 'row_out_rf':13, 'ref_rf_in':16, 'batch_rf':4,\
#            # 'ref_rf_we':64, 'col_kernel_noc':3, 'ch_in_noc':2, 'col_out_noc':13, 'ch_out_noc':2,\
#            # 'ref_gb_we':64, 'ch_out_gb':2, 'ref_gb_in':64, 'ch_in_gb':32,\
#            # 'ref_gb_out':64, 'col_out_dram':1, 'ch_out_dram':4, 'batch_dram':1
#            # }
#            ]               
#test_looporder=['row_out_rf', 'col_out_rf', 'batch_rf','ref_rf_out','ref_rf_in', 'ref_rf_we', \
#                'ch_in_noc','ch_out_noc','col_kernel_noc', 'row_kernel_noc',\
#                'ref_gb_we',   'ref_gb_we', 'ch_in_gb','ch_out_gb', 'ref_gb_out',  \
#                'ch_in_dram', 'ch_out_dram', 'batch_dram'\
#               ]  




##shiDianNao
#test_child=[{'col_kernel_rf':11, 'row_kernel_rf':11,'ch_in_rf':3,\
#             'batch_noc':1,'col_out_noc':16,'row_out_noc':16,\
#             'ch_out_gb':96,'col_out_gb':4,'row_out_gb':4,  \
#             'ch_out_dram':1, 'batch_dram':1\
#            },\
#            {'col_kernel_rf':5, 'row_kernel_rf':5,'ch_in_rf':48,\
#             'batch_noc':1,'col_out_noc':16,'row_out_noc':16,\
#             'ch_out_gb':86,'col_out_gb':2,'row_out_gb':2,  \
#             'ch_out_dram':3, 'batch_dram':1\
#            },\
#            {'col_kernel_rf':3, 'row_kernel_rf':3,'ch_in_rf':256,\
#             'batch_noc':1,'col_out_noc':13,'row_out_noc':13,\
#             'ch_out_gb':24,'col_out_gb':1,'row_out_gb':1,  \
#             'ch_out_dram':16, 'batch_dram':1\
#            },\
#            {'col_kernel_rf':3, 'row_kernel_rf':3,'ch_in_rf':192,\
#             'batch_noc':1,'col_out_noc':13,'row_out_noc':13,\
#             'ch_out_gb':48,'col_out_gb':1,'row_out_gb':1,  \
#             'ch_out_dram':8, 'batch_dram':1\
#            },\
#            {'col_kernel_rf':3, 'row_kernel_rf':3,'ch_in_rf':192,\
#             'batch_noc':1,'col_out_noc':13,'row_out_noc':13,\
#             'ch_out_gb':32,'col_out_gb':1,'row_out_gb':1,  \
#             'ch_out_dram':8, 'batch_dram':1\
#            }\
#            ]               
#test_looporder=['ref_gb_we','ref_rf_in','ref_rf_out','col_kernel_rf', 'row_kernel_rf','ch_in_rf', \
#                'batch_noc','col_out_noc','row_out_noc',\
#                'ref_gb_in','ref_gb_out','ch_out_gb','ref_gb_we','col_out_gb','row_out_gb', \
#                'ch_out_dram', 'batch_dram'\
#               ]  



input_stride_list=[4,1,1,1,1]
stride_list=input_stride_list
print(arch_life(test_child,stride_list,default_hw,df_order=test_looporder))
#exit()

#exit()
#input_stride_list=[4,1,1,1,1]
##exit()
##shiDianNao baseline generation
#dnn=input_dnn
#stride_list=input_stride_list
##fine tune loop order based on memory accumulation
#sum_score=0
#dnn=copy.deepcopy(input_dnn)
#stride_list=copy.deepcopy(input_stride_list)
#best_layer_breakdown=[]
#best_dict=[]
#for layer in range(len(dnn)):
#    #fine tune loop order based on memory accumulation
#    bscore=random_life(test_looporder,[dnn[layer]],200,[stride_list[layer]],3,n=200,return_best_dict=True)
#    best_dict+=bscore[1]
#    best_layer_breakdown+=bscore[2]
#    print(bscore[0])
#    sum_score+=bscore[0]
#    print(best_dict)
#print(sum_score,best_dict,best_layer_breakdown)
#exit()





#shiDianNao_lporder=[[8, 4, 7, 2, 1, 1, 3, 2, 0, 0, 0, 3, 4, 1, 1, 0, 0, 4, 3, 5, 1, 3, 0, 0, 2, 1, 0, 5, 3, 1, 3, 2, 1, 0], [6, 4, 2, 0, 1, 1, 2, 1, 1, 0, 4, 1, 4, 0, 0, 0, 0, 7, 3, 4, 4, 3, 4, 0, 2, 0, 0, 1, 0, 4, 1, 2, 0, 0], [6, 2, 6, 3, 4, 4, 3, 1, 0, 0, 1, 5, 4, 2, 2, 0, 0, 6, 3, 0, 2, 0, 2, 2, 2, 1, 0, 3, 5, 4, 1, 2, 0, 0], [7, 1, 3, 6, 1, 3, 1, 0, 0, 0, 3, 2, 4, 1, 2, 0, 0, 4, 5, 0, 0, 4, 0, 1, 1, 0, 0, 6, 3, 3, 0, 0, 0, 0], [2, 1, 4, 1, 1, 2, 3, 0, 0, 0, 3, 3, 3, 1, 0, 0, 0, 6, 7, 6, 1, 3, 0, 2, 1, 1, 0, 1, 1, 4, 3, 2, 0, 0]]
#shiDianNao_child= [{'ch_out_rf': 16, 'ch_out_noc': 1, 'ch_out_gb': 6, 'ch_out_dram': 1, 'ch_in_rf': 1, 'ch_in_noc': 1, 'ch_in_gb': 3, 'ch_in_dram': 1, 'batch_rf': 1, 'batch_noc': 1, 'batch_gb': 1, 'batch_dram': 1, 'col_out_rf': 1, 'col_out_noc': 11, 'col_out_gb': 5, 'col_out_dram': 1, 'row_out_rf': 7, 'row_out_noc': 1, 'row_out_gb': 1, 'row_out_dram': 8, 'row_kernel_rf': 11, 'row_kernel_noc': 1, 'row_kernel_gb': 1, 'row_kernel_dram': 1, 'col_kernel_rf': 1, 'col_kernel_noc': 11, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 4, 'ch_out_noc': 4, 'ch_out_gb': 8, 'ch_out_dram': 2, 'ch_in_rf': 3, 'ch_in_noc': 4, 'ch_in_gb': 4, 'ch_in_dram': 1, 'batch_rf': 1, 'batch_noc': 1, 'batch_gb': 1, 'batch_dram': 1, 'col_out_rf': 27, 'col_out_noc': 1, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 3, 'row_out_noc': 3, 'row_out_gb': 3, 'row_out_dram': 1, 'row_kernel_rf': 5, 'row_kernel_noc': 1, 'row_kernel_gb': 1, 'row_kernel_dram': 1, 'col_kernel_rf': 1, 'col_kernel_noc': 5, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 1, 'ch_out_noc': 3, 'ch_out_gb': 32, 'ch_out_dram': 4, 'ch_in_rf': 32, 'ch_in_noc': 2, 'ch_in_gb': 4, 'ch_in_dram': 1, 'batch_rf': 1, 'batch_noc': 1, 'batch_gb': 1, 'batch_dram': 1, 'col_out_rf': 1, 'col_out_noc': 13, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 13, 'row_out_noc': 1, 'row_out_gb': 1, 'row_out_dram': 1, 'row_kernel_rf': 3, 'row_kernel_noc': 1, 'row_kernel_gb': 1, 'row_kernel_dram': 1, 'col_kernel_rf': 1, 'col_kernel_noc': 3, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 16, 'ch_out_noc': 1, 'ch_out_gb': 12, 'ch_out_dram': 2, 'ch_in_rf': 2, 'ch_in_noc': 2, 'ch_in_gb': 48, 'ch_in_dram': 1, 'batch_rf': 1, 'batch_noc': 1, 'batch_gb': 1, 'batch_dram': 1, 'col_out_rf': 1, 'col_out_noc': 13, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 13, 'row_out_noc': 1, 'row_out_gb': 1, 'row_out_dram': 1, 'row_kernel_rf': 3, 'row_kernel_noc': 1, 'row_kernel_gb': 1, 'row_kernel_dram': 1, 'col_kernel_rf': 1, 'col_kernel_noc': 3, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 128, 'ch_out_noc': 2, 'ch_out_gb': 1, 'ch_out_dram': 1, 'ch_in_rf': 3, 'ch_in_noc': 1, 'ch_in_gb': 64, 'ch_in_dram': 1, 'batch_rf': 1, 'batch_noc': 1, 'batch_gb': 1, 'batch_dram': 1, 'col_out_rf': 13, 'col_out_noc': 1, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_out_noc': 13, 'row_out_gb': 1, 'row_out_dram': 1, 'row_kernel_rf': 1, 'row_kernel_noc': 3, 'row_kernel_gb': 1, 'row_kernel_dram': 1, 'col_kernel_rf': 1, 'col_kernel_noc': 3, 'col_kernel_gb': 1, 'col_kernel_dram': 1}]




#print("================================================================================================")
#print("\n \n")

#shiDianNao_lporder=arch_sample_results_df(len(shiDianNao_lporder),shiDianNao_lporder)
#print(arch_life(shiDianNao_child,stride_list,df_order=shiDianNao_lporder))




input_stride_list=[4,1,1,1,1]

#num_pe*rf_vol+gb_vol
possible_hw_values={ \
'num_pe':[64,128,256,512,1024], \
'gb_vol':[1,2,4,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048], \
'rf_vol':[1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,768,1024], \

}
def generate_all_possible_hw(possible_hw_values):
    hw_pool=[]
    for num_pe in possible_hw_values['num_pe']:
        for gb_vol in possible_hw_values['gb_vol']:
            for rf_vol in possible_hw_values['rf_vol']:
                hw_pool.append({'gb_vol':gb_vol*8*1024, 'rf_vol':rf_vol*8,'num_pe':num_pe,'num_rf':num_pe}) 
    return hw_pool
def filter_hw_pool(hw_pool,budget):
    hw_pool=copy.deepcopy(hw_pool)
    filtered_pool=[]
    for hw_spec in hw_pool:
        tmp_val=eval_func(hw_spec)
        if (tmp_val <= budget*1) and (tmp_val >= budget*0.8):
            filtered_pool.append(hw_spec)
    return filtered_pool

#eyeriss
tmp_hw_spec={ \
    'gb_vol':108*1024*8, \
    'rf_vol':512*8, \
    'num_pe':168, \
    'num_rf':168
}

test_child=[{'ch_out_rf': 2, 'ch_out_noc': 3, 'ch_out_gb': 16, 'ch_out_dram': 1, 'ch_in_rf': 1, 'ch_in_noc': 1, 'ch_in_gb': 3, 'batch_rf': 2, 'batch_dram': 2, 'col_out_noc': 4, 'col_out_dram': 14, 'row_out_rf': 56, 'row_kernel_rf': 11, 'col_kernel_noc': 11}, {'ch_out_rf': 1, 'ch_out_noc': 4, 'ch_out_gb': 16, 'ch_out_dram': 4, 'ch_in_rf': 6, 'ch_in_noc': 2, 'ch_in_gb': 4, 'batch_rf': 4, 'batch_dram': 1, 'col_out_noc': 4, 'col_out_dram': 7, 'row_out_rf': 27, 'row_kernel_rf': 5, 'col_kernel_noc': 5}, {'ch_out_rf': 4, 'ch_out_noc': 2, 'ch_out_gb': 16, 'ch_out_dram': 3, 'ch_in_rf': 8, 'ch_in_noc': 2, 'ch_in_gb': 16, 'batch_rf': 2, 'batch_dram': 2, 'col_out_noc': 13, 'col_out_dram': 1, 'row_out_rf': 13, 'row_kernel_rf': 3, 'col_kernel_noc': 3}, {'ch_out_rf': 2, 'ch_out_noc': 4, 'ch_out_gb': 8, 'ch_out_dram': 6, 'ch_in_rf': 12, 'ch_in_noc': 1, 'ch_in_gb': 16, 'batch_rf': 4, 'batch_dram': 1, 'col_out_noc': 13, 'col_out_dram': 1, 'row_out_rf': 13, 'row_kernel_rf': 3, 'col_kernel_noc': 3}, {'ch_out_rf': 2, 'ch_out_noc': 4, 'ch_out_gb': 8, 'ch_out_dram': 4, 'ch_in_rf': 12, 'ch_in_noc': 1, 'ch_in_gb': 16, 'batch_rf': 4, 'batch_dram': 1, 'col_out_noc': 13, 'col_out_dram': 1, 'row_out_rf': 13, 'row_kernel_rf': 3, 'col_kernel_noc': 3}]

#test_child=[{'ch_out_rf': 1, 'ch_out_noc': 1, 'ch_out_gb': 1, 'ch_out_dram': 6, 'ch_in_rf': 1, 'ch_in_noc': 1, 'ch_in_gb': 1, 'batch_rf': 1, 'batch_dram': 1, 'col_out_noc': 1, 'col_out_dram': 28, 'row_out_rf': 28, 'row_kernel_rf': 5, 'col_kernel_noc': 5}, {'ch_out_rf': 1, 'ch_out_noc': 1, 'ch_out_gb': 1, 'ch_out_dram': 16, 'ch_in_rf': 1, 'ch_in_noc': 2, 'ch_in_gb': 6, 'batch_rf': 1, 'batch_dram': 1, 'col_out_noc': 1, 'col_out_dram': 10, 'row_out_rf': 10, 'row_kernel_rf': 5, 'col_kernel_noc': 5}, {'ch_out_rf': 1, 'ch_out_noc': 1, 'ch_out_gb': 1, 'ch_out_dram': 120, 'ch_in_rf': 1, 'ch_in_noc': 1, 'ch_in_gb': 11, 'batch_rf': 1, 'batch_dram': 1, 'col_out_noc': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_kernel_rf': 5, 'col_kernel_noc': 5}, {'ch_out_rf': 1, 'ch_out_noc': 1, 'ch_out_gb': 1, 'ch_out_dram': 84, 'ch_in_rf': 1, 'ch_in_noc': 1, 'ch_in_gb': 120, 'batch_rf': 1, 'batch_dram': 1, 'col_out_noc': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_kernel_rf': 1, 'col_kernel_noc': 1}, {'ch_out_rf': 1, 'ch_out_noc': 1, 'ch_out_gb': 1, 'ch_out_dram': 10, 'ch_in_rf': 1, 'ch_in_noc': 1, 'ch_in_gb': 84, 'batch_rf': 1, 'batch_dram': 1, 'col_out_noc': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_kernel_rf': 1, 'col_kernel_noc': 1}]

#print(arch_life(test_child,stride_list,tmp_hw_spec,df_order=test_looporder))
#exit()

#baseline_gen(test_looporder,input_dnn,input_stride_list,tmp_hw_spec,n=200)
#exit()


#shidiannao
tmp_hw_spec={ \
    'gb_vol':250*1024*8, \
    'rf_vol':92*8, \
    'num_pe':256, \
    'num_rf':256
}


#tpu
tmp_hw_spec={ \
    'gb_vol':338*1024*8, \
    'rf_vol':32*8, \
    'num_pe':256, \
    'num_rf':256
}

#eyeriss
test_child=[{'ch_out_rf': 2, 'ch_out_noc': 3, 'ch_out_gb': 8, 'ch_out_dram': 2, 'ch_in_rf': 1, 'ch_in_noc': 1, 'ch_in_gb': 3, 'batch_rf': 2, 'batch_dram': 2, 'col_out_noc': 5, 'col_out_dram': 11, 'row_out_rf': 55, 'row_kernel_rf': 11, 'col_kernel_noc': 11}, {'ch_out_rf': 8, 'ch_out_noc': 1, 'ch_out_gb': 4, 'ch_out_dram': 8, 'ch_in_rf': 3, 'ch_in_noc': 1, 'ch_in_gb': 16, 'batch_rf': 1, 'batch_dram': 4, 'col_out_noc': 27, 'col_out_dram': 1, 'row_out_rf': 27, 'row_kernel_rf': 5, 'col_kernel_noc': 5}, {'ch_out_rf': 4, 'ch_out_noc': 4, 'ch_out_gb': 8, 'ch_out_dram': 3, 'ch_in_rf': 8, 'ch_in_noc': 1, 'ch_in_gb': 32, 'batch_rf': 2, 'batch_dram': 2, 'col_out_noc': 13, 'col_out_dram': 1, 'row_out_rf': 13, 'row_kernel_rf': 3, 'col_kernel_noc': 3}, {'ch_out_rf': 2, 'ch_out_noc': 4, 'ch_out_gb': 12, 'ch_out_dram': 4, 'ch_in_rf': 12, 'ch_in_noc': 1, 'ch_in_gb': 16, 'batch_rf': 2, 'batch_dram': 2, 'col_out_noc': 13, 'col_out_dram': 1, 'row_out_rf': 13, 'row_kernel_rf': 3, 'col_kernel_noc': 3}, {'ch_out_rf': 4, 'ch_out_noc': 2, 'ch_out_gb': 16, 'ch_out_dram': 2, 'ch_in_rf': 6, 'ch_in_noc': 2, 'ch_in_gb': 16, 'batch_rf': 2, 'batch_dram': 2, 'col_out_noc': 13, 'col_out_dram': 1, 'row_out_rf': 13, 'row_kernel_rf': 3, 'col_kernel_noc': 3}]
test_child=[{'ch_out_rf': 2, 'ch_out_noc': 2, 'ch_out_gb': 24, 'ch_out_dram': 1, 'ch_in_rf': 1, 'ch_in_noc': 1, 'ch_in_gb': 3, 'batch_rf': 2, 'batch_dram': 2, 'col_out_noc': 4, 'col_out_dram': 14, 'row_out_rf': 56, 'row_kernel_rf': 11, 'col_kernel_noc': 11}, {'ch_out_rf': 4, 'ch_out_noc': 1, 'ch_out_gb': 16, 'ch_out_dram': 4, 'ch_in_rf': 3, 'ch_in_noc': 1, 'ch_in_gb': 16, 'batch_rf': 1, 'batch_dram': 4, 'col_out_noc': 27, 'col_out_dram': 1, 'row_out_rf': 27, 'row_kernel_rf': 5, 'col_kernel_noc': 5}, {'ch_out_rf': 3, 'ch_out_noc': 4, 'ch_out_gb': 8, 'ch_out_dram': 4, 'ch_in_rf': 8, 'ch_in_noc': 1, 'ch_in_gb': 32, 'batch_rf': 2, 'batch_dram': 2, 'col_out_noc': 13, 'col_out_dram': 1, 'row_out_rf': 13, 'row_kernel_rf': 3, 'col_kernel_noc': 3}, {'ch_out_rf': 2, 'ch_out_noc': 4, 'ch_out_gb': 8, 'ch_out_dram': 6, 'ch_in_rf': 8, 'ch_in_noc': 1, 'ch_in_gb': 24, 'batch_rf': 4, 'batch_dram': 1, 'col_out_noc': 13, 'col_out_dram': 1, 'row_out_rf': 13, 'row_kernel_rf': 3, 'col_kernel_noc': 3}, {'ch_out_rf': 4, 'ch_out_noc': 2, 'ch_out_gb': 16, 'ch_out_dram': 2, 'ch_in_rf': 8, 'ch_in_noc': 2, 'ch_in_gb': 12, 'batch_rf': 2, 'batch_dram': 2, 'col_out_noc': 13, 'col_out_dram': 1, 'row_out_rf': 13, 'row_kernel_rf': 3, 'col_kernel_noc': 3}]


# ShiDiannao os
test_looporder=[
            'ref_rf_we','ref_rf_in','ref_rf_out','col_kernel_rf','row_kernel_rf','ch_in_rf',\
            'col_out_noc','row_out_noc','ch_out_noc',\
            'ref_gb_out','ref_gb_in','ch_out_gb','ref_gb_we','col_out_gb','row_out_gb',\
            'ch_out_dram','batch_dram',\
]

test_child=[{'ch_out_noc': 8, 'ch_out_gb': 12, 'ch_out_dram': 1, 'ch_in_rf': 3, 'batch_dram': 4, 'col_out_noc': 28, 'col_out_gb': 2, 'row_out_noc': 1, 'row_out_gb': 56, 'row_kernel_rf': 11, 'col_kernel_rf': 11}, {'ch_out_noc': 4, 'ch_out_gb': 16, 'ch_out_dram': 4, 'ch_in_rf': 48, 'batch_dram': 4, 'col_out_noc': 9, 'col_out_gb': 3, 'row_out_noc': 7, 'row_out_gb': 4, 'row_kernel_rf': 5, 'col_kernel_rf': 5}, {'ch_out_noc': 16, 'ch_out_gb': 3, 'ch_out_dram': 8, 'ch_in_rf': 256, 'batch_dram': 4, 'col_out_noc': 13, 'col_out_gb': 1, 'row_out_noc': 1, 'row_out_gb': 13, 'row_kernel_rf': 3, 'col_kernel_rf': 3}, {'ch_out_noc': 16, 'ch_out_gb': 4, 'ch_out_dram': 6, 'ch_in_rf': 192, 'batch_dram': 4, 'col_out_noc': 13, 'col_out_gb': 1, 'row_out_noc': 1, 'row_out_gb': 13, 'row_kernel_rf': 3, 'col_kernel_rf': 3}, {'ch_out_noc': 16, 'ch_out_gb': 4, 'ch_out_dram': 4, 'ch_in_rf': 192, 'batch_dram': 4, 'col_out_noc': 13, 'col_out_gb': 1, 'row_out_noc': 1, 'row_out_gb': 13, 'row_kernel_rf': 3, 'col_kernel_rf': 3}]

test_child=[{'ch_out_noc': 16, 'ch_out_gb': 6, 'ch_out_dram': 1, 'ch_in_rf': 3, 'batch_dram': 4, 'col_out_noc': 14, 'col_out_gb': 4, 'row_out_noc': 1, 'row_out_gb': 56, 'row_kernel_rf': 11, 'col_kernel_rf': 11}, {'ch_out_noc': 8, 'ch_out_gb': 8, 'ch_out_dram': 4, 'ch_in_rf': 48, 'batch_dram': 4, 'col_out_noc': 9, 'col_out_gb': 3, 'row_out_noc': 3, 'row_out_gb': 9, 'row_kernel_rf': 5, 'col_kernel_rf': 5}, {'ch_out_noc': 16, 'ch_out_gb': 3, 'ch_out_dram': 8, 'ch_in_rf': 256, 'batch_dram': 4, 'col_out_noc': 1, 'col_out_gb': 13, 'row_out_noc': 13, 'row_out_gb': 1, 'row_kernel_rf': 3, 'col_kernel_rf': 3}, {'ch_out_noc': 16, 'ch_out_gb': 4, 'ch_out_dram': 6, 'ch_in_rf': 192, 'batch_dram': 4, 'col_out_noc': 13, 'col_out_gb': 1, 'row_out_noc': 1, 'row_out_gb': 13, 'row_kernel_rf': 3, 'col_kernel_rf': 3}, {'ch_out_noc': 16, 'ch_out_gb': 4, 'ch_out_dram': 4, 'ch_in_rf': 192, 'batch_dram': 4, 'col_out_noc': 1, 'col_out_gb': 13, 'row_out_noc': 13, 'row_out_gb': 1, 'row_kernel_rf': 3, 'col_kernel_rf': 3}]


#TPU
#test_looporder=[
#                'col_out_rf','row_out_rf','batch_rf','ref_rf_in','ref_rf_out','ref_rf_we',\
#                'row_kernel_noc','col_kernel_noc','ch_in_noc','ch_out_noc',\
#                'ref_gb_we','ref_gb_in','col_out_gb','row_out_gb','ch_out_gb','ref_gb_out','ch_in_gb',\
#                'batch_dram','ch_out_dram','ch_in_dram',\
#]
test_looporder= ['ref_rf_in','ref_rf_we','ref_rf_out','col_out_rf','row_out_rf','batch_rf',\
                    'col_kernel_noc','row_kernel_noc','ch_out_noc','ch_in_noc',\
                    'ch_in_gb','ref_gb_in','ch_out_gb','ref_gb_out',\
                     'ref_gb_we','batch_dram','ch_out_dram','ch_in_dram']

test_child=[{'ch_out_noc': 2, 'ch_out_gb': 1, 'ch_out_dram': 48, 'ch_in_noc': 1, 'ch_in_gb': 3, 'ch_in_dram': 1, 'batch_rf': 1, 'batch_dram': 4, 'col_out_rf': 56, 'row_out_rf': 56, 'row_kernel_noc': 11, 'col_kernel_noc': 11}, {'ch_out_noc': 8, 'ch_out_gb': 1, 'ch_out_dram': 32, 'ch_in_noc': 1, 'ch_in_gb': 24, 'ch_in_dram': 2, 'batch_rf': 4, 'batch_dram': 1, 'col_out_rf': 27, 'row_out_rf': 27, 'row_kernel_noc': 5, 'col_kernel_noc': 5}, {'ch_out_noc': 12, 'ch_out_gb': 1, 'ch_out_dram': 32, 'ch_in_noc': 2, 'ch_in_gb': 64, 'ch_in_dram': 2, 'batch_rf': 4, 'batch_dram': 1, 'col_out_rf': 13, 'row_out_rf': 13, 'row_kernel_noc': 3, 'col_kernel_noc': 3}, {'ch_out_noc': 12, 'ch_out_gb': 1, 'ch_out_dram': 32, 'ch_in_noc': 2, 'ch_in_gb': 48, 'ch_in_dram': 2, 'batch_rf': 4, 'batch_dram': 1, 'col_out_rf': 13, 'row_out_rf': 13, 'row_kernel_noc': 3, 'col_kernel_noc': 3}, {'ch_out_noc': 8, 'ch_out_gb': 1, 'ch_out_dram': 32, 'ch_in_noc': 3, 'ch_in_gb': 32, 'ch_in_dram': 2, 'batch_rf': 4, 'batch_dram': 1, 'col_out_rf': 13, 'row_out_rf': 13, 'row_kernel_noc': 3, 'col_kernel_noc': 3}]
#test_child=[{'ch_out_noc': 2, 'ch_out_gb': 1, 'ch_out_dram': 48, 'ch_in_noc': 1, 'ch_in_gb': 3, 'ch_in_dram': 1, 'batch_rf': 1, 'batch_dram': 4, 'col_out_rf': 56, 'row_out_rf': 56, 'row_kernel_noc': 11, 'col_kernel_noc': 11}, {'ch_out_noc': 8, 'ch_out_gb': 1, 'ch_out_dram': 32, 'ch_in_noc': 1, 'ch_in_gb': 24, 'ch_in_dram': 2, 'batch_rf': 4, 'batch_dram': 1, 'col_out_rf': 27, 'row_out_rf': 27, 'row_kernel_noc': 5, 'col_kernel_noc': 5}, {'ch_out_noc': 12, 'ch_out_gb': 1, 'ch_out_dram': 32, 'ch_in_noc': 2, 'ch_in_gb': 64, 'ch_in_dram': 2, 'batch_rf': 4, 'batch_dram': 1, 'col_out_rf': 13, 'row_out_rf': 13, 'row_kernel_noc': 3, 'col_kernel_noc': 3}, {'ch_out_noc': 12, 'ch_out_gb': 1, 'ch_out_dram': 32, 'ch_in_noc': 2, 'ch_in_gb': 48, 'ch_in_dram': 2, 'batch_rf': 4, 'batch_dram': 1, 'col_out_rf': 13, 'row_out_rf': 13, 'row_kernel_noc': 3, 'col_kernel_noc': 3}, {'ch_out_noc': 8, 'ch_out_gb': 1, 'ch_out_dram': 32, 'ch_in_noc': 3, 'ch_in_gb': 32, 'ch_in_dram': 2, 'batch_rf': 4, 'batch_dram': 1, 'col_out_rf': 13, 'row_out_rf': 13, 'row_kernel_noc': 3, 'col_kernel_noc': 3}]
#print(arch_life(test_child,stride_list,tmp_hw_spec,df_order=test_looporder))
#exit()


hw_pool=generate_all_possible_hw(possible_hw_values)
print('hw space size: ', len(hw_pool))
hw_pool=filter_hw_pool(hw_pool,512*168*8+108*1024*8)
print('hw space size after prunning: ',len(hw_pool))
#df_order=arch_sample_results_df(5,\
#                               [[9, 8, 1, 6, 2, 1, 2, 0, 1, 0, 1, 5, 3, 1, 2, 0, 0], [1, 5, 0, 3, 0, 2, 0, 0, 1, 0, 5, 0, 3, 2, 1, 1, 0], [7, 3, 5, 5, 3, 4, 0, 2, 0, 0, 2, 1, 0, 2, 0, 0, 0], [3, 8, 3, 0, 3, 0, 0, 0, 1, 0, 4, 2, 3, 2, 0, 1, 0], [8, 3, 7, 4, 3, 4, 3, 0, 0, 0, 3, 2, 3, 2, 0, 1, 0]], \
#['row_out_rf', 'col_kernel_rf', 'ch_out_rf', 'batch_rf', 'col_out_rf', 'ref_rf_we', 'ref_rf_in', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_kernel_noc', 'ch_in_noc', 'row_out_noc', 'ch_out_noc'])

#df_config_dict=[{'ch_out_rf': 3, 'ch_out_noc': 4, 'ch_out_gb': 4, 'ch_out_dram': 2, 'ch_in_rf': 3, 'ch_in_noc': 1, 'ch_in_gb': 1, 'ch_in_dram': 1, 'batch_rf': 1, 'batch_gb': 4, 'batch_dram': 1, 'col_out_rf': 8, 'col_out_gb': 7, 'col_out_dram': 1, 'row_out_rf': 1, 'row_out_noc': 8, 'row_out_gb': 7, 'row_out_dram': 1, 'row_kernel_rf': 1, 'row_kernel_noc': 11, 'row_kernel_gb': 1, 'row_kernel_dram': 1, 'col_kernel_rf': 11, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 4, 'ch_out_noc': 8, 'ch_out_gb': 1, 'ch_out_dram': 8, 'ch_in_rf': 2, 'ch_in_noc': 4, 'ch_in_gb': 3, 'ch_in_dram': 2, 'batch_rf': 1, 'batch_gb': 4, 'batch_dram': 1, 'col_out_rf': 9, 'col_out_gb': 3, 'col_out_dram': 1, 'row_out_rf': 1, 'row_out_noc': 9, 'row_out_gb': 1, 'row_out_dram': 3, 'row_kernel_rf': 5, 'row_kernel_noc': 1, 'row_kernel_gb': 1, 'row_kernel_dram': 1, 'col_kernel_rf': 5, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 4, 'ch_out_noc': 4, 'ch_out_gb': 24, 'ch_out_dram': 1, 'ch_in_rf': 32, 'ch_in_noc': 2, 'ch_in_gb': 1, 'ch_in_dram': 4, 'batch_rf': 1, 'batch_gb': 2, 'batch_dram': 2, 'col_out_rf': 13, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_out_noc': 13, 'row_out_gb': 1, 'row_out_dram': 1, 'row_kernel_rf': 1, 'row_kernel_noc': 3, 'row_kernel_gb': 1, 'row_kernel_dram': 1, 'col_kernel_rf': 3, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 2, 'ch_out_noc': 4, 'ch_out_gb': 6, 'ch_out_dram': 8, 'ch_in_rf': 16, 'ch_in_noc': 2, 'ch_in_gb': 2, 'ch_in_dram': 3, 'batch_rf': 2, 'batch_gb': 1, 'batch_dram': 2, 'col_out_rf': 13, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_out_noc': 13, 'row_out_gb': 1, 'row_out_dram': 1, 'row_kernel_rf': 1, 'row_kernel_noc': 3, 'row_kernel_gb': 1, 'row_kernel_dram': 1, 'col_kernel_rf': 3, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 4, 'ch_out_noc': 2, 'ch_out_gb': 16, 'ch_out_dram': 2, 'ch_in_rf': 12, 'ch_in_noc': 4, 'ch_in_gb': 1, 'ch_in_dram': 4, 'batch_rf': 1, 'batch_gb': 4, 'batch_dram': 1, 'col_out_rf': 13, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_out_noc': 13, 'row_out_gb': 1, 'row_out_dram': 1, 'row_kernel_rf': 1, 'row_kernel_noc': 3, 'row_kernel_gb': 1, 'row_kernel_dram': 1, 'col_kernel_rf': 3, 'col_kernel_gb': 1, 'col_kernel_dram': 1}]
#print(arch_life(df_config_dict,stride_list,hw_pool[27],df_order=df_order))
#exit()

#baseline_gen(test_looporder,input_dnn,input_stride_list,tmp_hw_spec,n=200)
#exit()




tmp_hw_spec={ \
    'gb_vol':108*1024*8, \
    'rf_vol':512*8, \
    'num_pe':168, \
    'num_rf':168
}
#identify the most demanding layer
layer_break_down=fine_tune([[0]*sum([10,7])]*len(input_dnn),input_dnn,rf_noc_template[5],input_stride_list,tmp_hw_spec,n=200)[2]
most_demanding_layer=np.argmin(layer_break_down) 
#most_demanding_layer=1
print('most demanding layer is: ',most_demanding_layer)
#cycle_scaling=0.1         #change back
#mutation_cycle_scaling=0.1



#generate hardware space
#multi_thread this process and use exhuastive approach
#check if hw_spec withing the bondary of budget: if eval_func(*hw)~budget

#if so add to the pool
#if not pass
hw_score_pool=Queue()
cur_best_hw={}
cur_best_score=-9e11
back_up_pool=[]
#set most demanding layer; comment out when doing no_hw
#input_dnn=[input_dnn[most_demanding_layer]]
#input_stride_list=[input_stride_list[most_demanding_layer]]

#set the best hw for no_hw, comment out when doing hw search
#hw_pool=[{'gb_vol': 108*1024*8, 'rf_vol': 4096, 'num_pe': 168, 'num_rf': 168}] 

#hw_pool=copy.deepcopy(hw_pool[0:len(hw_pool)//5])
hw_pool=copy.deepcopy(hw_pool) #change back from hw
#11,1
hw_pool=copy.deepcopy([hw_pool[40],hw_pool[38],hw_pool[34],hw_pool[30],hw_pool[39]])
highest_rf_pool=[]
for tmp_hw_spec in hw_pool:
    def search(input_rf,cycle_scaling,mutation_cycle_scaling):
        pop_list=[]
        best_pop_cluster=[]
        invalid_hw_design=False
        start_t=time.time()
        whole_dnn_score=0 
        for layer in range(len(input_dnn)):
            print('layer ',layer)
            dnn=[copy.deepcopy(input_dnn[layer])]
            stride_list=[input_stride_list[layer]]
            #initial points
            reference_starting_points=[]
            start_point=[]
            #layer list need to be changed
            for i in range(sum([10,7])):
                start_point.append(0)
            reference_starting_points.append(start_point)

            #fine_tune([[5, 2, 2, 0, 1, 1, 0, 9, 3, 6, 6, 1, 4, 1, 0, 1, 0, 5, 4, 3, 0, 0, 0, 0]],dnn,input_rf,stride_list,tmp_hw_spec,n=200)

            for i in reference_starting_points:
                pop_list.append(i)
                child=sample_results_df(pop_list[-1],input_rf)

    #            print('start point score:', random_life([child,child,child,child,child,child], \
    #                                                    [copy.deepcopy(dnn[0]),copy.deepcopy(dnn[0]),copy.deepcopy(dnn[0]),copy.deepcopy(dnn[0]),copy.deepcopy(dnn[0]),copy.deepcopy(dnn[0])], \
    #                                                    sample_num,[1,1,1,1,1,1],1,tmp_hw_spec,n=int(50*cycle_scaling)))
            #exit()



            #generate initial population
            while len(pop_list)<int(max_pop):
                pop_list.append(lo_random_pop())
                
            #get the score for the initial population
            print('evaluating the initial population')
            score_board=[]
            #if we keep getting out of boundary score, then most likely this design is not valid
            max_trial_ratio=0.2
            max_trials=0
            #save looporder points for projection
            np_x=np.reshape(pop_list,(len(pop_list),len(pop_list[0])))
            np_y=[]
            for i in range(0,len(pop_list)):
                child=sample_results_df(pop_list[i],input_rf)
                score=random_life(child,dnn,sample_num,stride_list,1,tmp_hw_spec,n=int(50*cycle_scaling))
                np_y.append(score)
                #invalid_hw_design
                if max_trials >= max_trial_ratio*len(pop_list):
                    invalid_hw_design=True
                    break
                if score <=-9e11:
                    max_trials+=1
                #print('child score',score)
                score_board.append(score)
            if invalid_hw_design:
                print('Invalid hardware design')
                break
            pop_list,score_board=pop_ranking(pop_list,score_board)
            #print(score_board)
            #np_y=np.reshape(np_y,(len(np_y),1))
            #sio.savemat('looporder_data.mat',{'x':np_x,'y':np_y})



            #exit()
            #ev for loop order
            print('life cycles started')
            #iterate for n cycles
            score1=0
            tmp_time=time.time()
            for _ in range(int(40*mutation_cycle_scaling)):
                #continue                                                        #change back...
                #if not saturate birth and mutate
                if len(pop_list) < max_pop:         
                    #print('generating pop')
                    size=int(2*dying_rate*(max_pop))                                   # right now birth control is through max_pop; can be done through current pop
                    if size%2 !=0:
                        size+=1
                    pos=np.random.randint(size,size=size)                                # only top "size" number of pop have rights of birth
                    
                    #new born
                    for i in range(0,len(pos),2):                                        #You give the lower rankings right to give birth???  i changed it
                        
                        new_child=lo_give_birth(pop_list[pos[i]],pop_list[pos[i+1]])        #will try no birth only mutate next                                            
                        new_child=lo_mutate(new_child,prop_m)            
                        pop_list.append(new_child)
                        new_child=sample_results_df(new_child,input_rf)
                        #get the scores for the new born
                        #merged_child=merge_noc(new_child,noc)
                        merged_child=new_child
                        score_board.append(random_life(merged_child,dnn,sample_num,stride_list,1,tmp_hw_spec,n=int(50*cycle_scaling)))
                #else kill and birth and mutate
                else:
                    #print(score_board)
                    #print('killing')
                    #rank
                    #ganky way of sorting pop_list
                    pop_list,score_board=pop_ranking(pop_list,score_board)
                    #kill
                    pop_list = pop_list[0:int((max_pop)*(1-dying_rate))]
                    score_board=score_board[0:int((max_pop)*(1-dying_rate))]
                    score1=score_board[0]
                    whole_dnn_score+=score1
                    print('highest score: ',score1)
                    highest_child=sample_results_df(pop_list[0],input_rf)
                    print(highest_child)
                    #print(time.time()-tmp_time)
                    #tmp_time=time.time()
           
            best_pop_cluster.append(pop_list[0:k])
            pop_list=[]
            print('one layer takes', time.time()-start_t)
        return whole_dnn_score,invalid_hw_design,best_pop_cluster

    #distributed in a multiprocessing fashion
    print('RF/NOC template evaluation starts')
    rf_noc_time=time.time()
    whole_dnn_score_rf=[]
    noc_rf_q=Queue()
    def worker_rf_noc_template(work_load):
        #the first element of the work load indicate the index of the workload
        work_load=copy.deepcopy(work_load)
        rf_noc_template_copy=copy.deepcopy(rf_noc_template)
        rf_noc_template_batch=[]
        for template_idx in work_load[1]:
            rf_noc_template_batch.append(rf_noc_template_copy[template_idx])
        tmp_whole_dnn_score_rf=[]
        for input_rf in rf_noc_template_batch:
            print('new_template')
            whole_dnn_score=search(copy.deepcopy(input_rf),0.15,0.15)[0]
            tmp_whole_dnn_score_rf.append(whole_dnn_score)
        try:
            noc_rf_q.put((work_load[0],tmp_whole_dnn_score_rf),False)
        except Empty:
            raise Exception("There is no room in the queue in rf template stage")
    if not noc_rf_q.empty():
        print('Some Trash in the noc_rf_template Queue')
        exit()
#    work_load= [[0,rf_noc_template[0:2]]]

    work_load=[[0,list(range(0,1))], \
               [1,list(range(1,2))], \
               [2,list(range(2,3))], \
               [3,list(range(3,4))], \
               [4,list(range(4,5))], \
               [5,list(range(5,6))], \
]



#    work_load=[[0,list(range(0,1))], \
#               [1,list(range(1,2))], \
#               [2,list(range(2,3))], \
#               [3,list(range(3,4))], \
#               [4,list(range(4,5))], \
#               [5,list(range(5,6))], \
#               [6,list(range(6,7))], \
#               [7,list(range(7,8))], \
#               [8,list(range(8,9))], \
#               [9,list(range(9,10))], \
#               [10,list(range(10,11))], \
#               [11,list(range(11,12))], \
#               [12,list(range(12,13))], \
#               [13,list(range(13,14))], \
#               [14,list(range(14,15))], \
#               [15,list(range(15,16))], \
#               [16,list(range(16,17))], \
#               [17,list(range(17,18))], \
#               [18,list(range(18,19))], \
#               [19,list(range(19,20))], \
#               [20,list(range(20,21))], \
#               [21,list(range(21,22))], \
#               [22,list(range(22,23))], \
#               [23,list(range(23,24))], \
#]

    processes = [multiprocessing.Process(target=worker_rf_noc_template, args=([load])) for load in work_load]

    tmp_dump_yard={}

    for p in processes:
        p.start()
        time.sleep(2)

    time.sleep(10)
    while not noc_rf_q.empty():
        tmp_batch=noc_rf_q.get()
        tmp_dump_yard[tmp_batch[0]]=tmp_batch[1]

    for p in processes:
        p.join()

    #too many dump_yard...
    while not noc_rf_q.empty():
        tmp_batch=noc_rf_q.get()
        tmp_dump_yard[tmp_batch[0]]=tmp_batch[1]

    load_size=len(tmp_dump_yard)
    for load_idx in range(load_size):
        whole_dnn_score_rf+=tmp_dump_yard[load_idx]
    print('template_scores',whole_dnn_score_rf)
    highest_rf=rf_noc_template[np.argmax(whole_dnn_score_rf)]
    highest_rf_pool.append(highest_rf)
    print('RF/NOC template evaluation takes ', time.time()-rf_noc_time)

print('RF templates decision finished')
print('each hw has a corresponding rf/noc template? ', len(highest_rf_pool)==len(hw_pool))
print('Highest rf',highest_rf_pool)

def hw_worker(load):
    hw_idx=load[0]
    hw_pool=load[1]
    tmp_hw_scores=[]
    for sub_hw_idx, tmp_hw_spec in enumerate(hw_pool):
        highest_rf=highest_rf_pool[hw_idx*len(hw_pool)+sub_hw_idx]   
        #highest_rf=rf_noc_template[0]           #change back from hw
        results=search(highest_rf,1,1) #change back from hw
        invalid_hw_design=results[1]
        best_pop_cluster=results[2]
        if invalid_hw_design:
            print('invalid_hw_design !!!')
            continue
        #re group pop_cluster
        #here needs to be fixed later, because it allows for more combo
        pop_list=[]
        for i in range(len(best_pop_cluster[0])):
            pop_list.append([])
            for j in range(len(input_dnn)):
                pop_list[i].append(best_pop_cluster[j][i])


        print(pop_list)





        dnn=input_dnn
        stride_list=input_stride_list
        #fine tune loop order based
        best_score=[]
        best_dict=[]
        best_layer_breakdown=[]
        for i in pop_list[0:k]:
            print(i)
            bscore=fine_tune(i,dnn,highest_rf,stride_list,tmp_hw_spec,n=int(250))   #change back from hw
            best_score.append(bscore[0])
            best_dict.append(bscore[1])
            best_layer_breakdown.append(bscore[2])
            print('best score', best_score)
            print('break down',best_layer_breakdown[-1])
        best_idx=[i for best_score,i in sorted(zip(best_score,list(range(k))),reverse=True)]
        print('best is',best_idx[0])
        print('best loop order',pop_list[best_idx[0]])
        print('best config dict',best_dict[best_idx[0]])
        print('best break down',best_layer_breakdown[best_idx[0]])
        end_time=time.time()-start_time
        print(end_time/3600)
        #budget exceeded
        print('current best score :',best_score[best_idx[0]])
        tmp_hw_scores.append(best_score[best_idx[0]])
    hw_score_pool.put((hw_idx,tmp_hw_scores))


if not hw_score_pool.empty():
    print('Some Trash in the HW Queue')
    exit()

#work_load=[[0,hw_pool[0:4]], \
#           [1,hw_pool[4:8]], \
#           [2,hw_pool[8:12]], \
#           [3,hw_pool[12:16]], \
#           [4,hw_pool[16:20]], \
#           [5,hw_pool[20:24]], \
#           [6,hw_pool[24:28]], \
#           [7,hw_pool[28:32]], \
#           [8,hw_pool[32:36]], \
#           [9,hw_pool[36:40]], \

#]
work_load=[
           [0,hw_pool[0:1]], \
           [1,hw_pool[1:2]], \
           [2,hw_pool[2:3]], \
]

processes = [multiprocessing.Process(target=hw_worker, args=([load])) for load in work_load]
tmp_dump_yard={}
hw_score_pool_list=[]
for p in processes:
    p.start()
    time.sleep(2)
time.sleep(10)
#clear pool as much as possible
while not hw_score_pool.empty():
    tmp_batch=hw_score_pool.get()
    tmp_dump_yard[tmp_batch[0]]=tmp_batch[1]
for p in processes:
    p.join()
while not hw_score_pool.empty():
    tmp_batch=hw_score_pool.get()
    tmp_dump_yard[tmp_batch[0]]=tmp_batch[1]


load_size=len(tmp_dump_yard)
print('load_size', load_size)
for load_idx in range(load_size):
    print(tmp_dump_yard[load_idx])
    hw_score_pool_list+=tmp_dump_yard[load_idx]
print('hardware-scores: ', hw_score_pool_list)
# print('worst score', worst_score)
