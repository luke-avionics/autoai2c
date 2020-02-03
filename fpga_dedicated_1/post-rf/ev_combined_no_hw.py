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


def _gcd(l):
    def find_gcd(x, y): 
        while(y): 
            x, y = y, x % y 
      
        return x 

      
    num1=l[0] 
    num2=l[1] 
    gcd=find_gcd(num1,num2) 
      
    for i in range(2,len(l)): 
        gcd=find_gcd(gcd,l[i]) 
    return gcd


def fpga_tiling_generator(input_dnn,buffer_limit,dsp_limit,bit_width=16):
    
    tmp_layer=1
    ch_in=[]
    ch_out=[]
    row_out=[]
    col_out=[]
    col_kernel=[]
    row_kernel=[]
    kernel_3_index=[]
    layer_ctr=0
    for layer in input_dnn:
        ch_in.append(layer[1]['ch_in'][0])
        ch_out.append(layer[1]['ch_out'][0])    
        row_out.append(layer[1]['row_out'][0])
        col_out.append(layer[1]['col_out'][0])
        row_kernel.append(layer[1]['row_kernel'][0])
        col_kernel.append(layer[1]['col_kernel'][0])
        if layer[1]['row_kernel'][0] ==3:
            kernel_3_index.append(layer_ctr)
        layer_ctr+=1

    try:
        ch_in.remove(3)
    except:
        pass
    ch_out_bram=_gcd(ch_out)
    ch_in_bram=_gcd(ch_in)
    
    col_out_bram=_gcd(col_out)
    row_out_bram=_gcd(row_out)
    row_kernel_bram=max(row_kernel)
    col_kernel_bram=max(col_kernel)





    #buffer size calc
    output_b_size=ch_out_bram*col_out_bram*row_out_bram*bit_width
    input_b_size=ch_in_bram*row_out_bram*col_out_bram*bit_width
    weight_b_size=ch_in_bram*ch_out_bram*col_kernel_bram*row_kernel_bram*bit_width
    print((output_b_size+input_b_size+weight_b_size)/8/1024, 'kB buffer used')
    if (output_b_size+input_b_size+weight_b_size) > buffer_limit:
        raise Exception('buffer size exceeded') 
    #    
    dram_tiling_head=[]
    for layer in input_dnn:
        dram_tiling_head.append({})
        dram_tiling_head[-1]['ch_out_dram']=max(layer[1]['ch_out'][0]/ch_out_bram,1)
        dram_tiling_head[-1]['ch_in_dram']=max(layer[1]['ch_in'][0]/ch_in_bram,1)
        dram_tiling_head[-1]['row_out_dram']=max(layer[1]['row_out'][0]/row_out_bram,1)
        dram_tiling_head[-1]['col_out_dram']=max(layer[1]['col_out'][0]/col_out_bram,1)
        dram_tiling_head[-1]['col_kernel_dram']=max(layer[1]['col_kernel'][0]/col_kernel_bram,1)
        dram_tiling_head[-1]['row_kernel_dram']=max(layer[1]['row_kernel'][0]/row_kernel_bram,1)
        dram_tiling_head[-1]['batch_dram']=1

    noc_template=[['col_kernel_noc','row_kernel_noc','ch_in_noc','ch_out_noc'], \
                      ['col_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      ['row_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      ['row_out_noc','col_out_noc','ch_out_noc'], \
                      ]

    bram_noc_tiling=[]
        
    #1
    col_kernel_noc=r_factors(col_kernel_bram)
    row_kernel_noc=r_factors(row_kernel_bram)
    ch_in_noc=r_factors(ch_in_bram)
    ch_out_noc=r_factors(ch_out_bram)
    for i in col_kernel_noc:
        for j in row_kernel_noc:
            for k in ch_in_noc:
                for l in ch_out_noc:
                    if i*j*k*l<=dsp_limit:
                        bram_noc_tiling.append({})
                        bram_noc_tiling[-1]['batch_gb']=1
                        bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/l
                        bram_noc_tiling[-1]['ch_out_noc']=l
                        bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram/k
                        bram_noc_tiling[-1]['ch_in_noc']=k
                        bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram/j
                        bram_noc_tiling[-1]['row_kernel_noc']=j
                        bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram/i
                        bram_noc_tiling[-1]['col_kernel_noc']=i
                        bram_noc_tiling[-1]['row_out_gb']=row_out_bram
                        bram_noc_tiling[-1]['col_out_gb']=col_out_bram

    #2
    col_out_noc=r_factors(col_out_bram)
    for i in col_kernel_noc:
        for j in col_out_noc:
            for k in ch_in_noc:
                for l in ch_out_noc:
                    if i*j*k*l<=dsp_limit:
                        bram_noc_tiling.append({})
                        bram_noc_tiling[-1]['batch_gb']=1
                        bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/l
                        bram_noc_tiling[-1]['ch_out_noc']=l
                        bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram/k
                        bram_noc_tiling[-1]['ch_in_noc']=k
                        bram_noc_tiling[-1]['col_out_gb']=col_out_bram/j
                        bram_noc_tiling[-1]['col_out_noc']=j
                        bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram/i
                        bram_noc_tiling[-1]['col_kernel_noc']=i
                        bram_noc_tiling[-1]['row_out_gb']=row_out_bram
                        bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram



    #3
    for i in row_kernel_noc:
        for j in col_out_noc:
            for k in ch_in_noc:
                for l in ch_out_noc:
                    if i*j*k*l<=dsp_limit:
                        bram_noc_tiling.append({})
                        bram_noc_tiling[-1]['batch_gb']=1
                        bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/l
                        bram_noc_tiling[-1]['ch_out_noc']=l
                        bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram/k
                        bram_noc_tiling[-1]['ch_in_noc']=k
                        bram_noc_tiling[-1]['col_out_gb']=col_out_bram/j
                        bram_noc_tiling[-1]['col_out_noc']=j
                        bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram/i
                        bram_noc_tiling[-1]['row_kernel_noc']=i
                        bram_noc_tiling[-1]['row_out_gb']=row_out_bram
                        bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram

    #4
    row_out_noc=r_factors(row_out_bram)
    for i in col_out_noc:
        for j in row_out_noc:
            for k in ch_out_noc:
                if i*j*k <= dsp_limit:
                    bram_noc_tiling.append({})
                    bram_noc_tiling[-1]['batch_gb']=1
                    bram_noc_tiling[-1]['col_out_gb']=col_out_bram/i
                    bram_noc_tiling[-1]['col_out_noc']=i
                    bram_noc_tiling[-1]['row_out_gb']=col_out_bram/j
                    bram_noc_tiling[-1]['row_out_noc']=j
                    bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/k
                    bram_noc_tiling[-1]['ch_out_noc']=k
                    bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram
                    bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram
                    bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram
    result_tiling_pool=[]
    for i in bram_noc_tiling:
        result_tiling_pool.append(copy.deepcopy(dram_tiling_head))
        for j in range(len(result_tiling_pool[-1])):
            result_tiling_pool[-1][j]=dict(list(result_tiling_pool[-1][j].items())+list(i.items()))

    for i in result_tiling_pool:
        for j in kernel_3_index:
            try:
                if i[j]['row_kernel_gb']==5:
                    i[j]['row_kernel_gb']=3
            except:
                pass
            try:
                if i[j]['row_kernel_noc']==5:
                    i[j]['row_kernel_noc']=3
            except:
                pass
            try:
                if i[j]['col_kernel_gb']==5:
                    i[j]['col_kernel_gb']=3
            except:
                pass
            try:
                if i[j]['col_kernel_noc']==5:
                    i[j]['col_kernel_noc']=3
            except:
                pass
    for i in result_tiling_pool:
        try:
            i[0]['ch_in_noc']=1
        except:
            pass
        i[0]['ch_in_gb']=3
        i[0]['ch_in_dram']=1

    return (result_tiling_pool[0])
#print(fpga_tiling_generator(input_dnn,16*1024*1024,800))
#print(_gcd([8,4,4,4,4]))
#exit()



def eval_func(hw_spec):
    eval_val=hw_spec['gb_vol']+hw_spec['num_pe']*hw_spec['rf_vol']
    return eval_val


def random_life(df_order, tiling_pool,input_stride_list,hw_spec,return_best_dict=False):
    #after smapling a loop-order, routine to optimize tiling factors to get the energy feedback     
    score_board=[]    
    df_order=copy.deepcopy(df_order)    
    for i in tiling_pool:
       	tiling_for_all_layers=[]
        for _ in range(len(df_order)):
            tiling_for_all_layers.append(i)
        score_board.append(arch_life(tiling_for_all_layers,input_stride_list,hw_spec,df_order=df_order)[0])
    score_pair=sorted(zip(score_board,list(range(len(score_board)))),reverse=True)
    
    if return_best_dict:
        return score_pair[0][0], tiling_pool[score_pair[0][1]]    
    else:
        return  score_pair[0][0]





#def random_life(df_order,dnn,num_samples,stride_list,init_multiplier,hw_spec,n=200,return_best_dict=False):
#    #after smapling a loop-order, routine to optimize tiling factors to get the energy feedback


#    #df_order: input loop-order
#    #dnn: user input DNN specs
#    #num_samples: max_number of population during tiling factor optimization
#    #stride_list: stride numbers for DNN layers
#    #init_multiplier: initial population multiplier in tiling factor optimization
#    #hw_spec: input hw_spec
#    #n: number of iteration to go through in tiling factor optimization 
#    #return_best_dict: wether to return the detail results, otherwise only score(penalty) returned

#    df_order=copy.deepcopy(df_order)
#    dnn=copy.deepcopy(dnn)
#    layer_wise=(type(df_order[0])==list)
#    if layer_wise:
#        #generate reference df_order
#        ref_df_order=[]
#        for j in range(len(dnn)):
#            ref_df_order.append([])
#            for i in df_order[j]:
#                if 'ref' not in i:
#                    ref_df_order[j].append(i)
#        #generate net_arch
#        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#        #right now because df_order for each layer has the same amount of components
#        #they are different in the order of the components
#        #we use value 1 to deem one component is not in looporder
#        net_arch=gen_net_arch(ref_df_order[0],dnn)
#        
#    else:
#        #generate reference df_order
#        ref_df_order=[]
#        for i in df_order:
#            if 'ref' not in i:
#                ref_df_order.append(i)
#        #generate net_arch
#        net_arch=gen_net_arch(ref_df_order,dnn)    
#    #initial max_num pop
#    ev_dict1=ev_dict(stride_list,net_arch,ref_df_order,max_pop=num_samples,true_df_order=df_order,hw_spec=hw_spec)
#    #optimize for n cycles
#    ev_dict1.search(n=n,init_multiplier=init_multiplier)       #TODO: add search for n cycles or search for convergence?
#    #return the score
#    score=ev_dict1.best_score
#    if return_best_dict:
#        return score,ev_dict1.best_dict,ev_dict1.best_layer_breakdown
#    else:
#        return score


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
input_dnn=[\
[1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[128,0],'ch_in':[64,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[128,0],'ch_in':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[512,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
]
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
#input_dnn=[\
#[4, {'ch_out':[96,0],'ch_in':[3,0],'batch':[4,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[11,0],'col_kernel':[11,0]}],\
#
#[1,{'ch_out':[256,0],'ch_in':[48,0],'batch':[4,0],'col_out':[27,0],'row_out':[27,0],'row_kernel':[5,0],'col_kernel':[5,0]}],\
#
#[1,{'ch_out':[384,0],'ch_in':[256,0],'batch':[4,0],'col_out':[13,0],'row_out':[13,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#
#[1,{'ch_out':[384,0],'ch_in':[192,0],'batch':[4,0],'col_out':[13,0],'row_out':[13,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#
#[1,{'ch_out':[256,0],'ch_in':[192,0],'batch':[4,0],'col_out':[13,0],'row_out':[13,0],'row_kernel':[3,0],'col_kernel':[3,0]}]\
#]




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



input_stride_list=[1,1,1,1,1]
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




input_stride_list=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

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
#fpga dedicated zed
tmp_hw_spec={\
    'gb_vol':512*1024*8, \
    'rf_vol':512*8, \
    'num_pe':220, \
    'num_rf':220
}

#fpga dedicated 706
tmp_hw_spec={\
    'gb_vol':16*1024*1024, \
    'rf_vol':512*8, \
    'num_pe':824, \
    'num_rf':824
}

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
hw_pool=copy.deepcopy([hw_pool[39],hw_pool[37],hw_pool[33],hw_pool[29],hw_pool[38]])
hw_pool=[tmp_hw_spec]
highest_rf_pool=[]

for tmp_hw_spec in hw_pool:
    tiling_pool=fpga_tiling_generator(input_dnn,tmp_hw_spec['gb_vol'],tmp_hw_spec['num_pe'])

    def search(input_rf,cycle_scaling,mutation_cycle_scaling):
        pop_list=[]
        best_pop_cluster=[]
        invalid_hw_design=False
        start_t=time.time()
        whole_dnn_score=0 
        dnn=copy.deepcopy(input_dnn)
        stride_list=copy.deepcopy(input_stride_list)
        reference_starting_points=[]
        start_point=[]
        for i in range(sum([10,7])):
            start_point.append(0)
        
        reference_starting_points.append([])
        for _ in range(len(dnn)):
            reference_starting_points[-1].append(start_point)
        for i in reference_starting_points:
            pop_list.append(i)
            child=[]
            for j in range(len(dnn)):
                child.append(sample_results_df(pop_list[-1][j],input_rf))

        #generate initial population
        while len(pop_list)<int(max_pop):
            pop_list.append(arch_lo_random_pop(len(dnn)))


        #get the score for the initial population
        print('evaluating the initial population')
        score_board=[]
        for i in range(0,len(pop_list)):
            child=[]
            for j in range(len(dnn)):
                child.append(sample_results_df(pop_list[i][j],input_rf))
            score=random_life(child, tiling_pool, stride_list, tmp_hw_spec)
            score_board.append(score)
        pop_list,score_board=pop_ranking(pop_list,score_board)



        print('life cycles started')
        score1=0
        for _ in range(int(40*mutation_cycle_scaling)):
            #if not saturate birth and mutate
            if len(pop_list) < max_pop:         
                #print('generating pop')
                size=int(2*dying_rate*(max_pop))                                   # right now birth control is through max_pop; can be done through current pop
                if size%2 !=0:
                    size+=1
                pos=np.random.randint(size,size=size)                                # only top "size" number of pop have rights of birth
                #new born
                for i in range(0,len(pos),2):                                        #You give the lower rankings right to give birth???  i changed it
                    tmp_rand=np.random.rand()
                    p1=pop_list[pos[i]]
                    p2=pop_list[pos[i+1]]

                    if tmp_rand < 0.33:
                        new_child=[]
                        if(len(p1)!=len(p2)):
                            raise Exception('looporder layer length not consistent')
                        for p_layer in range(len(p1)):
                            tmp_child=(lo_give_birth(p1[p_layer],p2[p_layer]))
                            tmp_child=lo_mutate(tmp_child,prop_m)
                            new_child.append(tmp_child)
                        pop_list.append(new_child)

                    elif tmp_rand< 0.66:
                        new_child=[]
                        for p_layer in range(len(p1)):
                            new_child.append(lo_mutate(p1[p_layer],prop_m))
                        pop_list.append(new_child)
                    else:
                        new_child=[]
                        for p_layer in range(len(p1)):
                            new_child.append(lo_mutate(p2[p_layer],prop_m))
                        pop_list.append(new_child)
                        
                    new_child_str=[]
                    for p_layer in range(len(dnn)):
                        new_child_str.append(sample_results_df(new_child[p_layer],input_rf))
                    score_board.append(random_life(new_child_str, tiling_pool, stride_list, tmp_hw_spec))

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
                #best_loop_order
                new_child_str=[]
                for p_layer in range(len(dnn)):
                    new_child_str.append(sample_results_df(pop_list[0][p_layer],input_rf))
                print(new_child_str)
                #print(time.time()-tmp_time)
                #tmp_time=time.time()
        return whole_dnn_score

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
            whole_dnn_score=search(copy.deepcopy(input_rf),1,1)[0]
            tmp_whole_dnn_score_rf.append(whole_dnn_score)
        try:
            noc_rf_q.put((work_load[0],tmp_whole_dnn_score_rf),False)
        except Empty:
            raise Exception("There is no room in the queue in rf template stage")


    if not noc_rf_q.empty():
        print('Some Trash in the noc_rf_template Queue')
        exit()

    work_load=[[0,list(range(0,1))], \
               [1,list(range(1,2))], \
               [2,list(range(2,3))], \
               [3,list(range(3,4))], \
              ]

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






