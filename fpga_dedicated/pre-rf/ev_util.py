from random import random, randint,shuffle
import numpy as np
import test_for_eyeriss as simnas
import time
from itertools import combinations,permutations
import copy
from  multiprocessing import Queue
import multiprocessing
import math

default_hw={ \
    'gb_vol':108*1024*8, \
    'rf_vol':6893, \
    'num_pe':168, \
    'num_rf':168
}

##############################
#shared util funcs
#############################

def life_eval(actions,stride,hw_spec,df_order=None):
    #function to query chip_estimator and get energy+latency feedback

    #actions: tiling factors for a specific loop-order
    #stride: the stride number for this CONV layer operation
    #hw_spec: hw specs for evaluation
    #df_order: loop-order for evaluation 
    #           !!!!if not provided PLS provide it in chip_estimator
    #           !!!!legacy functionality, so always try to provide specific loop-order here
    try:
        #input isolation
        input_actions=dict(actions)
        if df_order:
            input_df_order=list(df_order)
        else:
            input_df_order=None
        ene_results=simnas.sample_energy(input_actions,stride,hw_spec,input_df_order=input_df_order)
        penalty=-ene_results[0]*1e-8-ene_results[1]*100
        #print(ene_results[0],ene_results[1])
    #if design hw constraint exceeded, 
    #if exceeded return extremely large penalty
    except Exception as e:
        if 'resource' in str(e):
            pass
        else:
            print('error:',e)
        #print('child:',actions)
        penalty=-9e12                                  #very strong penalty to over budget
    return penalty

actions= {'ch_out_rf':16, 'ch_in_rf':2, 'row_kernel_rf':5, 'ref_rf_we':64, 'row_out_rf':27, 'ref_rf_in':16, 'batch_rf':1,\
            'ref_rf_out':64, 'col_kernel_noc':5, 'ch_in_noc':1, 'col_out_noc':27, 'ch_out_noc':1,\
            'ref_gb_we':64, 'ch_out_gb':4, 'ref_gb_in':64, 'ch_in_gb':24,\
            'ref_gb_out':64, 'col_out_dram':1, 'ch_out_dram':4, 'batch_dram':4,\
            }    
stride=1
hw_spec=default_hw
df_order=['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf', 'ref_rf_we',\
                'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc',\
                'ref_gb_we', 'ch_out_gb', 'ref_gb_in',  'ch_in_gb', 'ref_gb_out', \
                'col_out_dram', 'ch_out_dram', 'batch_dram'\
               ]
life_eval(actions,stride,hw_spec,df_order=df_order)
    
def arch_life(child,input_stride_list,hw_spec,df_order=None):
    #evaluate the energy consumption for all layers in one network

    #child: a list of tiling factors dictionary --- [{},{},{},....],each corresponding to a layer
    #input_stride_list: a list of stride, each corresponding to a layer
    #hw_spec: hw spec used
    #df_order: loop-order in list of list, each inner list --- one single layer
    
    #return: total penalty for all layers and breakdown for each layer
    score=0
    #input isolation
    layer_wise=(type(df_order[0])==list)
    layer_break_down=[]
    stride_list=list(input_stride_list)
    for i in range(len(child)):
        if not layer_wise:
            layer_score=life_eval(child[i],stride_list[i],hw_spec,df_order=df_order)                                                                  #######
        else:
            layer_score=life_eval(child[i],stride_list[i],hw_spec,df_order=df_order[i])
        if layer_score > -9e12:
            layer_break_down.append(layer_score)
            score+=layer_score   
        else:
            #print('layer:',i)
            layer_break_down.append(-9e12)
            return -9e12,layer_break_down
    return score,layer_break_down
    
def pop_ranking(pop_list,score_board):
    #rank population list according to score_board, not working if directly using sorted(zip())
    #too ganky
    pop_indices=list(range(len(pop_list)))
    results = [(pop_indx,score_num) for score_num,pop_indx in sorted(zip(score_board,pop_indices),reverse=True)]
    pop_indices=[results[x][0] for x in list(range(len(results)))]
    tmp_pop=[]
    #maybe can get rid of the deepcopy
    for i in range(len(pop_list)):
        tmp_pop.append(pop_list[pop_indices[i]])
    pop_list=copy.deepcopy(tmp_pop)   
    score_board=[results[x][1] for x in list(range(len(results)))]
    return pop_list,score_board
    
    
#############################
#df order  specific
#############################
    
def lo_random_pop(layer_list=[10,7]):
    #randomly sample a specific loop-order
    #layer_list: denoting the number of components in each memory level, with NoC,RF levels excluded, as they are in a list pool to iterate 
    #            SRAM(GB) level
    #            DRAM level 
    #right now does not handle the repetitive population
    actions=[]
    for idx,size in enumerate(layer_list):
        for i in range(size-1,0,-1):
            actions.append(randint(0,i))
        actions.append(0)
    return actions
    
#print(random_pop([7,5,4,4]))
#exit()

def lo_give_birth(input_str1,input_str2):
    #combine two randomly generated loop-oders
    #with possiblity:
    #                 1. half features from str1 and half features from str2
    #                 2. entirely from str1
    #                 3. entirely from str2
    str1=list(input_str1)
    str2=list(input_str2)
    #random number to decide which part of the parents to inherit
    num=randint(0,2)
    if num==0:
        str3=str1[0:int(len(str1)/2)]+str2[int(len(str2)/2):]
    elif num==1:
        str3=str1
    else:
        str3=str2
    return str3

def lo_mutate(input_str1,prop,layer_list=[10,7]):
    #randomly permute some features of a input loop-oder
    #input_str1: input loop-order
    #prop: probability to mutate
    #layer_list: denoting number of components in each memory level
    str1=list(input_str1)
    if random()<=prop:
        #how many features under risk of mutation                              #currently 25 percent of features under risk of mutation
        size=int(0.25*(sum(layer_list)))
        if size<1:
            size=1
        pos=list(range(0,sum(layer_list)))
        #randomly pick 'size' of features
        shuffle(pos,random=random)
        pos=pos[0:size]
        ref_pos=[]
        #features number range for each feature 
        for i in layer_list:
            ref_pos+=list(range(i-1,0,-1))
            ref_pos+=[0]
        for i in pos:
            if ref_pos[i]==0:
                continue
            else:
                str1=str1[0:i]+[randint(0,ref_pos[i])]+str1[i+1:]
    return str1


#noc_template to be considered 
noc_template=[['col_kernel_noc','row_kernel_noc','ch_in_noc','ch_out_noc'], \
                      ['col_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      ['row_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      ['row_kernel_noc','ch_in_noc','row_out_noc','ch_out_noc'], \
                      ['col_kernel_noc','ch_in_noc','row_out_noc','ch_out_noc'], \
                      ['row_out_noc','col_out_noc','ch_out_noc'], \
                      ]

rf_noc_template=noc_template



def sample_results_df(input_actions,input_rf,layer_list=[10,7]):
    #translate index fomrat loop-order to string format 
    #input_actions: index format loop-order, output from lo_random_pop, lo_give_birth, lo_mutate
    #input_rf: rf_noc_template to take
    #layer_list: see above...
    actions=list(input_actions)
    df_dict ={
              #0:['col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc','row_kernel_noc','row_out_noc','batch_noc'], \
              0:['ref_gb_we','ch_out_gb', 'ref_gb_in','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb','ref_gb_out'], \
              1:['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
    }
    df_order=[]
    offset=0
    for idx,size in enumerate(layer_list):
        for i in range(offset,offset+size):
            try:
                ele=df_dict[idx][int(actions[i])]
            except:
                print('DATAFLOW INTEPRETATION ERROR!')
                print(actions)
                print(int(actions[i]))
                exit()            
            df_order.append(ele)
            df_dict[idx].remove(ele)  
        offset+=size
    return copy.deepcopy(input_rf)+df_order
#[[2, 2, 5, 4, 0, 3, 0, 2, 0, 0, 1, 4, 3, 1, 2, 0, 0], [5, 5, 3, 4, 0, 2, 3, 2, 0, 0, 0, 5, 2, 2, 1, 1, 0], [9, 7, 5, 5, 0, 2, 0, 0, 0, 0, 6, 3, 1, 0, 2, 0, 0], [3, 8, 0, 3, 0, 4, 3, 1, 0, 0, 6, 2, 2, 2, 0, 0, 0], [7, 6, 0, 2, 0, 3, 3, 2, 0, 0, 0, 1, 0, 0, 1, 1, 0]]
#print(sample_results_df([2, 2, 5, 4, 0, 3, 0, 2, 0, 0, 1, 4, 3, 1, 2, 0, 0], \
#['row_out_rf', 'col_kernel_rf', 'ch_out_rf', 'batch_rf', 'col_out_rf', 'ref_rf_we', 'ref_rf_in', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_kernel_noc', 'ch_in_noc', 'row_out_noc', 'ch_out_noc']))
#print(sample_results_df([6, 7, 6, 4, 1, 1, 0, 1, 0, 0, 6, 5, 4, 3, 2, 1, 0],['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf', 'ref_rf_we', 'col_kernel_noc','row_kernel_noc','ch_in_noc','ch_out_noc']))

def arch_sample_results_df(dnn_layer_num,input_actions,input_rf,layer_list=[10,7]):
    #translate layers of input loop-order to string format
    input_actions=copy.deepcopy(input_actions)
    arch_df=[]
    for i in range(dnn_layer_num):
        arch_df.append(sample_results_df(input_actions[i],input_rf,layer_list=layer_list))
    return arch_df
        

#get the final results loop-order post_rf
#print(arch_sample_results_df(5,\
#                              [[8, 5, 3, 2, 5, 2, 3, 1, 0, 0, 5, 5, 1, 2, 0, 1, 0], [5, 3, 2, 4, 4, 4, 3, 1, 1, 0, 6, 3, 4, 1, 2, 0, 0], [3, 0, 0, 4, 1, 1, 1, 2, 0, 0, 5, 1, 0, 1, 0, 1, 0], [6, 7, 7, 5, 4, 4, 0, 0, 0, 0, 3, 1, 0, 3, 0, 0, 0], [8, 0, 0, 0, 0, 1, 0, 1, 1, 0, 5, 2, 2, 1, 0, 0, 0]],\
#                            ['row_out_rf', 'col_kernel_rf', 'ref_rf_out', 'row_kernel_rf', 'col_out_rf', 'batch_rf', 'ref_rf_we', 'ch_out_rf', 'ch_in_rf', 'ref_rf_in', 'row_kernel_noc', 'ch_in_noc', 'row_out_noc', 'ch_out_noc']))

def arch_lo_random_pop(dnn_layer_num,layer_list=[10,7]):
    #just leave it 
    arch_lo_pop=[]
    for _ in range(dnn_layer_num):
        arch_lo_pop.append(lo_random_pop(layer_list=layer_list))
    return arch_lo_pop
    

#######################
#layer level util func
#######################
#find the factors of a number
def r_factors(x):
    #find the factors of a number
    factor_list=[]
    for i in range(1, x + 1):
        if x % i == 0:
            factor_list.append(i)
    return factor_list
def diff_cal(factors):
    diff_sum=0
    for i in range(1,len(factors)):
        diff_sum+=abs(factors[i]-factors[i-1])
    return diff_sum
        
def factor_n(x,n=3,flexible_factor=1):
    #return the factor combo of length n for number x
    #flexible number:
    #               return factor combo of length n for number [x,flexible_factor)
    #               with requirement that the factors in in factor combo can not differ too much which is bad for resource partition


    #force one if n==1
    if n==1:
        flexible_factor=1
    #initialize max diff among factors and if this is original input or not
    diff_sum_min=math.inf
    input=True
    result=[]
    for _ in range(flexible_factor):
        #return factors of x
        factor_list=r_factors(x)
        num=factor_list[-1]
        tmp_list=[]
        for i in factor_list:
            for _ in range(n):
                tmp_list.append(i)
        # Get all combinations of factor_list
        # and length n
        comb = combinations(tmp_list, n) 
        for i in list(comb):
            mult=1
            for f in i:
                mult*=f
            if mult==num and (i not in result):               
                if input:
                    result.append(i)
                else:
                    if diff_cal(i)<diff_sum_min:
                        result.append(i)
                        diff_sum_min=diff_cal(i)
        if input:
            for i in result:
                tmp_diff_sum=diff_cal(i)
                if tmp_diff_sum<diff_sum_min:
                    diff_sum_min=tmp_diff_sum
        x+=1
        input=False
    return result

def permute_factor(input_factor_list):
    #permute the order within each factor in the factor_list
    #input  isolation
    factor_list=copy.deepcopy(input_factor_list)
    result=[]
    for f in factor_list:
        perm = permutations(f)     
        # Print the obtained permutations                        
        for i in list(perm): 
            if i not in result:                             
                result.append(i)
    return result


def random_pop_dict(config_dict,df_order,factor_list_dict):                      #ideally pop should not be a complete dict
    #randomly sample tiling factor combo for a specific loop-order

    #config_dict: with each key representing the data dimension: ch_in, ch_out,.... available
    #             with values representing the entire size of this dimension and how many levels we will tiling them into
    #df_order: loop-order 
    #factor_list_dict: possible tiling factor combos for each key in config_dict,
    #                  with value in list format representing factor combo of lengh = values in config_dict


    
    #########################################                                          #it adds up too much computation                                                     
    #the reason we did not fuse factor list into random pop
    #is that, we dont want to do all the permutation and ganky
    #stuff every time we call the random pop
    ##########################################
    
    df_dict={}
    #inefficient nested for loop....
    for key in config_dict.keys():
        #random sample a value combo
        try:
            pos=randint(0,len(factor_list_dict[key])-1)
        except:
            print('random pop dict erro exit')
            print(factor_list_dict)
            print(key)
            print(factor_list_dict[key])
            exit()
        value=factor_list_dict[key][pos]
        ctr=0
        for sub_key in df_order:
            if key in sub_key:
                df_dict[sub_key]=value[ctr]
                ctr+=1                    
            
    #return should be a complete df_dict
    return df_dict

    
def give_birth(str1,str2,config_dict):
    #combine or inherit the features of two tiling factos combo

    str3={}
    keys=list(config_dict.keys())
    #random number to decide which part of the parents to inherit
    num=randint(0,2)
    if num==0:
        #inherit from parent one
        for key in keys[0:int(len(keys)/2)]:
            for sub_key in str1.keys():
                if key in sub_key:
                    str3[sub_key]=str1[sub_key]                    
        #inherit from parent two
        for key in keys[int(len(keys)/2):]:
            for sub_key in str2.keys():
                if key in sub_key:
                    str3[sub_key]=str2[sub_key]
    elif num==1:
        str3=str1
    else:
        str3=str2
    
    #handling the special cases of kernel
    for key in str1.keys():
        if 'kernel' in key:
            str3[key]=11
            
    return str3


def mutate(str1,prop,config_dict,df_order,factor_list_dict):
    #with prop possiblity to randomly permute some features of a given tiling factor
    if random()<prop:
        keys=list(config_dict.keys())
        #currently only mutate one position
        pos=randint(0,len(keys)-1)
        key=keys[pos]
        sub_pos=randint(0,len(factor_list_dict[key])-1)
        ctr=0
        for sub_key in df_order:
            if key in sub_key:
                str1[sub_key]=factor_list_dict[key][sub_pos][ctr]
                ctr+=1
    return str1



#######################
#arch level util func
#######################
 

def arch_factor_list_dict(input_arch_config_dict):
    #generate all possible tiling factor combo for all layers 

    #input_arch_config_dict:
    #                       with each outer list representing each layer
    #                       with each dict within is a config_dict defined in random_pop_dict

    #input isolation
    arch_config_dict=copy.deepcopy(input_arch_config_dict)
    arch_factor_list=[]
    for config_dict in arch_config_dict:
        config_dict=config_dict[1]
        factor_list_dict={}
        for key in config_dict.keys():
            tmp_input=config_dict[key]
            #currently flexible_factor has a range of 5% of the original input
            #tmp_input.append(1)
            tmp_input.append(math.ceil((tmp_input[0]*0.05)))
            factor_list_dict[key]=permute_factor(factor_n(*tmp_input))
        arch_factor_list.append(factor_list_dict)
    return arch_factor_list

def arch_random_pop(input_arch_config_dict,input_df_order,input_arch_factor_list):
    #randomly sample tiling factors for layers of CONV
    #while random_pop_dict defined above is for single layer

    #input isolation
    arch_config_dict=copy.deepcopy(input_arch_config_dict)
    df_order=copy.deepcopy(input_df_order)
    arch_factor_list=copy.deepcopy(input_arch_factor_list)
    layer_wise=(type(df_order[0])==list)
    pop=[]
    layer=0
    for (factor_list,config_dict) in zip(arch_factor_list,arch_config_dict):
        #kernel_size=config_dict[2]
        config_dict=config_dict[1]
        if not layer_wise:
            pop.append(random_pop_dict(config_dict,df_order,factor_list))
        else:
            pop.append(random_pop_dict(config_dict,df_order[layer],factor_list))
        layer+=1
    return pop
def arch_give_birth(input_str1,input_str2,input_arch_config_dict):
    #input_isolation
    str1=copy.deepcopy(input_str1)
    str2=copy.deepcopy(input_str2)
    #arch wise inheritance
    str3=[]
    #random number to decide which part of the parents to inherit
    num=randint(0,2)
    if num==0:
        str3=str1[0:int(len(str1)/2)]+str2[int(len(str2)/2):]
    elif num==1:
        str3=str1
    else:
        str3=str2
    #layer wise inheritance                                                           #layer wise
    #......
            
    return str3
    
def arch_mutate(str1,prop, arch_factor_list,mutate_pos_num=4):
    #randomly permute input tiling factor combo features in all-layer scope
    #prop: probability of mutation
    #arch_factor_list: tiling factors choices pool
    #mutate_pos_num: number of position under potential mutation

    #max number of positions within one layer
    max_num=len(list(arch_factor_list[0].keys()))
    offset=0
    pos=[]
    for _ in range(len(arch_factor_list)):
        #randomly sample n positions 
        tmp_pos=list(range(offset,max_num+offset))
        shuffle(tmp_pos,random=random)
        pos+=list(tmp_pos[0:mutate_pos_num])
        offset+=max_num
    #pos=np.random.randint(max_num,size=mutate_pos_num)
    for indx in pos: 
        #decide which layer the pos belong
        layer=indx//len(list(arch_factor_list[0].keys()))
        #decide which part of the layer the pos belongs to
        in_layer=indx%len(list(arch_factor_list[0].keys()))         
        #fetch the part name ch_in, ch_out,col_out,row_out......
        key=list(arch_factor_list[layer].keys())[in_layer]
        #randomly sample a element from all the value combo in form of factors
        sub_pos=randint(0,len(arch_factor_list[layer][key])-1)
        element=arch_factor_list[layer][key][sub_pos]
        #assign the element back to str1
        ctr=0
        for i in str1[layer].keys():
            #i is the full name of the part: ch_in_rf,ch_out_noc,ch_in_noc.....
            if key in i:
                str1[layer][i]=element[ctr]
                ctr+=1   
    return str1

    
#####################    
#threading util
####################

def multi_p(func,args,output_q,num_worker_threads,dump_yard):
    #routine to distribute workers to multi cores
    #BETTER leave it

    #length of args has to be the multiple of num_worker_threads
    args=list(args)
    run_ites=int((len(args))//num_worker_threads)
    for run_ite in range(run_ites):
        processes = [multiprocessing.Process(target=func, args=([args[i]])) for i in range(run_ite*num_worker_threads,(run_ite+1)*num_worker_threads)]
        #print(len(processes))
        #print('queue size: ',score_pair.qsize())
        for p in processes:
            p.start()
        while not output_q.empty():
            pair=output_q.get()
            dump_yard.append(pair)
        for p in processes:
            p.join()
    while not output_q.empty():
        pair=output_q.get()
        dump_yard.append(pair)
    return None



###################################
#combined level
###################################
def gen_net_arch(df_order,dnn):
    #convert a dnn dimension input (from user) to a arch_config_dict format, according to given df_order
    #basically specfying how many levels of memory we want to tile the data dimnension to
    net_arch=copy.deepcopy(dnn)
    for i in df_order:
        if 'ch_out' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['ch_out'][1]+=1
        elif 'ch_in' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['ch_in'][1]+=1
        elif 'batch' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['batch'][1]+=1
        elif 'col_out' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['col_out'][1]+=1
        elif 'row_out' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['row_out'][1]+=1
        elif 'row_kernel' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['row_kernel'][1]+=1 
        elif 'col_kernel' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['col_kernel'][1]+=1                
    return net_arch



#following func assume 3.7's order preservation in dict

#prevent overflow induced hanging in multi processing 
#so we need to streamline the communication between each process
def compress_dict(input_df_config_dict):
    df_config_dict=copy.deepcopy(input_df_config_dict)
    compressed=[]
    for layer in range(len(df_config_dict)):
        compressed.append([])
        for key in list(df_config_dict[layer].keys()):
            compressed[-1].append(df_config_dict[layer][key])
    return compressed
def decompress_dict(input_compressed,reference):
    compressed=copy.deepcopy(input_compressed)
    decompressed=copy.deepcopy(reference)
    for layer in range(len(compressed)):
        ctr=0
        for key in list(decompressed[layer].keys()):
            decompressed[layer][key]=compressed[layer][ctr]
            ctr+=1
    return decompressed
# dnn=[\
# (4, {'ch_out':(96,0),'ch_in':(3,0),'batch':(1,0),'col_out':(56,0),'row_out':(56,0)}, 11),\

# (1,{'ch_out':(256,0),'ch_in':(96,0),'batch':(1,0),'col_out':(27,0),'row_out':(27,0)}, 5),\

# (1,{'ch_out':(384,0),'ch_in':(384,0),'batch':(1,0),'col_out':(13,0),'row_out':(13,0)}, 3),\

# (1,{'ch_out':(256,0),'ch_in':(384,0),'batch':(1,0),'col_out':(13,0),'row_out':(13,0)}, 3),\

# (1,{'ch_out':(256,0),'ch_in':(384,0),'batch':(1,0),'col_out':(13,0),'row_out':(13,0)}, 3)\
# ]


