from random import random, randint,shuffle
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


def dram_variant_looporder(input_lp_order_dram,input_lp_order_sram):
    return None


def dram_invariant_looporder(input_lp_order):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................
    if not len(input_lp_order[1:])==len(set(input_lp_order[1:])):
        raise Exception('Please provide lp_order with no duplicate elements')
    input_rf=rf_noc_template[input_lp_order[0]]
    lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
    lp_order_template=['ch_out_gb','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb']    
    lp_order_string=[]
    input_actions=input_lp_order[1:9]
    for i in range(len(lp_order_template)):
        lp_order_string.append(lp_order_template[input_actions[i]])
    index_lst=list(range(len(lp_order_template_dram)))
    shuffle(index_lst)
    for i in index_lst:
        lp_order_string.append(lp_order_template_dram[i])
    return copy.deepcopy(input_rf)+copy.deepcopy(lp_order_string)



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
#[1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[128,0],'ch_in':[64,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[128,0],'ch_in':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
]


#fpga dedicated 706
tmp_hw_spec={\
    'gb_vol':16*1024*1024, \
    'rf_vol':512*8, \
    'num_pe':824, \
    'num_rf':824
}






############################
#user interface
############################


#generate the design space of all possible tiling factors
#the space is partitioned according to alloc_slots based on the rf_noc_template choice (PE array)
(tiling_pool,alloc_slots,space_partition)=fpga_tiling_generator(input_dnn,tmp_hw_spec['gb_vol'],tmp_hw_spec['num_pe'],return_partitioned_space=True)
print(tiling_pool[0])
print(len(tiling_pool))
print(alloc_slots)
print(space_partition)

#for _ in range(100):
#
#    input_lp_order=[]
#    #pick a pe array
#    pe_array=randint(0,3)
#    #include the pe array to the lp_order
#    input_lp_order.append(pe_array)
#    #complete the rest of the lp_order
#    tmp_list=list(range(10))
#    shuffle(tmp_list)
#    input_lp_order+=tmp_list
#    #translate the lp_order to string format
#    lp_order_string=dram_invariant_looporder(input_lp_order)
#    #choose the applicable tiling space
#    partitioned_choices=space_partition[pe_array]
#    #generate a tiling scheme according to chosen space 
#    #!!!!!!!!!!!!!!!! NOTE that if pe_array==3; len(partitioned_choices)=3; otherwise len(partitioned_choices)=4
#    tiling_scheme=[]    
#    for i in partitioned_choices:
#        tiling_scheme.append(randint(0,i-1))
#    #translate the tiling_scheme to string(dict) format
#    tiling_scheme_string=tiling_translation(tiling_scheme,tiling_pool,alloc_slots,pe_array)
#    #check if larger than DSP limit
#    if not dsp_check(tiling_scheme_string, tmp_hw_spec['num_pe']):
#        print('DSP limit exceeded')
#        continue    
#    else:
#        #pass for EDP feedback
#        print(life_eval(tiling_scheme_string,1,tmp_hw_spec,df_order=lp_order_string))
#


def exhuastive_shuffling(lst1):
    lst1_pool=[]
    if len(lst1)!=1:
        for permt in exhuastive_shuffling(lst1[1:]):
            for i in range(0,len(permt)+1):
                tmp=list(permt)
                tmp.insert(i,lst1[0])
                lst1_pool.append(tmp)
        
    else:
        return [lst1]
    return lst1_pool




lst_pool=exhuastive_shuffling(list(range(7)))
print(len(lst_pool))
score_list=[]
print_freq=100
ctr=0
#for pe_array in range(0,4):
#    for tiling in tiling_pool[alloc_slots[pe_array]:alloc_slots[pe_array+1]]:
#        for lp_order in lst_pool:
#            ctr+=1
#            input_lp_order=[]
#            input_lp_order.append(pe_array)
#            input_lp_order+=lp_order
#            lp_order_string=dram_invariant_looporder(input_lp_order)
#            if not dsp_check(tiling, tmp_hw_spec['num_pe']):
#                #print('DSP limit exceeded')
#                continue
#            else:
#                #pass for EDP feedback
#                score_list.append(life_eval(tiling[0],1,tmp_hw_spec,df_order=lp_order_string))
#                if ctr%100==0:
#                    print('current max: ', max(score_list))
#



for pe_array in range(0,4):
    for tiling in tiling_pool[alloc_slots[pe_array]:alloc_slots[pe_array+1]]:
        for lp_order_batch in range(0,len(lst_pool),10):
            ctr+=1
            score_q=Queue()
            def worker(i):
                try:
                    input_lp_order=[]
                    input_lp_order.append(pe_array)
                    input_lp_order+=lst_pool[i]
                    lp_order_string=dram_invariant_looporder(input_lp_order)
                    if not dsp_check(tiling, tmp_hw_spec['num_pe']):
                        #print('DSP limit exceeded')
                        pass
                    else:
                        #pass for EDP feedback
                        score_q.put(life_eval(tiling[0],1,tmp_hw_spec,df_order=lp_order_string))
                except NameError:
                    raise Exception("There is no room in the queue in rf template stage")
            if not score_q.empty():
                print('Some Trash in the score_q Queue')
                exit()
            
            work_load=list(range(lp_order_batch,lp_order_batch+10))
            processes = [multiprocessing.Process(target=worker, args=([load])) for load in work_load]
            tmp_dump_yard=[]
            
            for p in processes:
                p.start()
                time.sleep(0.02)
            time.sleep(2)
            while not score_q.empty():
                tmp_batch=score_q.get()
                tmp_dump_yard.append(tmp_batch)
            for p in processes:
                p.join()
            #too many dump_yard...
            while not score_q.empty():
                tmp_batch=score_q.get()
                tmp_dump_yard.append(tmp_batch)
            
            score_pair=sorted(tmp_dump_yard,reverse=True)
            #add to the best score
            score_list.append(score_pair[0])
            if ctr%2==0:
                print('current max: ', max(score_list))

