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


#def dram_invariant_looporder(input_lp_order):
#    # input_lp_order:[range(0,4),                                                                           ]
#    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................
#    if not len(input_lp_order[1:])==len(set(input_lp_order[1:])):
#        raise Exception('Please provide lp_order with no duplicate elements')
#    input_rf=rf_noc_template[input_lp_order[0]]
#    lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
#    lp_order_template=['ref_gb_we','ch_out_gb', 'ref_gb_in','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb','ref_gb_out']
#    lp_order_string=[]
#    input_actions=input_lp_order[1:12]
#    for i in range(len(lp_order_template)):
#        lp_order_string.append(lp_order_template[input_actions[i]])
#    index_lst=list(range(len(lp_order_template_dram)))
#    shuffle(index_lst)
#    for i in index_lst:
#        lp_order_string.append(lp_order_template_dram[i])
#    return copy.deepcopy(input_rf)+copy.deepcopy(lp_order_string)

def dram_invariant_looporder(pe_array, input_lp_order_dram, input_lp_order_gb,input_lp_order_rf):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................
    
    input_lp_order_dram=copy.deepcopy(input_lp_order_dram)
    input_lp_order_gb=copy.deepcopy(input_lp_order_gb)
    input_lp_order_rf=copy.deepcopy(input_lp_order_rf)
    if not (len(input_lp_order_gb)==len(set(input_lp_order_gb)) and len(input_lp_order_dram)==len(set(input_lp_order_dram))):
        raise Exception('Please provide lp_order with no duplicate elements')
    input_rf=copy.deepcopy(rf_noc_template[pe_array])
    lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
    lp_order_template_gb=['ch_out_gb','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb']
    lp_order_template_rf= ['col_out_rf', 'batch_rf', 'row_out_rf', 'ch_out_rf', 'row_kernel_rf',  'ch_in_rf',  'col_kernel_rf']

    lp_order_string=[]
    for i in range(len(lp_order_template_rf)):
        lp_order_string.append(lp_order_template_rf[input_lp_order_rf[i]])
    lp_order_string.append(input_rf)
    for i in range(len(lp_order_template_gb)):
        lp_order_string.append(lp_order_template_gb[input_lp_order_gb[i]])
    for i in range(len(lp_order_template_dram)):
        lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    return copy.deepcopy(lp_order_string)


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


#fpga dedicated 706
tmp_hw_spec={\
    'gb_vol':2*1024*1024, \
    'rf_vol':512, \
    'num_pe':156, \
    'num_rf':156
}



tiling1=asic_tiling_generator(input_dnn,hw_spec)
print(tiling1.os_rf_gb_tiling_choices_num[5][5])
print(tiling1.tiling_translation(5,3,5,[9,2,4,0,0,0,0],[0,1,2,3,4,1,1]))
exit()

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

for _ in range(100):
    #pick a pe array
    pe_array=randint(0,3)
    #complete the rest of the lp_order
    input_lp_order_rf=list(range(7))
    shuffle(input_lp_order_rf)
    input_lp_order_gb=list(range(7))
    shuffle(input_lp_order_gb)
    input_lp_order_dram=list(range(7))
    shuffle(input_lp_order_dram)
    #translate the lp_order to string format
    lp_order_string=dram_invariant_looporder(pe_array,input_lp_order_dram, input_lp_order_gb,input_lp_order_rf)
    #choose the applicable tiling space
    partitioned_choices=space_partition[pe_array]
    #generate a tiling scheme according to chosen space 
    #!!!!!!!!!!!!!!!! NOTE that if pe_array==3; len(partitioned_choices)=3; otherwise len(partitioned_choices)=4
    tiling_scheme=[]    
    for i in partitioned_choices:
        tiling_scheme.append(randint(0,i-1))
    #translate the tiling_scheme to string(dict) format
    tiling_scheme_string=tiling_translation(tiling_scheme,tiling_pool,alloc_slots,pe_array,space_partition)
    print(tiling_scheme_string[0])
    exit()
    #check if larger than DSP limit
    if not dsp_check(tiling_scheme_string, tmp_hw_spec['num_pe']):
        print('DSP limit exceeded')
        continue    
    else:
        #pass for EDP feedback
        print(life_eval(tiling_scheme_string,1,tmp_hw_spec,df_order=lp_order_string))






