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

# def dram_invariant_looporder(pe_array, input_lp_order_dram, input_lp_order_gb,input_lp_order_rf):
    # # input_lp_order:[range(0,4),                                                                           ]
    # #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................
    
    # input_lp_order_dram=copy.deepcopy(input_lp_order_dram)
    # input_lp_order_gb=copy.deepcopy(input_lp_order_gb)
    # input_lp_order_rf=copy.deepcopy(input_lp_order_rf)
    # if not (len(input_lp_order_gb)==len(set(input_lp_order_gb)) and len(input_lp_order_dram)==len(set(input_lp_order_dram))):
        # raise Exception('Please provide lp_order with no duplicate elements')
    # input_rf=copy.deepcopy(rf_noc_template[pe_array])
    # lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
    # lp_order_template_gb=['ch_out_gb','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb']
    # lp_order_template_rf= ['col_out_rf', 'batch_rf', 'row_out_rf', 'ch_out_rf', 'row_kernel_rf',  'ch_in_rf',  'col_kernel_rf']

    # lp_order_string=[]
    # for i in range(len(lp_order_template_rf)):
        # lp_order_string.append(lp_order_template_rf[input_lp_order_rf[i]])
    # lp_order_string+=input_rf
    # for i in range(len(lp_order_template_gb)):
        # lp_order_string.append(lp_order_template_gb[input_lp_order_gb[i]])
    # for i in range(len(lp_order_template_dram)):
        # lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    # return copy.deepcopy(lp_order_string)
def dram_invariant_looporder(pe_array, input_lp_order_dram, input_lp_order_gb):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................
    if not (len(input_lp_order_gb)==len(set(input_lp_order_gb)) and len(input_lp_order_dram)==len(set(input_lp_order_dram))):
        raise Exception('Please provide lp_order with no duplicate elements')
    input_rf=rf_noc_template[pe_array]
    lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
    lp_order_template=['ch_out_gb','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb']
    lp_order_string=[]
    for i in range(len(lp_order_template)):
        lp_order_string.append(lp_order_template[input_lp_order_gb[i]])
    for i in range(len(lp_order_template_dram)):
        lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
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



# tiling1=asic_tiling_generator(input_dnn,hw_spec)
# print(tiling1.rs2_rf_gb_tiling_choices_num[5][5])
# print(tiling1.tiling_translation(5,1,5,[7,9,0,4,0,0,0],[0,1,2,3,4,1,1]))
# exit()

############################
#user interface
############################


#generate the design space of all possible tiling factors
#the space is partitioned according to alloc_slots based on the rf_noc_template choice (PE array)
tiling1=fpga_tiling_generator(input_dnn,tmp_hw_spec)

for _ in range(100):
    #pick a pe array
    pe_array=randint(0,3)
    #complete the rest of the lp_order: local buffer(rf), global buffer(gb), dram 
    #input_lp_order_rf=list(range(7))
    #shuffle(input_lp_order_rf)
    input_lp_order_gb=list(range(7))
    shuffle(input_lp_order_gb)
    input_lp_order_dram=list(range(7))
    shuffle(input_lp_order_dram)
    #translate the lp_order to string format
    lp_order_string=dram_invariant_looporder(pe_array,input_lp_order_dram, input_lp_order_gb)

    #choose the applicable tiling space
    #!!!ATTENTION HERE!!!
    #1. you need to decide which layer of the network you are optimizing on when fixing lp_order
    #2. you need to specify the pe_array_dim_choices style, to favor different trade off sizes among each pe dimension;
    #           currently it is under 10, i.e. 0-9
    layer=0
    pe_array_dim_choices=randint(0,9)
    tiling_space_1=tiling1.tiling_space_partition(pe_array,layer,pe_array_dim_choices) 
    #now you have a space format to choose the tiling from
    #tiling_space_1 is a list of size 7 EXCEPT when pe_array==0, that time the size is 5
    #each element specify how many choices you could have to each tiling.. (for actual implication of these choices talk to me)
    tiling_choices=[]
    for i in tiling_space_1:
        tiling_choices.append(randint(0,i-1))
    #The above tiling choices actually specify how data will be cut into chunks 
    #you need to specify in which order these chunks will be assigned to each memory respectively
    #the following tiling_choices_order will do so
    #tiling_choices_order is a list with the same size as tiling_choices
    #                                     each element is a number from 0 to 5, specify 6 different choices
    tiling_choices_order=[]
    for i in range(len(tiling_choices)):
        tiling_choices_order.append(randint(0,1))
    #next translate the tiling scheme to dict/string format for energy mode
    #Guess what... NO DSP LIMIT now !!! already enforced
    #but.....there is something else .....
    #no memory check, the 900000000.. value means that the memory consumption exceeded
    tiling_string=tiling1.tiling_translation(layer,pe_array,pe_array_dim_choices,tiling_choices,tiling_choices_order)
    #pass for EDP feedback
    #print(pe_array)
    print(life_eval(tiling_string,1,tmp_hw_spec,df_order=lp_order_string))







