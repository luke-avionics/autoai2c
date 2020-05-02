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
score=[]
for _ in range(500):
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
    #
    #1. you need to specify the pe_array_dim_choices style, to favor different trade off sizes among each pe dimension;
    #           currently it is under 10, i.e. 0-9
    layer=0
    pe_array_dim_choices=randint(0,9)
    tiling_space_1=tiling1.tiling_space_partition(pe_array,layer,pe_array_dim_choices) 
    print(len(tiling_space_1))
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
    #                                     each element is a number from 0 to 1, specify 2 different choices
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
    if life_eval(tiling_string,1,tmp_hw_spec,df_order=lp_order_string)[1]:
        score.append(life_eval(tiling_string,1,tmp_hw_spec,df_order=lp_order_string)[0])
    print(sorted(score,reverse=True)[0])







