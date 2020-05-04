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
#for saving np to matlab 
import scipy.io as sio





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
    lp_order_string+=input_rf
    for i in range(len(lp_order_template_gb)):
        lp_order_string.append(lp_order_template_gb[input_lp_order_gb[i]])
    for i in range(len(lp_order_template_dram)):
        lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    return copy.deepcopy(lp_order_string)



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
    'gb_vol':10*1024*8, \
    'rf_vol':512*8, \
    'num_pe':129, \
    'num_rf':129
}



# tiling1=asic_tiling_generator(input_dnn,hw_spec)
# print(tiling1.rs2_rf_gb_tiling_choices_num[5][5])
# print(tiling1.tiling_translation(5,1,5,[7,9,0,4,0,0,0],[0,1,2,3,4,1,1]))
# exit()

############################
#user interface
############################


def tiling_generator(input_dnn,tmp_hw_spec,bw=16):
    choices = {'ch_in': [], 'ch_out': [], 'col_kernel': [], 'row_kernel': [], 'col_out': [], 'row_out': [], 'batch': []}
    for layer in input_dnn:
        choices['ch_in']+=r_factors(layer[1]['ch_in'][0])
        choices['ch_out'] += r_factors(layer[1]['ch_out'][0])
        choices['col_kernel'] += r_factors(layer[1]['col_kernel'][0])
        choices['row_kernel'] += r_factors(layer[1]['row_kernel'][0])
        choices['col_out'] += r_factors(layer[1]['col_out'][0])
        choices['row_out'] += r_factors(layer[1]['row_out'][0])
        choices['batch'] += r_factors(layer[1]['batch'][0])
    for i in choices:
        choices[i]=set(choices[i])
    choices_len_rf=[]
    choices_len_gb=[]
    largest_rf=tmp_hw_spec["rf_vol"]/bw
    largest_gb=tmp_hw_spec["gb_vol"]/bw/100
    for i in choices:
        rf_bound=0
        gb_bound=0
        for syze in choices[i]:
            if largest_rf> syze:
                rf_bound+=1
            if largest_gb> syze:
                gb_bound+=1
        choices_len_rf.append(rf_bound)
        choices_len_gb.append(gb_bound)


    return choices, choices_len_rf, choices_len_gb
print(tiling_generator(input_dnn,tmp_hw_spec))
exit()




def pe_array_dimention_optimizer(input_dnn,tmp_hw_spec, range_c=0.9, consideration_range=10):
    #the number of tiles needed as penalty
    num_pe=tmp_hw_spec['num_pe']
    dim_3=[]
    dim_4=[]
    for i in range(int(range_c*num_pe),num_pe):
        dim_3+=factor_n(i,3)
        dim_4+=factor_n(i,4)
    dim_4=permute_factor(dim_4)
    dim_3=permute_factor(dim_3)
    pe_array_pool={}
    pe_array_pool[0]=[]
    pe_array_pool[1]=[]
    pe_array_pool[2]=[]
    pe_array_pool[3] = []

    score_board=[]
    #pe array 1
    for i in dim_4:
        score=0
        for layer in input_dnn:
            score+=(math.ceil(layer[1]['col_kernel'][0]/i[0])+ math.ceil(layer[1]['row_kernel'][0]/i[1])\
                    +math.ceil(layer[1]['ch_in'][0]/i[2]) + math.ceil(layer[1]['ch_out'][0]/i[3]))
        score_board.append(score)
    pe_array_pool[0]+=[i[1] for i in sorted(zip(score_board,dim_4))][0:consideration_range]



    score_board=[]
    #pe array 2
    for i in dim_4:
        score=0
        for layer in input_dnn:
            score+=(math.ceil(layer[1]['col_kernel'][0]/i[0])+ math.ceil(layer[1]['col_out'][0]/i[1])\
                    +math.ceil(layer[1]['ch_in'][0]/i[2]) +    math.ceil(layer[1]['ch_out'][0]/i[3]))
        score_board.append(score)
    pe_array_pool[1]+=[i[1] for i in sorted(zip(score_board,dim_4))][0:consideration_range]


    score_board=[]
    #pe array 2
    for i in dim_4:
        score=0
        for layer in input_dnn:
            score+=(math.ceil(layer[1]['row_kernel'][0]/i[0])+ math.ceil(layer[1]['col_out'][0]/i[1])\
                    +math.ceil(layer[1]['ch_in'][0]/i[2]) +    math.ceil(layer[1]['ch_out'][0]/i[3]))
        score_board.append(score)
    pe_array_pool[2]+=[i[1] for i in sorted(zip(score_board,dim_4))][0:consideration_range]

    score_board=[]
    #pe array 3
    for i in dim_3:
        score=0
        for layer in input_dnn:
            score+=(math.ceil(layer[1]['row_out'][0]/i[0])+ math.ceil(layer[1]['col_out'][0]/i[1])\
                    +math.ceil(layer[1]['ch_out'][0]/i[2]))
        score_board.append(score)
    pe_array_pool[3]+=[i[1] for i in sorted(zip(score_board,dim_3))][0:consideration_range]



    return pe_array_pool



def tiling_translation( tiling_rf, tiling_gb,tiling_pe, pe_array,pe_array_pool, tiling_choices_dict, input_dnn):
    tiling_str = []
    pe_array_dim_choices_dict={}
    pe_array_dim_choices_dict[0]= copy.deepcopy(noc_template[0])
    pe_array_dim_choices_dict[1]= copy.deepcopy(noc_template[1])
    pe_array_dim_choices_dict[2]= copy.deepcopy(noc_template[2])
    pe_array_dim_choices_dict[3]= copy.deepcopy(noc_template[3])
    for layer in input_dnn:
        tiling_str.append({})
        tiling_dim=0
        for i in tiling_choices_dict:
            tiling_str[-1][i+"_rf"] =min(tiling_choices_dict[i][tiling_rf[tiling_dim]],layer[1][i][0])
            if i+"_noc" in list(pe_array_dim_choices_dict[pe_array]):
                pe_idx= list(pe_array_dim_choices_dict[pe_array]).index(i+"_noc")
                tiling_str[-1][i + "_noc"]=min(pe_array_pool[pe_array][tiling_pe][pe_idx], math.ceil(layer[1][i][0]/tiling_str[-1][i+"_rf"]))
                tiling_str[-1][i+"_gb"] =min(tiling_choices_dict[i][tiling_gb[tiling_dim]], math.ceil(layer[1][i][0]/tiling_str[-1][i+"_rf"]/tiling_str[-1][i+"_noc"]))
                tiling_str[-1][i + "_dram"]= math.ceil(layer[1][i][0]/tiling_str[-1][i+"_rf"]/tiling_str[-1][i+"_noc"]/tiling_str[-1][i+"_gb"])
            else:
                tiling_str[-1][i + "_gb"] = min(tiling_choices_dict[i][tiling_gb[tiling_dim]],math.ceil(layer[1][i][0] / tiling_str[-1][i + "_rf"]))
                tiling_str[-1][i + "_dram"] =1
            tiling_dim+=1

    tiling_dim = 0
    consumption={}
    for i in tiling_choices_dict:
        consumption[i + "_rf"] = tiling_choices_dict[i][tiling_rf[tiling_dim]]
        if i+"_noc" in list(pe_array_dim_choices_dict[pe_array]):
            pe_idx= list(pe_array_dim_choices_dict[pe_array]).index(i+"_noc")
            consumption[i + "_noc"] = pe_array_pool[pe_array][tiling_pe][pe_idx]
        consumption[i + "_gb"] = tiling_choices_dict[i][tiling_gb[tiling_dim]]
        consumption[i + "_dram"] = 1
        tiling_dim += 1
    return tiling_str,consumption

def get_score_whole_dnn(tiling_string,consumption,tmp_hw_spec,lp_order_string,input_dnn):
    #check for resource consumption
    [penalty, buffer_not_exceed]=life_eval(consumption, 1, tmp_hw_spec, df_order=lp_order_string)
    if not buffer_not_exceed:
        return [penalty[0], buffer_not_exceed]
    edp_raw=[0,0]
    for layer in range(len(input_dnn)):
        [penalty, buffer_not_exceed] = life_eval(tiling_string[layer], input_dnn[layer][0], tmp_hw_spec, df_order=lp_order_string)
        if not buffer_not_exceed:
            print('a oh...')
            return [penalty[0], buffer_not_exceed]
        else:
            edp_raw[0]+=penalty[0]
            edp_raw[1]+=penalty[1]
    return  edp_raw[0]* edp_raw[1], True

    # print(life_eval(tiling_string,input_dnn[0][0],tmp_hw_spec,df_order=lp_order_string))

def resource_allocator_depth_std(input_dnn,tmp_hw_spec):
    tmp_hw_spec1 = { \
        'gb_vol': 10 * 1024 * 8, \
        'rf_vol': 512 * 8, \
        'num_pe': 129, \
        'num_rf': 129
    }
    tmp_hw_spec2 = { \
        'gb_vol': 10 * 1024 * 8, \
        'rf_vol': 512 * 8, \
        'num_pe': 129, \
        'num_rf': 129
    }

    return tmp_hw_spec1, tmp_hw_spec2

# [tiling_choices_dict,tiling_space_rf,tiling_space_gb,pe_array_dim_choices_dict,pe_array_dim_space_all]=tiling_generator(input_dnn,tmp_hw_spec)
#
# print(tiling_choices_dict)
# print(tiling_space_rf)
# print(tiling_space_gb)
# print(pe_array_dim_choices_dict)
# print(pe_array_dim_space_all)
# exit()
# print(pe_array_dim_choices_dict[0])
# [tiling_str,consumption]=tiling_translation([4,2,1,1,2,2,0],[4,2,1,1,2,2,0],[1,1,2,2],0,tiling_choices_dict,pe_array_dim_choices_dict,input_dnn)
# print(consumption)
# exit()

#generate the design space of all possible tiling factors
#the space is partitioned according to alloc_slots based on the rf_noc_template choice (PE array)

[tmp_hw_spec,tmp_hw_spec2]=resource_allocator_depth_std(input_dnn,tmp_hw_spec)
[tiling_choices_dict,tiling_space_rf,tiling_space_gb,pe_array_dim_choices_dict,pe_array_dim_space_all]=tiling_generator(input_dnn,tmp_hw_spec)
[tiling_choices_dict_dw,tiling_space_rf_dw,tiling_space_gb_dw,pe_array_dim_choices_dict_dw,pe_array_dim_space_all_dw]=tiling_generator_dw(input_dnn,tmp_hw_spec2)
pe_array_pool=pe_array_dimention_optimizer(input_dnn,tmp_hw_spec)
pe_arra_pool_dw=pe_array_dimention_optimizer_dw(input_dnn,tmp_hw_spec2)
#exit()
for _ in range(100):
    #pick a pe array
    pe_array=randint(0,3)
    #complete the rest of the lp_order: local buffer(rf), global buffer(gb), dram

    # one set for standard conv/group conv
    input_lp_order_rf=list(range(7))
    shuffle(input_lp_order_rf)
    input_lp_order_gb=list(range(7))
    shuffle(input_lp_order_gb)
    input_lp_order_dram=list(range(7))
    shuffle(input_lp_order_dram)

    # another set for depthwise conv


    #translate the lp_order to string format
    lp_order_string=dram_invariant_looporder(pe_array,input_lp_order_dram, input_lp_order_gb,input_lp_order_rf)


    #pe array
    # pe_array_dim_space_1=pe_array_dim_space_all[pe_array]
    # pe_array_dim_choices=[]
    # for i in pe_array_dim_space_1:
    #     pe_array_dim_choices.append(randint(0,i-1))

    #one set for standard conv/group conv
    pe_array_dim_choices=randint(0,9)
    #register file
    tiling_choices=[]
    for i in tiling_space_rf:
        tiling_choices.append(randint(0,i-1))
    #global buffer
    tiling_choices1=[]
    for i in tiling_space_gb:
        tiling_choices1.append(randint(0,i-1))
    print(tiling_choices1)

    #another set for depthwise conv

    pe_array_dim_choices_dw=randint(0,9)
    #register file
    tiling_choices_dw=[]
    for i in tiling_space_rf_dw:
        tiling_choices_dw.append(randint(0,i-1))
    #global buffer
    tiling_choices1_dw=[]
    for i in tiling_space_gb_dw:
        tiling_choices1_dw.append(randint(0,i-1))


    #next translate the tiling scheme to dict/string format for energy mode
    #Guess what... NO DSP LIMIT now !!! already enforced
    #but.....there is something else .....
    #no memory check, the 900000000.. value means that the memory consumption exceeded
    [tiling_string,consumption]=tiling_translation(tiling_choices,tiling_choices1,pe_array_dim_choices,pe_array,pe_array_pool,tiling_choices_dict,input_dnn)
    #pass for EDP feedback
    #print(pe_array)
    # print(tiling_string)
    # print(lp_order_string)
    penalty=get_score_whole_dnn(tiling_string, consumption, tmp_hw_spec, lp_order_string, input_dnn)
    print(penalty)

    # get_score_whole_dnn(tiling_string,input_dnn[0][0],tmp_hw_spec,df_order=lp_order_string)
    # print(life_eval(tiling_string,input_dnn[0][0],tmp_hw_spec,df_order=lp_order_string))







