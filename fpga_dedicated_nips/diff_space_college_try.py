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
from sympy.solvers import solve
from sympy import Symbol


def Quartic(args):
    a, b, c, d, e = args
    if a == 0:
        return Cubic(args[1:])
    if e == 0:
        res = Cubic(args[0:-1])
        res.append((0, 0))
        return res
    if a != 1:
        b /= a
        c /= a
        d /= a
        e /= a

    cb = -c
    cc = -4.0 * e + d * b
    cd = -(b * b * e + d * d) + 4.0 * c * e
    if cd == 0:
        print('cd==0')
    q = (3.0 * cc - (cb * cb)) / 9.0
    r = -(27.0 * cd) + cb * (9.0 * cc - 2.0 * (cb * cb))
    r /= 54.0
    discrim = q * q * q + r * r
    term1 = (cb / 3.0)
    if discrim > 0:
        s = r + math.sqrt(discrim)
        s = -math.pow(-s, (1.0 / 3.0)) if s < 0 else math.pow(s, (1.0 / 3.0))
        t = r - math.sqrt(discrim)
        t = -math.pow(-t, (1.0 / 3.0)) if t < 0 else math.pow(t, (1.0 / 3.0))
        y1 = -term1 + s + t
    else:
        if discrim == 0:
            r13 = -math.pow(-r, (1.0 / 3.0)) if r > 0 else math.pow(r, (1.0 / 3.0))
            y1 = -term1 + 2.0 * r13
        else:
            q = -q
            dum1 = q * q * q
            dum1 = math.acos(r / math.sqrt(dum1))
            r13 = 2.0 * math.sqrt(q)
            y1 = -term1 + r13 * math.cos(dum1 / 3.0)

    term1 = b / 4.0
    sqR = -c + term1 * b + y1
    RRe = RIm = DRe = DIm = ERe = EIm = z1Re = z1Im = z2Re = 0
    if sqR >= 0:
        if sqR == 0:
            dum1 = -(4.0 * e) + y1 * y1
            if dum1 < 0:
                z1Im = 2.0 * math.sqrt(-dum1)
            else:
                z1Re = 2.0 * math.sqrt(dum1)
                z2Re = -z1Re
        else:
            RRe = math.sqrt(sqR)
            z1Re = -(8.0 * d + b * b * b) / 4.0 + b * c
            z1Re /= RRe
            z2Re = -z1Re
    else:
        RIm = math.sqrt(-sqR)
        z1Im = -(8.0 * d + b * b * b) / 4.0 + b * c
        z1Im /= RIm
        z1Im = -z1Im
    z1Re += -(2.0 * c + sqR) + 3.0 * b * term1
    z2Re += -(2.0 * c + sqR) + 3.0 * b * term1

    if z1Im == 0:
        if z1Re >= 0:
            DRe = math.sqrt(z1Re)
        else:
            DIm = math.sqrt(-z1Re)
        if z2Re >= 0:
            ERe = math.sqrt(z2Re)
        else:
            EIm = math.sqrt(-z2Re)
    else:
        r = math.sqrt(z1Re * z1Re + z1Im * z1Im)
        r = math.sqrt(r)
        dum1 = math.atan2(z1Im, z1Re)
        dum1 /= 2
        ERe = DRe = r * math.cos(dum1)
        DIm = r * math.sin(dum1)
        EIm = -DIm
    return [(-term1 + (RRe + DRe) / 2, (RIm + DIm) / 2),
            (-(term1 + DRe / 2) + RRe / 2, (-DIm + RIm) / 2),
            (-(term1 + RRe / 2) + ERe / 2, (-RIm + EIm) / 2),
            (-(term1 + (RRe + ERe) / 2), -(RIm + EIm) / 2)]

def dram_invariant_looporder(pe_array, input_lp_order_dram, input_lp_order_gb):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................
    
    input_lp_order_dram=copy.deepcopy(input_lp_order_dram)
    input_lp_order_gb=copy.deepcopy(input_lp_order_gb)
    if not (len(input_lp_order_gb)==len(set(input_lp_order_gb)) and len(input_lp_order_dram)==len(set(input_lp_order_dram))):
        raise Exception('Please provide lp_order with no duplicate elements')
    input_rf=copy.deepcopy(noc_template[pe_array])
    lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
    lp_order_template_gb=['ch_out_gb','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb']

    lp_order_string=[]
    lp_order_string+=input_rf
    for i in range(len(lp_order_template_gb)):
        lp_order_string.append(lp_order_template_gb[input_lp_order_gb[i]])
    for i in range(len(lp_order_template_dram)):
        lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    return copy.deepcopy(lp_order_string)


def dram_invariant_looporder_dw(pe_array, input_lp_order_dram, input_lp_order_gb):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................

    input_lp_order_dram = copy.deepcopy(input_lp_order_dram)
    input_lp_order_gb = copy.deepcopy(input_lp_order_gb)

    if not (len(input_lp_order_gb) == len(set(input_lp_order_gb)) and len(input_lp_order_dram) == len(
            set(input_lp_order_dram))):
        raise Exception('Please provide lp_order with no duplicate elements')
    input_rf = copy.deepcopy(noc_template_dw[pe_array])
    lp_order_template_dram = ['col_out_dram', 'ch_out_dram', 'batch_dram', 'row_out_dram',
                              'col_kernel_dram', 'row_kernel_dram']
    lp_order_template_gb = ['ch_out_gb', 'col_kernel_gb', 'row_out_gb', 'batch_gb', 'col_out_gb',
                            'row_kernel_gb']


    lp_order_string = []
    lp_order_string += input_rf
    for i in range(len(lp_order_template_gb)):
        lp_order_string.append(lp_order_template_gb[input_lp_order_gb[i]])
    for i in range(len(lp_order_template_dram)):
        lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    return copy.deepcopy(lp_order_string)


# input_dnn=[\
# [1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[128,0],'ch_in':[64,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[128,0],'ch_in':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# ]

input_dnn=[\
[1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
[1,{'ch_out':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
[1,{'ch_out':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
]

#fpga dedicated 706
tmp_hw_spec={\
    'gb_vol':20*1024*1024*8, \
    'rf_vol':8000, \
    'num_pe':900, \
    'num_rf':900
}




# tiling1=asic_tiling_generator(input_dnn,hw_spec)
# print(tiling1.rs2_rf_gb_tiling_choices_num[5][5])
# print(tiling1.tiling_translation(5,1,5,[7,9,0,4,0,0,0],[0,1,2,3,4,1,1]))
# exit()

############################
#user interface
############################


# def tiling_generator(input_dnn,tmp_hw_spec,bw=16):
#     choices = {'ch_in': [], 'ch_out': [], 'col_kernel': [], 'row_kernel': [], 'col_out': [], 'row_out': [], 'batch': []}
#     for layer in input_dnn:
#         choices['ch_in']+=r_factors(layer[1]['ch_in'][0])
#         choices['ch_out'] += r_factors(layer[1]['ch_out'][0])
#         choices['col_kernel'] += r_factors(layer[1]['col_kernel'][0])
#         choices['row_kernel'] += r_factors(layer[1]['row_kernel'][0])
#         choices['col_out'] += r_factors(layer[1]['col_out'][0])
#         choices['row_out'] += r_factors(layer[1]['row_out'][0])
#         choices['batch'] += r_factors(layer[1]['batch'][0])
#     for i in choices:
#         choices[i]=set(choices[i])
#     choices_len_rf=[]
#     choices_len_gb=[]
#     largest_rf=tmp_hw_spec["rf_vol"]/bw
#     largest_gb=tmp_hw_spec["gb_vol"]/bw/100
#     for i in choices:
#         rf_bound=0
#         gb_bound=0
#         for syze in choices[i]:
#             if largest_rf> syze:
#                 rf_bound+=1
#             if largest_gb> syze:
#                 gb_bound+=1
#         choices_len_rf.append(rf_bound)
#         choices_len_gb.append(gb_bound)
#
#
#     return choices, choices_len_rf, choices_len_gb
# [choices, choices_len_rf, choices_len_gb]=tiling_generator(input_dnn,tmp_hw_spec)
#
#
#
#
#
#
# def pe_array_dimention_optimizer(input_dnn,tmp_hw_spec, range_c=0.9, consideration_range=10):
#     #the number of tiles needed as penalty
#     num_pe=tmp_hw_spec['num_pe']
#     dim_3=[]
#     dim_4=[]
#     for i in range(int(range_c*num_pe),num_pe):
#         dim_3+=factor_n(i,3)
#         dim_4+=factor_n(i,4)
#     dim_4=permute_factor(dim_4)
#     dim_3=permute_factor(dim_3)
#     pe_array_pool={}
#     pe_array_pool[0]=[]
#     pe_array_pool[1]=[]
#     pe_array_pool[2]=[]
#     pe_array_pool[3] = []
#
#     score_board=[]
#     #pe array 1
#     for i in dim_4:
#         score=0
#         for layer in input_dnn:
#             score+=(math.ceil(layer[1]['col_kernel'][0]/i[0])+ math.ceil(layer[1]['row_kernel'][0]/i[1])\
#                     +math.ceil(layer[1]['ch_in'][0]/i[2]) + math.ceil(layer[1]['ch_out'][0]/i[3]))
#         score_board.append(score)
#     pe_array_pool[0]+=[i[1] for i in sorted(zip(score_board,dim_4))][0:consideration_range]
#
#
#
#     score_board=[]
#     #pe array 2
#     for i in dim_4:
#         score=0
#         for layer in input_dnn:
#             score+=(math.ceil(layer[1]['col_kernel'][0]/i[0])+ math.ceil(layer[1]['col_out'][0]/i[1])\
#                     +math.ceil(layer[1]['ch_in'][0]/i[2]) +    math.ceil(layer[1]['ch_out'][0]/i[3]))
#         score_board.append(score)
#     pe_array_pool[1]+=[i[1] for i in sorted(zip(score_board,dim_4))][0:consideration_range]
#
#
#     score_board=[]
#     #pe array 2
#     for i in dim_4:
#         score=0
#         for layer in input_dnn:
#             score+=(math.ceil(layer[1]['row_kernel'][0]/i[0])+ math.ceil(layer[1]['col_out'][0]/i[1])\
#                     +math.ceil(layer[1]['ch_in'][0]/i[2]) +    math.ceil(layer[1]['ch_out'][0]/i[3]))
#         score_board.append(score)
#     pe_array_pool[2]+=[i[1] for i in sorted(zip(score_board,dim_4))][0:consideration_range]
#
#     score_board=[]
#     #pe array 3
#     for i in dim_3:
#         score=0
#         for layer in input_dnn:
#             score+=(math.ceil(layer[1]['row_out'][0]/i[0])+ math.ceil(layer[1]['col_out'][0]/i[1])\
#                     +math.ceil(layer[1]['ch_out'][0]/i[2]))
#         score_board.append(score)
#     pe_array_pool[3]+=[i[1] for i in sorted(zip(score_board,dim_3))][0:consideration_range]
#
#
#
#     return pe_array_pool


def hardware_translation(ratio_noc,ratio_gb,pe_array,tmp_hw_spec,bw=16):

    #in_ch, out_ch, X, Y, X_K, Y_K
    rf_vol=tmp_hw_spec['rf_vol']/bw
    gb_vol=tmp_hw_spec['gb_vol']/bw
    pe_num=tmp_hw_spec['num_pe']
    #calculate rf
    consumption_dict={}
    consumption_dict["ch_in_rf"]=1
    consumption_dict["ch_out_rf"] = 1
    consumption_dict["col_out_rf"] = 1
    consumption_dict["row_out_rf"] =1
    consumption_dict["col_kernel_rf"] =1
    consumption_dict["row_kernel_rf"] = 1
    scaling_ratio=1
    #calculate pe
    if pe_array==3:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2])**(1/3)
        consumption_dict['row_out_noc']=max(math.floor(y*ratio_noc[0]),1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        scaling_ratio=(pe_num/(consumption_dict['row_out_noc']*consumption_dict['col_out_noc']*consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array==0:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2]/ratio_noc[3])**(1/4)
        consumption_dict['col_kernel_noc']=max(math.floor(y*ratio_noc[0]),1)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[1] * y), 1)
        consumption_dict['ch_in_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[3] ), 1)
        scaling_ratio = (pe_num / (consumption_dict['col_kernel_noc'] * consumption_dict['row_kernel_noc'] * consumption_dict['ch_in_noc']*consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 1:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2] / ratio_noc[3]) ** (1 / 4)
        consumption_dict['col_kernel_noc'] = max(math.floor(y * ratio_noc[0] ), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_in_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[3] ), 1)
        scaling_ratio = (pe_num / (consumption_dict['col_kernel_noc'] * consumption_dict['col_out_noc'] * consumption_dict['ch_in_noc']*consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 2:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2] / ratio_noc[3]) ** (1 / 4)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[0] ), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_in_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[3] ), 1)
        scaling_ratio = (pe_num /( consumption_dict['row_kernel_noc'] * consumption_dict['col_out_noc'] * consumption_dict['ch_in_noc']*consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    #calculate gb

    in_rf_consumption=consumption_dict["ch_in_rf"]*(consumption_dict["col_out_rf"]+consumption_dict['col_kernel_rf']-1)*(consumption_dict["row_out_rf"]+consumption_dict['row_kernel_rf']-1)
    out_rf_consumption=consumption_dict["ch_out_rf"] * consumption_dict["col_out_rf"] * consumption_dict["row_out_rf"]
    we_rf_consumption=consumption_dict["ch_in_rf"] * consumption_dict["ch_out_rf"] * consumption_dict["col_kernel_rf"]*consumption_dict["row_kernel_rf"]
    #print((in_rf_consumption +out_rf_consumption+we_rf_consumption)*16)
    in_rf_consumption_for_all_pes=in_rf_consumption
    out_rf_consumption_for_all_pes=out_rf_consumption
    we_rf_consumption_for_all_pes=we_rf_consumption
    for i in consumption_dict:
        if 'noc' in i:
            if 'ch_in' in i:
                in_rf_consumption_for_all_pes*=consumption_dict[i]
                we_rf_consumption_for_all_pes*=consumption_dict[i]
            elif 'ch_out' in i:
                out_rf_consumption_for_all_pes*=consumption_dict[i]
                we_rf_consumption_for_all_pes*=consumption_dict[i]
            elif ('col_out' in i) or ('row_out' in i):
                in_rf_consumption_for_all_pes*=consumption_dict[i]
                out_rf_consumption_for_all_pes*=consumption_dict[i]
            elif ('row_kernel' in i) or ('col_kernel' in i):
                we_rf_consumption_for_all_pes *= consumption_dict[i]
            else:
                pass
    a=(ratio_gb[0]*ratio_gb[2]*ratio_gb[3])*in_rf_consumption_for_all_pes+(ratio_gb[1] * ratio_gb[2] * ratio_gb[3])*out_rf_consumption_for_all_pes
    b=(ratio_gb[0] * ratio_gb[1] * ratio_gb[4]*ratio_gb[5])*we_rf_consumption_for_all_pes
    roots=np.roots([b,a,0,0,-gb_vol])
    roots=roots[np.isreal(roots)]
    z=0
    for i in roots:
        if float(i.real)>0:
            z=i.real
    consumption_dict["ch_in_gb"]=math.floor(max(ratio_gb[0]*z,1))
    consumption_dict["ch_out_gb"] = math.floor(max(ratio_gb[1] * z,1))
    consumption_dict["col_out_gb"] = math.floor(max(ratio_gb[2] * z,1))
    consumption_dict["row_out_gb"] = math.floor(max(ratio_gb[3] * z,1))
    consumption_dict["col_kernel_gb"] =math.floor( max(ratio_gb[4] * z,1))
    consumption_dict["row_kernel_gb"] =math.floor( max(ratio_gb[5] * z,1))
    scaling_ratio=(gb_vol/(consumption_dict["ch_in_gb"]*(consumption_dict["row_out_gb"]+consumption_dict['row_kernel_gb']-1)*(consumption_dict["col_out_gb"]+consumption_dict["col_kernel_gb"]-1)*in_rf_consumption_for_all_pes+consumption_dict["ch_out_gb"]*consumption_dict["row_out_gb"]*consumption_dict["col_out_gb"]*out_rf_consumption_for_all_pes+ \
                          consumption_dict["ch_in_gb"]*consumption_dict["ch_out_gb"] * consumption_dict["col_kernel_gb"] * consumption_dict["row_kernel_gb"] * we_rf_consumption_for_all_pes))
    while scaling_ratio<1:
        components_need_scaled = {}
        for i in consumption_dict:
            if ("gb" in i):
                if consumption_dict[i] != 1:
                    components_need_scaled[str(i)]=consumption_dict[i]
        if len(components_need_scaled)==sum(components_need_scaled.values()):
            print('can not fit the gb requirement')
            raise
        for i in components_need_scaled:
            consumption_dict[i] -=1
            consumption_dict[i] =max(consumption_dict[i],0)
        scaling_ratio = (gb_vol / (consumption_dict["ch_in_gb"] * (consumption_dict["row_out_gb"]+consumption_dict['row_kernel_gb']-1) * (consumption_dict["col_out_gb"]+consumption_dict["col_kernel_gb"]-1) * in_rf_consumption_for_all_pes + consumption_dict["ch_out_gb"] * consumption_dict["row_out_gb"] * consumption_dict["col_out_gb"] * out_rf_consumption_for_all_pes + \
                                   consumption_dict["ch_in_gb"] * consumption_dict["ch_out_gb"] * consumption_dict["col_kernel_gb"] * consumption_dict["row_kernel_gb"] * we_rf_consumption_for_all_pes))

    return consumption_dict

def hardware_translation_dw(ratio_noc,ratio_gb,pe_array,tmp_hw_spec,bw=16):
    #out_ch, X, Y, X_K, Y_K
    rf_vol=tmp_hw_spec['rf_vol']/bw
    gb_vol=tmp_hw_spec['gb_vol']/bw
    pe_num=tmp_hw_spec['num_pe']
    consumption_dict={}
    consumption_dict["ch_out_rf"] = 1
    consumption_dict["col_out_rf"] = 1
    consumption_dict["row_out_rf"] = 1
    consumption_dict["col_kernel_rf"] = 1
    consumption_dict["row_kernel_rf"] = 1
    #calculate pe
    if pe_array==3:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2])**(1/3)
        consumption_dict['row_out_noc']=max(math.floor(y*ratio_noc[0]),1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        scaling_ratio=(pe_num/(consumption_dict['row_out_noc']*consumption_dict['col_out_noc']*consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array==0:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2])**(1/3)
        consumption_dict['col_kernel_noc']=max(math.floor(y*ratio_noc[0]),1)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[1] * y), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        scaling_ratio =(pe_num / (consumption_dict['col_kernel_noc'] * consumption_dict['row_kernel_noc'] *consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 1:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2])**(1/3)
        consumption_dict['col_kernel_noc'] = max(math.floor(y * ratio_noc[0] ), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        scaling_ratio =(pe_num / (consumption_dict['col_kernel_noc'] * consumption_dict['col_out_noc'] *consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 2:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2])**(1/3)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[0] ), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        scaling_ratio = (pe_num / (consumption_dict['row_kernel_noc'] * consumption_dict['col_out_noc'] *consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    #calculate gb
    in_rf_consumption=consumption_dict["ch_out_rf"]*(consumption_dict["col_out_rf"]+consumption_dict['col_kernel_rf']-1)*(consumption_dict["row_out_rf"]+consumption_dict['row_kernel_rf']-1)
    out_rf_consumption=consumption_dict["ch_out_rf"] * consumption_dict["col_out_rf"] * consumption_dict["row_out_rf"]
    we_rf_consumption= consumption_dict["ch_out_rf"] * consumption_dict["col_kernel_rf"]*consumption_dict["row_kernel_rf"]
    #print((in_rf_consumption +out_rf_consumption+we_rf_consumption)*16)
    in_rf_consumption_for_all_pes=in_rf_consumption
    out_rf_consumption_for_all_pes=out_rf_consumption
    we_rf_consumption_for_all_pes=we_rf_consumption
    for i in consumption_dict:
        if 'noc' in i:
            if 'ch_out' in i:
                in_rf_consumption_for_all_pes *= consumption_dict[i]
                out_rf_consumption_for_all_pes*=consumption_dict[i]
                we_rf_consumption_for_all_pes*=consumption_dict[i]
            elif ('col_out' in i) or ('row_out' in i):
                in_rf_consumption_for_all_pes*=consumption_dict[i]
                out_rf_consumption_for_all_pes*=consumption_dict[i]
            elif ('row_kernel' in i) or ('col_kernel' in i):
                we_rf_consumption_for_all_pes *= consumption_dict[i]
            else:
                pass

    a=(ratio_gb[0]*ratio_gb[1]*ratio_gb[2])*in_rf_consumption_for_all_pes+(ratio_gb[0] * ratio_gb[1] * ratio_gb[2])*out_rf_consumption_for_all_pes
    b=(ratio_gb[0] * ratio_gb[3]*ratio_gb[4])*we_rf_consumption_for_all_pes
    
    # roots=np.roots([b,a,0,0,-gb_vol])
    # roots=roots[np.isreal(roots)]
    # z=0
    # for i in roots:
    #     if float(i.real)>0:
    #         z=i.real

    z=(gb_vol/(a+b))**(1/3)
    consumption_dict["ch_out_gb"] = math.floor(max(ratio_gb[0] * z,1))
    consumption_dict["col_out_gb"] = math.floor(max(ratio_gb[1] * z,1))
    consumption_dict["row_out_gb"] = math.floor(max(ratio_gb[2] * z,1))
    consumption_dict["col_kernel_gb"] =math.floor( max(ratio_gb[3] * z,1))
    consumption_dict["row_kernel_gb"] =math.floor( max(ratio_gb[4] * z,1))
    scaling_ratio=(gb_vol/(consumption_dict["ch_out_gb"]*(consumption_dict["row_out_gb"]+consumption_dict['row_kernel_gb']-1)*(consumption_dict["col_out_gb"]+consumption_dict['col_kernel_gb']-1)*in_rf_consumption_for_all_pes+consumption_dict["ch_out_gb"]*consumption_dict["row_out_gb"]*consumption_dict["col_out_gb"]*out_rf_consumption_for_all_pes+ \
                          consumption_dict["ch_out_gb"] * consumption_dict["col_kernel_gb"] * consumption_dict["row_kernel_gb"] * we_rf_consumption_for_all_pes))**(1/3)
    while scaling_ratio<1:
        components_need_scaled = {}
        for i in consumption_dict:
            if ("gb" in i):
                if consumption_dict[i] != 1:
                    components_need_scaled[str(i)]=consumption_dict[i]
        if len(components_need_scaled)==sum(components_need_scaled.values()):
            print('can not fit the gb requirement')
            raise
        for i in components_need_scaled:
            consumption_dict[i] -=1
            consumption_dict[i] =max(consumption_dict[i],0)
        scaling_ratio = (gb_vol / (consumption_dict["ch_out_gb"] * (consumption_dict["row_out_gb"] + consumption_dict['row_kernel_gb'] - 1) * (consumption_dict["col_out_gb"] + consumption_dict['col_kernel_gb'] - 1) * in_rf_consumption_for_all_pes + consumption_dict["ch_out_gb"] * consumption_dict["row_out_gb"] * consumption_dict["col_out_gb"] * out_rf_consumption_for_all_pes + \
                                   consumption_dict["ch_out_gb"] * consumption_dict["col_kernel_gb"] * consumption_dict["row_kernel_gb"] * we_rf_consumption_for_all_pes)) ** (1 / 3)

    return consumption_dict

def tiling_translation( consumption_dict, input_dnn):
    tiling_str = []
    for layer in input_dnn:
        tiling_str.append({})
        for i in consumption_dict:
            if "rf" in i:
                tiling_str[-1][i]=min(consumption_dict[i],layer[1][str(i)[:-3]][0])
        for i in consumption_dict:
            if "noc" in i:
                tiling_str[-1][i]=min(consumption_dict[i],math.ceil(layer[1][str(i)[:-4]][0]/tiling_str[-1][str(i)[:-4]+"_rf"]))
        for i in consumption_dict:
            if "gb" in i:
                try:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                        math.ceil(layer[1][str(i)[:-3]][0] / tiling_str[-1][str(i)[:-3] + "_noc"]/tiling_str[-1][str(i)[:-3] + "_rf"]))
                except KeyError:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                            math.ceil(layer[1][str(i)[:-3]][0] /
                                                      tiling_str[-1][str(i)[:-3] + "_rf"]))
                except:
                    raise
        tiling_str[-1]['batch_rf']=tiling_str[-1]['batch_gb']=1
        consumption_dict['batch_rf']=consumption_dict['batch_gb']=1
        dram_list=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
        for i in dram_list:
            consumption_dict[i]=1
            try:
                tiling_str[-1][i] =math.ceil(layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"]/ tiling_str[-1][str(i)[:-5] + "_noc"] / tiling_str[-1][
                                            str(i)[:-5] + "_rf"])
            except KeyError:
                tiling_str[-1][i] =math.ceil(layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"]/ tiling_str[-1][
                                            str(i)[:-5] + "_rf"])
            except:
                raise
    return tiling_str,consumption_dict

def tiling_translation_dw(consumption_dict, input_dnn):
    tiling_str = []
    for layer in input_dnn:
        tiling_str.append({})
        for i in consumption_dict:
            if "rf" in i:
                tiling_str[-1][i]=min(consumption_dict[i],layer[1][str(i)[:-3]][0])
        for i in consumption_dict:
            if "noc" in i:
                tiling_str[-1][i]=min(consumption_dict[i],math.ceil(layer[1][str(i)[:-4]][0]/tiling_str[-1][str(i)[:-4]+"_rf"]))
        for i in consumption_dict:
            if "gb" in i:
                try:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                        math.ceil(layer[1][str(i)[:-3]][0] / tiling_str[-1][str(i)[:-3] + "_noc"]/tiling_str[-1][str(i)[:-3] + "_rf"]))
                except KeyError:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                            math.ceil(layer[1][str(i)[:-3]][0] /
                                                      tiling_str[-1][str(i)[:-3] + "_rf"]))
                except:
                    raise
        tiling_str[-1]['batch_rf']=tiling_str[-1]['batch_gb']=1
        consumption_dict['batch_rf']=consumption_dict['batch_gb']=1
        dram_list=['col_out_dram', 'ch_out_dram', 'batch_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
        for i in dram_list:
            consumption_dict[i]=1
            try:
                tiling_str[-1][i] =math.ceil(layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"]/ tiling_str[-1][str(i)[:-5] + "_noc"] / tiling_str[-1][
                                            str(i)[:-5] + "_rf"])
            except KeyError:
                tiling_str[-1][i] =math.ceil(layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"]/ tiling_str[-1][
                                            str(i)[:-5] + "_rf"])
            except:
                raise
    return tiling_str,consumption_dict




def get_score_whole_dnn(tiling_string,consumption,tmp_hw_spec,lp_order_string,input_dnn):
    #check for resource consumption
    [penalty, buffer_not_exceed]=life_eval(consumption, 1, tmp_hw_spec, 0,group_num=1,df_order=lp_order_string)
    if not buffer_not_exceed:
        print('consumption is out of limit')
        return [(penalty[0],penalty[1]), buffer_not_exceed]
    edp_raw=[0,0]
    for layer in range(len(input_dnn)):
        [penalty, buffer_not_exceed] = life_eval(tiling_string[layer], input_dnn[layer][0], tmp_hw_spec,input_dnn[layer][2],group_num=input_dnn[layer][3],df_order=lp_order_string)
        if not buffer_not_exceed:
            print('a oh...')
            return [(penalty[0],penalty[1]), buffer_not_exceed]
        else:
            edp_raw[0]+=penalty[0]
            edp_raw[1]+=penalty[1]
    return  (edp_raw[0], edp_raw[1]), True

    # print(life_eval(tiling_string,input_dnn[0][0],tmp_hw_spec,df_order=lp_order_string))

def resource_allocator_depth_std(input_dnn,tmp_hw_spec):
    input_dnn=copy.deepcopy(input_dnn)
    input_dnn_std=[]
    input_dnn_dw=[]
    for layer in input_dnn:
        if layer[2]==1:
            input_dnn_dw.append(layer)
        else:
            input_dnn_std.append(layer)
    para_dw=0
    para_std=0
    comp_dw=0
    comp_std=0
    for layer in input_dnn:
        if layer[2]==0 or layer[2]==2:
            para_std+=  ((layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]+layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0]+layer[0]*layer[1]['ch_in'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])*layer[3])
            comp_std+=((layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])*layer[3])
        elif layer[2]==1:
            para_dw+=(layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]+layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0]+layer[0]*layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])
            comp_dw+=(layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])

    tmp_hw_spec1 = { \
        'gb_vol': math.ceil(tmp_hw_spec['gb_vol']*para_std/(para_dw+para_std)), \
        'rf_vol': math.ceil(tmp_hw_spec['rf_vol']*para_std/(para_dw+para_std)), \
        'num_pe': math.ceil(tmp_hw_spec['num_pe']*comp_std/(comp_dw+comp_std)), \
        'num_rf': math.ceil(tmp_hw_spec['num_rf']*comp_std/(comp_dw+comp_std))
    }
    tmp_hw_spec2 = { \
        'gb_vol': math.ceil(tmp_hw_spec['gb_vol']*para_dw/(para_dw+para_std)), \
        'rf_vol': math.ceil(tmp_hw_spec['rf_vol']*para_dw/(para_dw+para_std)), \
        'num_pe': math.ceil(tmp_hw_spec['num_pe']*comp_dw/(comp_dw+comp_std)), \
        'num_rf': math.ceil(tmp_hw_spec['num_rf']*comp_dw/(comp_dw+comp_std))
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


input_dnn= [[1, {'ch_out': [16, 0], 'ch_in': [3, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 0, 1], [1, {'ch_out': [96.0, 0], 'ch_in': [16.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [16.0, 0], 'ch_in': [96.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96.0, 0], 'ch_in': [16.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [24.0, 0], 'ch_in': [96.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [72.0, 0], 'ch_in': [24.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [72, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [24.0, 0], 'ch_in': [72.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [144.0, 0], 'ch_in': [24.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [2, {'ch_out': [144, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [32.0, 0], 'ch_in': [144.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [192.0, 0], 'ch_in': [32.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [192, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [32.0, 0], 'ch_in': [192.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96.0, 0], 'ch_in': [32.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [32.0, 0], 'ch_in': [96.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96.0, 0], 'ch_in': [32.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [32.0, 0], 'ch_in': [96.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [192.0, 0], 'ch_in': [32.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [2, {'ch_out': [192, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [64.0, 0], 'ch_in': [192.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [192.0, 0], 'ch_in': [64.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [192, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [64.0, 0], 'ch_in': [192.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [192.0, 0], 'ch_in': [64.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [192, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [64.0, 0], 'ch_in': [192.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [64.0, 0], 'ch_in': [64.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [64, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [64.0, 0], 'ch_in': [64.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [384.0, 0], 'ch_in': [64.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [384, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [112.0, 0], 'ch_in': [384.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [672.0, 0], 'ch_in': [112.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [672, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [112.0, 0], 'ch_in': [672.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [672.0, 0], 'ch_in': [112.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [2, {'ch_out': [672, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [184.0, 0], 'ch_in': [672.0, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [1104.0, 0], 'ch_in': [184.0, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [1104, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [184.0, 0], 'ch_in': [1104.0, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [1104.0, 0], 'ch_in': [184.0, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [1104, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [352.0, 0], 'ch_in': [1104.0, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [1504, 0], 'ch_in': [352, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]]
[tmp_hw_spec1,tmp_hw_spec2]=resource_allocator_depth_std(input_dnn,tmp_hw_spec)
print(tmp_hw_spec1)
print(tmp_hw_spec2)
input_dnn_dw = []
input_dnn_std = []
for layer in copy.deepcopy(input_dnn):
    if layer[2] == 1:
        input_dnn_dw.append(layer)
    else:
        input_dnn_std.append(layer)
tiling_noc_std=[8, 9, 8]
tiling_gb_std= [9, 2, 6, 6, 1, 9]
pe_array_std=3
tiling_noc_dw=[7, 6, 1]
tiling_gb_dw= [4, 9, 10, 10, 7]
pe_array_dw=3
consumption_std = hardware_translation(tiling_noc_std, tiling_gb_std, pe_array_std, tmp_hw_spec1)
consumption_dw = hardware_translation_dw(tiling_noc_dw, tiling_gb_dw, pe_array_dw, tmp_hw_spec2)
[tiling_string_std, consumption_std] = tiling_translation(consumption_std, input_dnn_std)
[tiling_string_dw, consumption_dw] = tiling_translation_dw(consumption_dw, input_dnn_dw)
print(consumption_dw)
print(consumption_std)
exit()
trials=500
out_of_limit_num=0
for _ in range(trials):

    #split the network from std/group and depthwise
    input_dnn_dw=[]
    input_dnn_std=[]
    for layer in copy.deepcopy(input_dnn):
        if layer[2]==1:
            input_dnn_dw.append(layer)
        else:
            input_dnn_std.append(layer)

    #pick a pe array
    pe_array_std=randint(0,3)
    pe_array_dw=randint(0,3)

    #complete the rest of the lp_order: local buffer(rf), global buffer(gb), dram
    # one set for standard conv/group conv
    input_lp_order_gb_std=list(range(7))
    shuffle(input_lp_order_gb_std)
    input_lp_order_dram_std=list(range(7))
    shuffle(input_lp_order_dram_std)

    # another set for depthwise conv
    input_lp_order_gb_dw=list(range(6))
    shuffle(input_lp_order_gb_dw)
    input_lp_order_dram_dw=list(range(6))
    shuffle(input_lp_order_dram_dw)

    #translate the lp_order to string format
    lp_order_string_std=dram_invariant_looporder(pe_array_std,input_lp_order_dram_std, input_lp_order_gb_std)
    lp_order_string_dw=dram_invariant_looporder_dw(pe_array_dw,input_lp_order_dram_dw, input_lp_order_gb_dw)


    #tiling represent the ratio of the dimensions for each buffer
    #can not be zero!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    #std
    tiling_noc_std=[]
    if pe_array_std==3:
        for i in range(3):
            tiling_noc_std.append(randint(1,10))
    else:
        for i in range(4):
            tiling_noc_std.append(randint(1,10))
    #dw
    tiling_noc_dw=[]
    for i in range(3):
        tiling_noc_dw.append(randint(1,10))
    #dw -- std
    tiling_gb_std= []
    tiling_gb_dw=[]
    for i in range(6):
        tiling_gb_std.append(randint(1,10))
    for i in range(5):
        tiling_gb_dw.append(randint(1,10))

    #dw -- std
    consumption_std = hardware_translation( tiling_noc_std, tiling_gb_std, pe_array_std, tmp_hw_spec1)
    consumption_dw = hardware_translation_dw(tiling_noc_dw, tiling_gb_dw, pe_array_dw, tmp_hw_spec2)
    [tiling_string_std, consumption_std] = tiling_translation(consumption_std, input_dnn_std)
    [tiling_string_dw, consumption_dw] = tiling_translation_dw(consumption_dw, input_dnn_dw)
    #pass for EDP feedback
    #print(pe_array)
    # print(tiling_string)
    # print(lp_order_string)
    penalty_std=get_score_whole_dnn(tiling_string_std, consumption_std, tmp_hw_spec1, lp_order_string_std, input_dnn_std)
    penalty_dw = get_score_whole_dnn(tiling_string_dw, consumption_dw, tmp_hw_spec2, lp_order_string_dw, input_dnn_dw)
    print(penalty_std)
    print(penalty_dw)
    total_score=(penalty_std[0][0]+penalty_dw[0][0])*(penalty_std[0][1]+penalty_dw[0][1])
    print(total_score)
    if (not penalty_std[1]) or (not penalty_dw[1]):
        out_of_limit_num+=1
print(out_of_limit_num/trials)

    # get_score_whole_dnn(tiling_string,input_dnn[0][0],tmp_hw_spec,df_order=lp_order_string)
    # print(life_eval(tiling_string,input_dnn[0][0],tmp_hw_spec,df_order=lp_order_string))







