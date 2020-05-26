from random import random, randint, shuffle
import numpy as np
import test_for_eyeriss as simnas
import time
from itertools import combinations, permutations
import copy
from multiprocessing import Queue
import multiprocessing
import math
from ev_util import *
# for saving np to matlab
import scipy.io as sio
from sympy.solvers import solve
from sympy import Symbol


def dram_invariant_looporder(pe_array, input_lp_order_dram, input_lp_order_gb, input_lp_order_rf):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................

    input_lp_order_dram = copy.deepcopy(input_lp_order_dram)
    input_lp_order_gb = copy.deepcopy(input_lp_order_gb)
    input_lp_order_rf = copy.deepcopy(input_lp_order_rf)
    if not (len(input_lp_order_gb) == len(set(input_lp_order_gb)) and len(input_lp_order_dram) == len(
            set(input_lp_order_dram))):
        raise Exception('Please provide lp_order with no duplicate elements')
    input_rf = copy.deepcopy(noc_template[pe_array])
    lp_order_template_dram = ['col_out_dram', 'ch_out_dram', 'batch_dram', 'ch_in_dram', 'row_out_dram',
                              'col_kernel_dram', 'row_kernel_dram']
    lp_order_template_gb = ['ch_out_gb', 'ch_in_gb', 'col_kernel_gb', 'row_out_gb', 'batch_gb', 'col_out_gb',
                            'row_kernel_gb']
    lp_order_template_rf = ['col_out_rf', 'batch_rf', 'row_out_rf', 'ch_out_rf', 'row_kernel_rf', 'ch_in_rf',
                            'col_kernel_rf']

    lp_order_string = []
    for i in range(len(lp_order_template_rf)):
        lp_order_string.append(lp_order_template_rf[input_lp_order_rf[i]])
    lp_order_string += input_rf
    for i in range(len(lp_order_template_gb)):
        lp_order_string.append(lp_order_template_gb[input_lp_order_gb[i]])
    for i in range(len(lp_order_template_dram)):
        lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    return copy.deepcopy(lp_order_string)


def dram_invariant_looporder_dw(pe_array, input_lp_order_dram, input_lp_order_gb, input_lp_order_rf):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................

    input_lp_order_dram = copy.deepcopy(input_lp_order_dram)
    input_lp_order_gb = copy.deepcopy(input_lp_order_gb)
    input_lp_order_rf = copy.deepcopy(input_lp_order_rf)
    if not (len(input_lp_order_gb) == len(set(input_lp_order_gb)) and len(input_lp_order_dram) == len(
            set(input_lp_order_dram))):
        raise Exception('Please provide lp_order with no duplicate elements')
    input_rf = copy.deepcopy(noc_template_dw[pe_array])
    lp_order_template_dram = ['col_out_dram', 'ch_out_dram', 'batch_dram', 'row_out_dram',
                              'col_kernel_dram', 'row_kernel_dram']
    lp_order_template_gb = ['ch_out_gb', 'col_kernel_gb', 'row_out_gb', 'batch_gb', 'col_out_gb',
                            'row_kernel_gb']
    lp_order_template_rf = ['col_out_rf', 'batch_rf', 'row_out_rf', 'ch_out_rf', 'row_kernel_rf',
                            'col_kernel_rf']

    lp_order_string = []
    for i in range(len(lp_order_template_rf)):
        lp_order_string.append(lp_order_template_rf[input_lp_order_rf[i]])
    lp_order_string += input_rf
    for i in range(len(lp_order_template_gb)):
        lp_order_string.append(lp_order_template_gb[input_lp_order_gb[i]])
    for i in range(len(lp_order_template_dram)):
        lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    return copy.deepcopy(lp_order_string)


# input_dnn=[\
# [1,{'ch_out':[32,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[32,0],'ch_in':[32,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[64,0],'ch_in':[32,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[256,0],'ch_in':[64,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[128,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[128,0],'ch_in':[128,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# ]

input_dnn = [ \
    [1, {'ch_out': [96, 0], 'ch_in': [3, 0], 'batch': [1, 0], 'col_out': [56, 0], 'row_out': [56, 0],
         'row_kernel': [11, 0], 'col_kernel': [11, 0]}, 0, 1], \
    [1, {'ch_out': [256, 0], 'ch_in': [48, 0], 'batch': [1, 0], 'col_out': [28, 0], 'row_out': [28, 0],
         'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 0, 1], \
    [1, {'ch_out': [384, 0], 'ch_in': [256, 0], 'batch': [1, 0], 'col_out': [14, 0], 'row_out': [14, 0],
         'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 0, 1], \
    [1, {'ch_out': [384, 0], 'ch_in': [192, 0], 'batch': [1, 0], 'col_out': [14, 0], 'row_out': [14, 0],
         'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 0, 1], \
    [1, {'ch_out': [256, 0], 'ch_in': [192, 0], 'batch': [1, 0], 'col_out': [14, 0], 'row_out': [14, 0],
         'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 0, 1], \
    ]
# input_dnn=[\
# [1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
# [1,{'ch_out':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
# [1,{'ch_out':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
# [1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
# [1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
# [1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
# [1,{'ch_out':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
# [1,{'ch_out':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
# ]

# eyeriss
tmp_hw_spec = { \
    'gb_vol': 108 * 1024 * 8, \
    'rf_vol': 512 * 8, \
    'num_pe': 129, \
    'num_rf': 129
}





def hardware_translation(ratio_rf, ratio_noc, ratio_gb, pe_array, tmp_hw_spec, bw=16):
    # in_ch, out_ch, X, Y, X_K, Y_K
    rf_vol = tmp_hw_spec['rf_vol'] / bw
    gb_vol = tmp_hw_spec['gb_vol'] / bw
    pe_num = tmp_hw_spec['num_pe']
    a = (ratio_rf[0] * (ratio_rf[2] + ratio_rf[4]) * (ratio_rf[3] + ratio_rf[5])) + (
                ratio_rf[1] * ratio_rf[2] * ratio_rf[3])
    # a=(ratio_rf[0]*(ratio_rf[2])*(ratio_rf[3]))+(ratio_rf[1] * ratio_rf[2] * ratio_rf[3])
    b = (ratio_rf[0] * ratio_rf[1] * ratio_rf[4] * ratio_rf[5])
    c = ratio_rf[0] * (ratio_rf[2] + ratio_rf[4]) + ratio_rf[0] * (ratio_rf[3] + ratio_rf[5])
    d = ratio_rf[0]
    x = 0
    # solve poly
    roots = np.roots([b, a, c, d, -rf_vol])
    roots = roots[np.isreal(roots)]
    for i in roots:
        if float(i.real) > 0:
            x = (i.real)
    # print(b*x**4+a*x**3+c*x**2+d*x)
    # print(ratio_rf[0]*x,ratio_rf[1]*x,ratio_rf[2]*x,ratio_rf[3]*x,ratio_rf[4]*x,ratio_rf[5]*x)
    # calculate rf
    consumption_dict = {}
    consumption_dict["ch_in_rf"] = math.floor(max(ratio_rf[0] * x, 1))
    consumption_dict["ch_out_rf"] = math.floor(max(ratio_rf[1] * x, 1))
    consumption_dict["col_out_rf"] = math.floor(max(ratio_rf[2] * x, 1))
    consumption_dict["row_out_rf"] = math.floor(max(ratio_rf[3] * x, 1))
    consumption_dict["col_kernel_rf"] = math.floor(max(ratio_rf[4] * x, 1))
    consumption_dict["row_kernel_rf"] = math.floor(max(ratio_rf[5] * x, 1))
    # print(consumption_dict)
    scaling_ratio = (rf_vol / (consumption_dict["ch_in_rf"] * (
                consumption_dict["row_out_rf"] + consumption_dict['row_kernel_rf'] - 1) * (
                                           consumption_dict["col_out_rf"] + consumption_dict["col_kernel_rf"] - 1) +
                               consumption_dict["ch_out_rf"] * consumption_dict["row_out_rf"] * consumption_dict[
                                   "col_out_rf"] + \
                               consumption_dict["ch_in_rf"] * consumption_dict["ch_out_rf"] * consumption_dict[
                                   "col_kernel_rf"] * consumption_dict["row_kernel_rf"]))
    while scaling_ratio < 1:
        components_need_scaled = {}
        for i in consumption_dict:
            if ("rf" in i):
                if consumption_dict[i] != 1:
                    components_need_scaled[str(i)] = consumption_dict[i]
        if len(components_need_scaled) == sum(components_need_scaled.values()):
            print('can not fit the gb requirement')
            raise
        for i in components_need_scaled:
            consumption_dict[i] -= 1
            consumption_dict[i] = max(consumption_dict[i], 0)
        scaling_ratio = (rf_vol / (consumption_dict["ch_in_rf"] * (
                    consumption_dict["row_out_rf"] + consumption_dict['row_kernel_rf'] - 1) * (
                                               consumption_dict["col_out_rf"] + consumption_dict["col_kernel_rf"] - 1) +
                                   consumption_dict["ch_out_rf"] * consumption_dict["row_out_rf"] * consumption_dict[
                                       "col_out_rf"] + \
                                   consumption_dict["ch_in_rf"] * consumption_dict["ch_out_rf"] * consumption_dict[
                                       "col_kernel_rf"] * consumption_dict["row_kernel_rf"]))

    # calculate pe
    if pe_array == 3:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2]) ** (1 / 3)
        consumption_dict['row_out_noc'] = max(math.floor(y * ratio_noc[0]), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1]), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2]), 1)
        scaling_ratio = (pe_num / (
                    consumption_dict['row_out_noc'] * consumption_dict['col_out_noc'] * consumption_dict['ch_out_noc']))
        if scaling_ratio < 1:
            components_need_scaled = []
            for i in consumption_dict:
                if "noc" in i:
                    if consumption_dict[i] != 1:
                        components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(
                    consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 0:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2] / ratio_noc[3]) ** (1 / 4)
        consumption_dict['col_kernel_noc'] = max(math.floor(y * ratio_noc[0]), 1)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[1] * y), 1)
        consumption_dict['ch_in_noc'] = max(math.floor(y * ratio_noc[2]), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[3]), 1)
        scaling_ratio = (pe_num / (
                    consumption_dict['col_kernel_noc'] * consumption_dict['row_kernel_noc'] * consumption_dict[
                'ch_in_noc'] * consumption_dict['ch_out_noc']))
        if scaling_ratio < 1:
            components_need_scaled = []
            for i in consumption_dict:
                if "noc" in i:
                    if consumption_dict[i] != 1:
                        components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(
                    consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 1:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2] / ratio_noc[3]) ** (1 / 4)
        consumption_dict['col_kernel_noc'] = max(math.floor(y * ratio_noc[0]), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1]), 1)
        consumption_dict['ch_in_noc'] = max(math.floor(y * ratio_noc[2]), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[3]), 1)
        scaling_ratio = (pe_num / (
                    consumption_dict['col_kernel_noc'] * consumption_dict['col_out_noc'] * consumption_dict[
                'ch_in_noc'] * consumption_dict['ch_out_noc']))
        if scaling_ratio < 1:
            components_need_scaled = []
            for i in consumption_dict:
                if "noc" in i:
                    if consumption_dict[i] != 1:
                        components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(
                    consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 2:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2] / ratio_noc[3]) ** (1 / 4)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[0]), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1]), 1)
        consumption_dict['ch_in_noc'] = max(math.floor(y * ratio_noc[2]), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[3]), 1)
        scaling_ratio = (pe_num / (
                    consumption_dict['row_kernel_noc'] * consumption_dict['col_out_noc'] * consumption_dict[
                'ch_in_noc'] * consumption_dict['ch_out_noc']))
        if scaling_ratio < 1:
            components_need_scaled = []
            for i in consumption_dict:
                if "noc" in i:
                    if consumption_dict[i] != 1:
                        components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(
                    consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    # calculate gb

    in_rf_consumption = consumption_dict["ch_in_rf"] * (
                consumption_dict["col_out_rf"] + consumption_dict['col_kernel_rf'] - 1) * (
                                    consumption_dict["row_out_rf"] + consumption_dict['row_kernel_rf'] - 1)
    out_rf_consumption = consumption_dict["ch_out_rf"] * consumption_dict["col_out_rf"] * consumption_dict["row_out_rf"]
    we_rf_consumption = consumption_dict["ch_in_rf"] * consumption_dict["ch_out_rf"] * consumption_dict[
        "col_kernel_rf"] * consumption_dict["row_kernel_rf"]
    print((in_rf_consumption + out_rf_consumption + we_rf_consumption) * 16)
    in_rf_consumption_for_all_pes = in_rf_consumption
    out_rf_consumption_for_all_pes = out_rf_consumption
    we_rf_consumption_for_all_pes = we_rf_consumption
    for i in consumption_dict:
        if 'noc' in i:
            if 'ch_in' in i:
                in_rf_consumption_for_all_pes *= consumption_dict[i]
                we_rf_consumption_for_all_pes *= consumption_dict[i]
            elif 'ch_out' in i:
                out_rf_consumption_for_all_pes *= consumption_dict[i]
                we_rf_consumption_for_all_pes *= consumption_dict[i]
            elif ('col_out' in i) or ('row_out' in i):
                in_rf_consumption_for_all_pes *= consumption_dict[i]
                out_rf_consumption_for_all_pes *= consumption_dict[i]
            elif ('row_kernel' in i) or ('col_kernel' in i):
                we_rf_consumption_for_all_pes *= consumption_dict[i]
            else:
                pass
    a = (ratio_gb[0] * ratio_gb[2] * ratio_gb[3]) * in_rf_consumption_for_all_pes + (
                ratio_gb[1] * ratio_gb[2] * ratio_gb[3]) * out_rf_consumption_for_all_pes
    b = (ratio_gb[0] * ratio_gb[1] * ratio_gb[4] * ratio_gb[5]) * we_rf_consumption_for_all_pes
    roots = np.roots([b, a, 0, 0, -gb_vol])
    roots = roots[np.isreal(roots)]
    z = 0
    for i in roots:
        if float(i.real) > 0:
            z = i.real
    consumption_dict["ch_in_gb"] = math.floor(max(ratio_gb[0] * z, 1))
    consumption_dict["ch_out_gb"] = math.floor(max(ratio_gb[1] * z, 1))
    consumption_dict["col_out_gb"] = math.floor(max(ratio_gb[2] * z, 1))
    consumption_dict["row_out_gb"] = math.floor(max(ratio_gb[3] * z, 1))
    consumption_dict["col_kernel_gb"] = math.floor(max(ratio_gb[4] * z, 1))
    consumption_dict["row_kernel_gb"] = math.floor(max(ratio_gb[5] * z, 1))
    scaling_ratio = (gb_vol / (consumption_dict["ch_in_gb"] * (
                consumption_dict["row_out_gb"] + consumption_dict['row_kernel_gb'] - 1) * (
                                           consumption_dict["col_out_gb"] + consumption_dict[
                                       "col_kernel_gb"] - 1) * in_rf_consumption_for_all_pes + consumption_dict[
                                   "ch_out_gb"] * consumption_dict["row_out_gb"] * consumption_dict[
                                   "col_out_gb"] * out_rf_consumption_for_all_pes + \
                               consumption_dict["ch_in_gb"] * consumption_dict["ch_out_gb"] * consumption_dict[
                                   "col_kernel_gb"] * consumption_dict[
                                   "row_kernel_gb"] * we_rf_consumption_for_all_pes))
    while scaling_ratio < 1:
        components_need_scaled = {}
        for i in consumption_dict:
            if ("gb" in i):
                if consumption_dict[i] != 1:
                    components_need_scaled[str(i)] = consumption_dict[i]
        if len(components_need_scaled) == sum(components_need_scaled.values()):
            print('can not fit the gb requirement')
            raise
        for i in components_need_scaled:
            consumption_dict[i] -= 1
            consumption_dict[i] = max(consumption_dict[i], 0)
        scaling_ratio = (gb_vol / (consumption_dict["ch_in_gb"] * (
                    consumption_dict["row_out_gb"] + consumption_dict['row_kernel_gb'] - 1) * (
                                               consumption_dict["col_out_gb"] + consumption_dict[
                                           "col_kernel_gb"] - 1) * in_rf_consumption_for_all_pes + consumption_dict[
                                       "ch_out_gb"] * consumption_dict["row_out_gb"] * consumption_dict[
                                       "col_out_gb"] * out_rf_consumption_for_all_pes + \
                                   consumption_dict["ch_in_gb"] * consumption_dict["ch_out_gb"] * consumption_dict[
                                       "col_kernel_gb"] * consumption_dict[
                                       "row_kernel_gb"] * we_rf_consumption_for_all_pes))

    return consumption_dict


def hardware_translation_dw(ratio_rf, ratio_noc, ratio_gb, pe_array, tmp_hw_spec, bw=16):
    # out_ch, X, Y, X_K, Y_K
    rf_vol = tmp_hw_spec['rf_vol'] / bw
    gb_vol = tmp_hw_spec['gb_vol'] / bw
    pe_num = tmp_hw_spec['num_pe']
    a = (ratio_rf[0] * (ratio_rf[1] + ratio_rf[3]) * (ratio_rf[2] + ratio_rf[4])) + (
                ratio_rf[0] * ratio_rf[1] * ratio_rf[2])
    # a=(ratio_rf[0]*(ratio_rf[2])*(ratio_rf[3]))+(ratio_rf[1] * ratio_rf[2] * ratio_rf[3])
    b = (ratio_rf[0] * ratio_rf[3] * ratio_rf[4])
    c = ratio_rf[0] * (ratio_rf[1] + ratio_rf[3]) + ratio_rf[0] * (ratio_rf[2] + ratio_rf[4])
    d = ratio_rf[0]
    x = 0
    # solve poly
    roots = np.roots([b + a, c, d, -rf_vol])
    roots = roots[np.isreal(roots)]
    for i in roots:
        if float(i.real) > 0:
            x = (i.real)
    # calculate rf
    consumption_dict = {}
    consumption_dict["ch_out_rf"] = math.floor(max(ratio_rf[0] * x, 1))
    consumption_dict["col_out_rf"] = math.floor(max(ratio_rf[1] * x, 1))
    consumption_dict["row_out_rf"] = math.floor(max(ratio_rf[2] * x, 1))
    consumption_dict["col_kernel_rf"] = math.floor(max(ratio_rf[3] * x, 1))
    consumption_dict["row_kernel_rf"] = math.floor(max(ratio_rf[4] * x, 1))
    scaling_ratio = (rf_vol / (consumption_dict["ch_out_rf"] * (
                consumption_dict["row_out_rf"] + consumption_dict['row_kernel_rf'] - 1) * (
                                           consumption_dict["col_out_rf"] + consumption_dict["col_kernel_rf"] - 1) +
                               consumption_dict["ch_out_rf"] * consumption_dict["row_out_rf"] * consumption_dict[
                                   "col_out_rf"] + \
                               consumption_dict["ch_out_rf"] * consumption_dict["col_kernel_rf"] * consumption_dict[
                                   "row_kernel_rf"]))
    while scaling_ratio < 1:
        components_need_scaled = {}
        for i in consumption_dict:
            if ("rf" in i):
                if consumption_dict[i] != 1:
                    components_need_scaled[str(i)] = consumption_dict[i]
        if len(components_need_scaled) == sum(components_need_scaled.values()):
            print('can not fit the gb requirement')
            raise
        for i in components_need_scaled:
            consumption_dict[i] -= 1
            consumption_dict[i] = max(consumption_dict[i], 0)
        scaling_ratio = (rf_vol / (consumption_dict["ch_out_rf"] * (
                    consumption_dict["row_out_rf"] + consumption_dict['row_kernel_rf'] - 1) * (
                                               consumption_dict["col_out_rf"] + consumption_dict["col_kernel_rf"] - 1) +
                                   consumption_dict["ch_out_rf"] * consumption_dict["row_out_rf"] * consumption_dict[
                                       "col_out_rf"] + \
                                   consumption_dict["ch_out_rf"] * consumption_dict["col_kernel_rf"] * consumption_dict[
                                       "row_kernel_rf"]))

    # calculate pe
    if pe_array == 3:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2]) ** (1 / 3)
        consumption_dict['row_out_noc'] = max(math.floor(y * ratio_noc[0]), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1]), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2]), 1)
        scaling_ratio = (pe_num / (
                    consumption_dict['row_out_noc'] * consumption_dict['col_out_noc'] * consumption_dict['ch_out_noc']))
        if scaling_ratio < 1:
            components_need_scaled = []
            for i in consumption_dict:
                if "noc" in i:
                    if consumption_dict[i] != 1:
                        components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(
                    consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 0:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2]) ** (1 / 3)
        consumption_dict['col_kernel_noc'] = max(math.floor(y * ratio_noc[0]), 1)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[1] * y), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2]), 1)
        scaling_ratio = (pe_num / (
                    consumption_dict['col_kernel_noc'] * consumption_dict['row_kernel_noc'] * consumption_dict[
                'ch_out_noc']))
        if scaling_ratio < 1:
            components_need_scaled = []
            for i in consumption_dict:
                if "noc" in i:
                    if consumption_dict[i] != 1:
                        components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(
                    consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 1:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2]) ** (1 / 3)
        consumption_dict['col_kernel_noc'] = max(math.floor(y * ratio_noc[0]), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1]), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2]), 1)
        scaling_ratio = (pe_num / (
                    consumption_dict['col_kernel_noc'] * consumption_dict['col_out_noc'] * consumption_dict[
                'ch_out_noc']))
        if scaling_ratio < 1:
            components_need_scaled = []
            for i in consumption_dict:
                if "noc" in i:
                    if consumption_dict[i] != 1:
                        components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(
                    consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 2:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2]) ** (1 / 3)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[0]), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1]), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2]), 1)
        scaling_ratio = (pe_num / (
                    consumption_dict['row_kernel_noc'] * consumption_dict['col_out_noc'] * consumption_dict[
                'ch_out_noc']))
        if scaling_ratio < 1:
            components_need_scaled = []
            for i in consumption_dict:
                if "noc" in i:
                    if consumption_dict[i] != 1:
                        components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(
                    consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    # calculate gb
    in_rf_consumption = consumption_dict["ch_out_rf"] * (
                consumption_dict["col_out_rf"] + consumption_dict['col_kernel_rf'] - 1) * (
                                    consumption_dict["row_out_rf"] + consumption_dict['row_kernel_rf'] - 1)
    out_rf_consumption = consumption_dict["ch_out_rf"] * consumption_dict["col_out_rf"] * consumption_dict["row_out_rf"]
    we_rf_consumption = consumption_dict["ch_out_rf"] * consumption_dict["col_kernel_rf"] * consumption_dict[
        "row_kernel_rf"]
    # print((in_rf_consumption +out_rf_consumption+we_rf_consumption)*16)
    in_rf_consumption_for_all_pes = in_rf_consumption
    out_rf_consumption_for_all_pes = out_rf_consumption
    we_rf_consumption_for_all_pes = we_rf_consumption
    for i in consumption_dict:
        if 'noc' in i:
            if 'ch_out' in i:
                in_rf_consumption_for_all_pes *= consumption_dict[i]
                out_rf_consumption_for_all_pes *= consumption_dict[i]
                we_rf_consumption_for_all_pes *= consumption_dict[i]
            elif ('col_out' in i) or ('row_out' in i):
                in_rf_consumption_for_all_pes *= consumption_dict[i]
                out_rf_consumption_for_all_pes *= consumption_dict[i]
            elif ('row_kernel' in i) or ('col_kernel' in i):
                we_rf_consumption_for_all_pes *= consumption_dict[i]
            else:
                pass

    a = (ratio_gb[0] * ratio_gb[1] * ratio_gb[2]) * in_rf_consumption_for_all_pes + (
            ratio_gb[0] * ratio_gb[1] * ratio_gb[2]) * out_rf_consumption_for_all_pes
    b = (ratio_gb[0] * ratio_gb[3] * ratio_gb[4]) * we_rf_consumption_for_all_pes

    # roots=np.roots([b,a,0,0,-gb_vol])
    # roots=roots[np.isreal(roots)]
    # z=0
    # for i in roots:
    #     if float(i.real)>0:
    #         z=i.real

    z = (gb_vol / (a + b)) ** (1 / 3)
    consumption_dict["ch_out_gb"] = math.floor(max(ratio_gb[0] * z, 1))
    consumption_dict["col_out_gb"] = math.floor(max(ratio_gb[1] * z, 1))
    consumption_dict["row_out_gb"] = math.floor(max(ratio_gb[2] * z, 1))
    consumption_dict["col_kernel_gb"] = math.floor(max(ratio_gb[3] * z, 1))
    consumption_dict["row_kernel_gb"] = math.floor(max(ratio_gb[4] * z, 1))
    scaling_ratio = (gb_vol / (consumption_dict["ch_out_gb"] * (
            consumption_dict["row_out_gb"] + consumption_dict['row_kernel_gb'] - 1) * (
                                       consumption_dict["col_out_gb"] + consumption_dict[
                                   'col_kernel_gb'] - 1) * in_rf_consumption_for_all_pes + consumption_dict[
                                   "ch_out_gb"] * consumption_dict["row_out_gb"] * consumption_dict[
                                   "col_out_gb"] * out_rf_consumption_for_all_pes + \
                               consumption_dict["ch_out_gb"] * consumption_dict["col_kernel_gb"] * consumption_dict[
                                   "row_kernel_gb"] * we_rf_consumption_for_all_pes)) ** (1 / 3)
    while scaling_ratio < 1:
        components_need_scaled = {}
        for i in consumption_dict:
            if ("gb" in i):
                if consumption_dict[i] != 1:
                    components_need_scaled[str(i)] = consumption_dict[i]
        if len(components_need_scaled) == sum(components_need_scaled.values()):
            print('can not fit the gb requirement')
            raise
        for i in components_need_scaled:
            consumption_dict[i] -= 1
            consumption_dict[i] = max(consumption_dict[i], 0)
        scaling_ratio = (gb_vol / (consumption_dict["ch_out_gb"] * (
                consumption_dict["row_out_gb"] + consumption_dict['row_kernel_gb'] - 1) * (
                                               consumption_dict["col_out_gb"] + consumption_dict[
                                           'col_kernel_gb'] - 1) * in_rf_consumption_for_all_pes + consumption_dict[
                                       "ch_out_gb"] * consumption_dict["row_out_gb"] * consumption_dict[
                                       "col_out_gb"] * out_rf_consumption_for_all_pes + \
                                   consumption_dict["ch_out_gb"] * consumption_dict["col_kernel_gb"] * consumption_dict[
                                       "row_kernel_gb"] * we_rf_consumption_for_all_pes)) ** (1 / 3)
    return consumption_dict


def tiling_translation(consumption_dict, input_dnn):
    tiling_str = []
    for layer in input_dnn:
        tiling_str.append({})
        for i in consumption_dict:
            if "rf" in i:
                tiling_str[-1][i] = min(consumption_dict[i], layer[1][str(i)[:-3]][0])
        for i in consumption_dict:
            if "noc" in i:
                tiling_str[-1][i] = min(consumption_dict[i],
                                        math.ceil(layer[1][str(i)[:-4]][0] / tiling_str[-1][str(i)[:-4] + "_rf"]))
        for i in consumption_dict:
            if "gb" in i:
                try:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                            math.ceil(layer[1][str(i)[:-3]][0] / tiling_str[-1][str(i)[:-3] + "_noc"] /
                                                      tiling_str[-1][str(i)[:-3] + "_rf"]))
                except KeyError:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                            math.ceil(layer[1][str(i)[:-3]][0] /
                                                      tiling_str[-1][str(i)[:-3] + "_rf"]))
                except:
                    raise
        tiling_str[-1]['batch_rf'] = tiling_str[-1]['batch_gb'] = 1
        consumption_dict['batch_rf'] = consumption_dict['batch_gb'] = 1
        dram_list = ['col_out_dram', 'ch_out_dram', 'batch_dram', 'ch_in_dram', 'row_out_dram', 'col_kernel_dram',
                     'row_kernel_dram']
        for i in dram_list:
            consumption_dict[i] = 1
            try:
                tiling_str[-1][i] = math.ceil(
                    layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"] / tiling_str[-1][
                        str(i)[:-5] + "_noc"] / tiling_str[-1][
                        str(i)[:-5] + "_rf"])
            except KeyError:
                tiling_str[-1][i] = math.ceil(
                    layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"] / tiling_str[-1][
                        str(i)[:-5] + "_rf"])
            except:
                raise
    return tiling_str, consumption_dict


def tiling_translation_dw(consumption_dict, input_dnn):
    tiling_str = []
    for layer in input_dnn:
        tiling_str.append({})
        for i in consumption_dict:
            if "rf" in i:
                tiling_str[-1][i] = min(consumption_dict[i], layer[1][str(i)[:-3]][0])
        for i in consumption_dict:
            if "noc" in i:
                tiling_str[-1][i] = min(consumption_dict[i],
                                        math.ceil(layer[1][str(i)[:-4]][0] / tiling_str[-1][str(i)[:-4] + "_rf"]))
        for i in consumption_dict:
            if "gb" in i:
                try:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                            math.ceil(layer[1][str(i)[:-3]][0] / tiling_str[-1][str(i)[:-3] + "_noc"] /
                                                      tiling_str[-1][str(i)[:-3] + "_rf"]))
                except KeyError:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                            math.ceil(layer[1][str(i)[:-3]][0] /
                                                      tiling_str[-1][str(i)[:-3] + "_rf"]))
                except:
                    raise
        tiling_str[-1]['batch_rf'] = tiling_str[-1]['batch_gb'] = 1
        consumption_dict['batch_rf'] = consumption_dict['batch_gb'] = 1
        dram_list = ['col_out_dram', 'ch_out_dram', 'batch_dram', 'row_out_dram', 'col_kernel_dram', 'row_kernel_dram']
        for i in dram_list:
            consumption_dict[i] = 1
            try:
                tiling_str[-1][i] = math.ceil(
                    layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"] / tiling_str[-1][
                        str(i)[:-5] + "_noc"] / tiling_str[-1][
                        str(i)[:-5] + "_rf"])
            except KeyError:
                tiling_str[-1][i] = math.ceil(
                    layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"] / tiling_str[-1][
                        str(i)[:-5] + "_rf"])
            except:
                raise
    return tiling_str, consumption_dict


def get_score_whole_dnn(tiling_string, consumption, tmp_hw_spec, lp_order_string, input_dnn):
    # check for resource consumption
    [penalty, buffer_not_exceed] = life_eval(consumption, 1, tmp_hw_spec, 0, group_num=1, df_order=lp_order_string)
    if not buffer_not_exceed:
        print('consumption is out of limit')
        return [(penalty[0], penalty[1]), buffer_not_exceed]
    edp_raw = [0, 0]
    for layer in range(len(input_dnn)):
        [penalty, buffer_not_exceed] = life_eval(tiling_string[layer], input_dnn[layer][0], tmp_hw_spec,
                                                 input_dnn[layer][2], group_num=input_dnn[layer][3],
                                                 df_order=lp_order_string)
        if not buffer_not_exceed:
            print('a oh...')
            return [(penalty[0], penalty[1]), buffer_not_exceed]
        else:
            edp_raw[0] += penalty[0]
            edp_raw[1] += penalty[1]
    return (edp_raw[0], edp_raw[1]), True

    # print(life_eval(tiling_string,input_dnn[0][0],tmp_hw_spec,df_order=lp_order_string))


def resource_allocator_depth_std(input_dnn, tmp_hw_spec):
    input_dnn = copy.deepcopy(input_dnn)
    input_dnn_std = []
    input_dnn_dw = []
    for layer in input_dnn:
        if layer[2] == 1:
            input_dnn_dw.append(layer)
        else:
            input_dnn_std.append(layer)
    para_dw = 0
    para_std = 0
    comp_dw = 0
    comp_std = 0
    for layer in input_dnn:
        if layer[2] == 0 or layer[2] == 2:
            para_std += ((layer[1]['ch_in'][0] * layer[1]['ch_out'][0] * layer[1]['col_kernel'][0] * \
                          layer[1]['row_kernel'][0] + layer[1]['ch_out'][0] * layer[1]['col_out'][0] * \
                          layer[1]['row_out'][0] + layer[0] * layer[1]['ch_in'][0] * layer[1]['col_out'][0] * \
                          layer[1]['row_out'][0]) * layer[3])
            comp_std += ((layer[1]['ch_in'][0] * layer[1]['ch_out'][0] * layer[1]['col_kernel'][0] * \
                          layer[1]['row_kernel'][0] * layer[1]['col_out'][0] * \
                          layer[1]['row_out'][0]) * layer[3])
        elif layer[2] == 1:
            para_dw += (layer[1]['ch_out'][0] * layer[1]['col_kernel'][0] * \
                        layer[1]['row_kernel'][0] + layer[1]['ch_out'][0] * layer[1]['col_out'][0] * \
                        layer[1]['row_out'][0] + layer[0] * layer[1]['ch_out'][0] * layer[1]['col_out'][0] * \
                        layer[1]['row_out'][0])
            comp_dw += (layer[1]['ch_out'][0] * layer[1]['col_kernel'][0] * \
                        layer[1]['row_kernel'][0] * layer[1]['col_out'][0] * \
                        layer[1]['row_out'][0])

    tmp_hw_spec1 = { \
        'gb_vol': math.ceil(tmp_hw_spec['gb_vol'] * para_std / (para_dw + para_std)), \
        'rf_vol': math.ceil(tmp_hw_spec['rf_vol']), \
        'num_pe': math.ceil(tmp_hw_spec['num_pe'] * comp_std / (comp_dw + comp_std)), \
        'num_rf': math.ceil(tmp_hw_spec['num_rf'] * comp_std / (comp_dw + comp_std))
    }
    tmp_hw_spec2 = { \
        'gb_vol': math.ceil(tmp_hw_spec['gb_vol'] * para_dw / (para_dw + para_std)), \
        'rf_vol': math.ceil(tmp_hw_spec['rf_vol']), \
        'num_pe': math.ceil(tmp_hw_spec['num_pe'] * comp_dw / (comp_dw + comp_std)), \
        'num_rf': math.ceil(tmp_hw_spec['num_rf'] * comp_dw / (comp_dw + comp_std))
    }

    return tmp_hw_spec1, tmp_hw_spec2


def lp_visualizer(lp_order, tiling, mode):
    lp_order = ref_location_optimization(lp_order, tiling[2], mode)
    indexer = {"ch_in_rf": "C0", "ch_in_noc": "C1", "ch_in_gb": "C2", "ch_in_dram": "C3", \
               "ch_out_rf": "M0", "ch_out_noc": "M1", "ch_out_gb": "M2", "ch_out_dram": "M3", \
               "row_out_rf": "E0", "row_out_noc": "E1", "row_out_gb": "E2", "row_out_dram": "E3", \
               "col_out_rf": "F0", "col_out_noc": "F1", "col_out_gb": "F2", "col_out_dram": "F3", \
               "row_kernel_rf": "R0", "row_kernel_noc": "R1", "row_kernel_gb": "R2", "row_kernel_dram": "R3", \
               "col_kernel_rf": "S0", "col_kernel_noc": "S1", "col_kernel_gb": "S2", "col_kernel_dram": "S3", \
               }
    tab = "  "
    lp_order = lp_order[::-1]
    if mode == 0 or mode == 2 or mode == 1:
        output_string = ""
        for layer in range(len(tiling)):
            output_string += "=" * 20 + "\n\r"
            output_string += "layer " + str(layer) + "\n\r"
            output_string += "=" * 20 + "\n\r"
            for i in range(len(lp_order)):
                if 'batch' in lp_order[i]:
                    continue
                if 'ref' in lp_order[i]:
                    output_string += i * tab
                    output_string += "//" + lp_order[i] + "- " * 10 + "\n\r"
                elif 'noc' in lp_order[i]:
                    output_string += i * tab
                    output_string += "parallel-for  (" + indexer[lp_order[i]] + "=0 ;" + indexer[
                        lp_order[i]] + "< " + str(tiling[layer][lp_order[i]]) + "; " + indexer[lp_order[i]] + "= " + \
                                     indexer[lp_order[i]] + "+1)"
                    output_string += "  //" + lp_order[i] + "\n\r"
                else:
                    output_string += i * tab
                    output_string += "for  (" + indexer[lp_order[i]] + "=0 ;" + indexer[lp_order[i]] + "< " + str(
                        tiling[layer][lp_order[i]]) + "; " + indexer[lp_order[i]] + "= " + indexer[lp_order[i]] + "+1)"
                    output_string += "  //" + lp_order[i] + "\n\r"
    return output_string


def access_count(tiling_string_std,tiling_string_dw,input_dnn_std,input_dnn_dw,tmp_hw_spec1,tmp_hw_spec2,lp_order_string_std,lp_order_string_dw):


    dram_to_gb = 0
    gb_to_noc = 0
    noc_to_rf =0
    rf_to_alu = 0
    lat_std=0
    lat_dw=0
    for layer in range(len(input_dnn_std)):
        tiling_string=tiling_string_std[layer]
        ene_results = simnas.sample_energy(tiling_string, input_dnn_std[layer][0], tmp_hw_spec1, input_dnn_std[layer][2], input_df_order=lp_order_string_std)
        ebit_dram_to_gb = 200 / 16  # energy/bit for the dram->gb data communication
        # gb->noc
        ebit_gb_to_noc = 3.78 / 8 / 2
        # noc->rf
        ebit_noc_to_rf = 0.89 / 8 * 2 / 3
        # rf->alu
        ebit_rf_to_alu = 0.89 / 8 / 2
        # print(ene_results)
        lat_std+=ene_results[1]
        dram_to_gb+=ene_results[2]['E_dram_to_gb']/ebit_dram_to_gb
        gb_to_noc+=ene_results[2]['E_gb_to_noc']/ebit_gb_to_noc
        noc_to_rf+=ene_results[2]['E_noc_to_rf']/ebit_noc_to_rf
        rf_to_alu+=ene_results[2]['E_rf_to_alu']/ebit_rf_to_alu

    for layer in range(len(input_dnn_dw)):
        tiling_string = tiling_string_dw[layer]
        ene_results = simnas.sample_energy(tiling_string, input_dnn_dw[layer][0], tmp_hw_spec2, input_dnn_dw[layer][2], input_df_order=lp_order_string_dw)
        ebit_dram_to_gb = 200 / 16  # energy/bit for the dram->gb data communication
        # gb->noc
        ebit_gb_to_noc = 3.78 / 8 / 2
        # noc->rf
        ebit_noc_to_rf = 0.89 / 8 * 2 / 3
        # rf->alu
        ebit_rf_to_alu = 0.89 / 8 / 2
        lat_dw+=ene_results[1]
        dram_to_gb+=ene_results[2]['E_dram_to_gb']/ebit_dram_to_gb
        gb_to_noc+=ene_results[2]['E_gb_to_noc']/ebit_gb_to_noc
        noc_to_rf+=ene_results[2]['E_noc_to_rf']/ebit_noc_to_rf
        rf_to_alu+=ene_results[2]['E_rf_to_alu']/ebit_rf_to_alu

    return dram_to_gb,gb_to_noc,noc_to_rf,rf_to_alu,max(lat_std,lat_dw)



input_dnn=[[1, {'ch_out': [16, 0], 'ch_in': [3, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 0, 1], [1, {'ch_out': [96.0, 0], 'ch_in': [16.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [24.0, 0], 'ch_in': [96.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [24.0, 0], 'ch_in': [24.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [24, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [24.0, 0], 'ch_in': [24.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [24.0, 0], 'ch_in': [24.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [24, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [24.0, 0], 'ch_in': [24.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [144.0, 0], 'ch_in': [24.0, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [2, {'ch_out': [144, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [32.0, 0], 'ch_in': [144.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [32.0, 0], 'ch_in': [32.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [32, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [32.0, 0], 'ch_in': [32.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [32.0, 0], 'ch_in': [32.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [32, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [32.0, 0], 'ch_in': [32.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [192.0, 0], 'ch_in': [32.0, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [2, {'ch_out': [192, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [64.0, 0], 'ch_in': [192.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [384.0, 0], 'ch_in': [64.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [384, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [64.0, 0], 'ch_in': [384.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [64.0, 0], 'ch_in': [64.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [64, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [64.0, 0], 'ch_in': [64.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [384.0, 0], 'ch_in': [64.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [384, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [112.0, 0], 'ch_in': [384.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [56.0, 0], 'ch_in': [56.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 2, 2], [1, {'ch_out': [112, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [56.0, 0], 'ch_in': [56.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 2, 2], [1, {'ch_out': [672.0, 0], 'ch_in': [112.0, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [2, {'ch_out': [672, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [184.0, 0], 'ch_in': [672.0, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [92.0, 0], 'ch_in': [92.0, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 2, 2], [1, {'ch_out': [184, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [92.0, 0], 'ch_in': [92.0, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 2, 2], [1, {'ch_out': [92.0, 0], 'ch_in': [92.0, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 2, 2], [1, {'ch_out': [184, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [176.0, 0], 'ch_in': [92.0, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 2, 2], [1, {'ch_out': [1504, 0], 'ch_in': [352, 0], 'batch': [1, 0], 'col_out': [4, 0], 'row_out': [4, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]]
[tmp_hw_spec1,tmp_hw_spec2]=resource_allocator_depth_std(input_dnn,tmp_hw_spec)
print(tmp_hw_spec1)
print(tmp_hw_spec2)

#results generation

opt_hw={'pe_array_std': 3, 'pe_array_dw': 2, 'input_lp_order_dram_std': [3, 0, 5, 2, 1, 4, 6], 'input_lp_order_gb_std': [2, 1, 6, 5, 3, 4, 0], 'input_lp_order_rf_std': [1, 4, 5, 6, 0, 2, 3], 'input_lp_order_dram_dw': [2, 4, 5, 0, 3, 1], 'input_lp_order_gb_dw': [2, 0, 4, 1, 5, 3], 'input_lp_order_rf_dw': [1, 5, 0, 2, 3, 4], 'tiling_noc_std': [1, 2, 6], 'tiling_gb_std': [2, 1, 5, 5, 3, 3], 'tiling_rf_std': [4, 6, 7, 4, 1, 7], 'tiling_noc_dw': [7, 1, 1], 'tiling_gb_dw': [3, 10, 4, 2, 4], 'tiling_rf_dw': [8, 6, 5, 10, 2]}
input_lp_order_dram_std= opt_hw['input_lp_order_dram_std']
input_lp_order_gb_std=opt_hw['input_lp_order_gb_std']
input_lp_order_rf_std=opt_hw['input_lp_order_rf_std']
input_lp_order_dram_dw= opt_hw['input_lp_order_dram_dw']
input_lp_order_gb_dw=opt_hw['input_lp_order_gb_dw']
input_lp_order_rf_dw= opt_hw['input_lp_order_rf_dw']
tiling_rf_std= opt_hw['tiling_rf_std']
tiling_noc_std=opt_hw['tiling_noc_std']
tiling_gb_std=opt_hw['tiling_gb_std']
pe_array_std=opt_hw['pe_array_std']
tiling_rf_dw=opt_hw['tiling_rf_dw']
tiling_noc_dw=opt_hw['tiling_noc_dw']
tiling_gb_dw=opt_hw['tiling_gb_dw']
pe_array_dw=opt_hw['pe_array_dw']
input_dnn_dw = []
input_dnn_std = []
for layer in copy.deepcopy(input_dnn):
    if layer[2] == 1:
        input_dnn_dw.append(layer)
    else:
        input_dnn_std.append(layer)
lp_order_string_std=dram_invariant_looporder(pe_array_std,input_lp_order_dram_std, input_lp_order_gb_std,input_lp_order_rf_std)
lp_order_string_dw=dram_invariant_looporder_dw(pe_array_dw,input_lp_order_dram_dw, input_lp_order_gb_dw,input_lp_order_rf_dw)
consumption_std = hardware_translation(tiling_rf_std, tiling_noc_std, tiling_gb_std, pe_array_std, tmp_hw_spec1)
consumption_dw = hardware_translation_dw(tiling_rf_dw, tiling_noc_dw, tiling_gb_dw, pe_array_dw, tmp_hw_spec2)
if pe_array_std==1:
    consumption_std['col_kernel_noc']=1
    consumption_std['col_out_noc'] = consumption_std['col_out_noc']*2
[tiling_string_std, consumption_std] = tiling_translation(consumption_std, input_dnn_std)
[tiling_string_dw, consumption_dw] = tiling_translation_dw(consumption_dw, input_dnn_dw)
# print(tiling_string_std)
# print(lp_order_string_std)
print(consumption_std)
print('input: ', consumption_std['ch_in_rf']*consumption_std['col_out_rf']*consumption_std['row_out_rf']*2)
print('output: ', consumption_std['ch_out_rf']*consumption_std['col_out_rf']*consumption_std['row_out_rf']*2)
print('weight: ', consumption_std['ch_in_rf']*consumption_std['ch_out_rf']*consumption_std['col_kernel_rf']*consumption_std['row_kernel_rf']*2)
print(consumption_dw)
print('input: ', consumption_dw['ch_out_rf']*consumption_dw['col_out_rf']*consumption_dw['row_out_rf']*2)
print('output: ', consumption_dw['ch_out_rf']*consumption_dw['col_out_rf']*consumption_dw['row_out_rf']*2)
print('weight: ', consumption_dw['ch_out_rf']*consumption_dw['col_kernel_rf']*consumption_dw['row_kernel_rf']*2)
#print(lp_visualizer(lp_order_string_dw,tiling_string_dw,1))
#print(lp_visualizer(lp_order_string_std,tiling_string_std,0))

access_count1=access_count(tiling_string_std,tiling_string_dw,input_dnn_std, input_dnn_dw,tmp_hw_spec1,tmp_hw_spec2,lp_order_string_std,lp_order_string_dw)
print(access_count1)
exit()