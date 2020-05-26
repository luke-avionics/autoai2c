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


layer=2
max_dim_choices=1
# input_dnn=[\
# # [1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[128,0],'ch_in':[64,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[128,0],'ch_in':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[512,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# # [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# ]


input_dnn=[\
[1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
[1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
[1,{'ch_out':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,4],\
[1,{'ch_out':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,16],\
[1,{'ch_out':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,8],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
]


#fpga dedicated 706
tmp_hw_spec={\
    'gb_vol':20*1024*1024, \
    'rf_vol':512, \
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
input_dnn=[[1, {'ch_out': [16, 0], 'ch_in': [3, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 0, 1], [1, {'ch_out': [96, 0], 'ch_in': [16, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [32, 0], 'ch_in': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96, 0], 'ch_in': [32, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1], [1, {'ch_out': [64, 0], 'ch_in': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [384, 0], 'ch_in': [64, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [2, {'ch_out': [384, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [112, 0], 'ch_in': [384, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [112, 0], 'ch_in': [112, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [112, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [112, 0], 'ch_in': [112, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [672, 0], 'ch_in': [112, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [2, {'ch_out': [672, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [184, 0], 'ch_in': [672, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [1104, 0], 'ch_in': [184, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [1104, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1], [1, {'ch_out': [352, 0], 'ch_in': [1104, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1], [1, {'ch_out': [1504, 0], 'ch_in': [352, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]]
opt_hw=[{'pe_array': 1, 'input_lp_order_gb': [6, 2, 0, 1, 4, 3, 5], 'input_lp_order_dram': [2, 5, 1, 0, 4, 6, 3], 'tiling_choices': [0, 0, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 1, 1, 0], 'input_dnn_all': [[1, {'ch_out': [16, 0], 'ch_in': [3, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 0, 1]], 'edp': 0.1689920596152856}, {'pe_array': 0, 'input_lp_order_gb': [4, 5, 3, 0, 2, 6, 1], 'input_lp_order_dram': [5, 3, 2, 0, 6, 4, 1], 'tiling_choices': [0, 0, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 1, 0, 0], 'input_dnn_all': [[1, {'ch_out': [96, 0], 'ch_in': [16, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 1.5206047588836997}, {'pe_array': 0, 'input_lp_order_gb': [5, 1, 0, 3, 4, 2], 'input_lp_order_dram': [4, 5, 3, 2, 1, 0], 'tiling_choices': [0, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 1, 0], 'input_dnn_all': [[1, {'ch_out': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1]], 'edp': 1.2650647320236836}, {'pe_array': 3, 'input_lp_order_gb': [2, 1, 4, 0, 3, 5, 6], 'input_lp_order_dram': [4, 2, 1, 6, 3, 0, 5], 'tiling_choices': [1, 1, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 0, 1, 1], 'input_dnn_all': [[1, {'ch_out': [32, 0], 'ch_in': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 2.3265534032216064}, {'pe_array': 1, 'input_lp_order_gb': [6, 1, 4, 3, 2, 5, 0], 'input_lp_order_dram': [0, 4, 5, 3, 6, 2, 1], 'tiling_choices': [0, 0, 0, 0, 0, 0, 0], 'tiling_choices_order': [0, 1, 1, 1, 1, 1, 0], 'input_dnn_all': [[1, {'ch_out': [96, 0], 'ch_in': [32, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 2.5307570850915955}, {'pe_array': 1, 'input_lp_order_gb': [5, 1, 4, 3, 2, 0], 'input_lp_order_dram': [5, 2, 1, 0, 4, 3], 'tiling_choices': [1, 1, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 1, 0], 'input_dnn_all': [[1, {'ch_out': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [3, 0], 'col_kernel': [3, 0]}, 1, 1]], 'edp': 1.5279658258707256}, {'pe_array': 3, 'input_lp_order_gb': [6, 2, 1, 5, 4, 0, 3], 'input_lp_order_dram': [0, 3, 5, 2, 1, 6, 4], 'tiling_choices': [0, 0, 0, 1, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 0, 0, 1], 'input_dnn_all': [[1, {'ch_out': [64, 0], 'ch_in': [96, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 6.831503909899486}, {'pe_array': 1, 'input_lp_order_gb': [3, 2, 4, 5, 6, 1, 0], 'input_lp_order_dram': [6, 4, 0, 2, 5, 3, 1], 'tiling_choices': [0, 0, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 1, 0, 1], 'input_dnn_all': [[1, {'ch_out': [384, 0], 'ch_in': [64, 0], 'batch': [1, 0], 'col_out': [32, 0], 'row_out': [32, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 23.20951193389497}, {'pe_array': 1, 'input_lp_order_gb': [5, 2, 0, 4, 1, 3], 'input_lp_order_dram': [5, 4, 0, 3, 1, 2], 'tiling_choices': [3, 1, 0, 0, 0, 0], 'tiling_choices_order': [0, 1, 1, 1, 1, 1], 'input_dnn_all': [[2, {'ch_out': [384, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1]], 'edp': 7.645732517183494}, {'pe_array': 0, 'input_lp_order_gb': [2, 4, 0, 5, 6, 1, 3], 'input_lp_order_dram': [4, 3, 0, 1, 6, 5, 2], 'tiling_choices': [0, 1, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 0, 0, 1], 'input_dnn_all': [[1, {'ch_out': [112, 0], 'ch_in': [384, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 10.76036507283751}, {'pe_array': 3, 'input_lp_order_gb': [1, 6, 2, 4, 5, 0, 3], 'input_lp_order_dram': [6, 2, 4, 0, 3, 1, 5], 'tiling_choices': [0, 0, 2, 2, 0, 0, 0], 'tiling_choices_order': [1, 1, 0, 1, 0, 0, 1], 'input_dnn_all': [[1, {'ch_out': [112, 0], 'ch_in': [112, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 1.8772375496215794}, {'pe_array': 1, 'input_lp_order_gb': [5, 3, 1, 4, 2, 0], 'input_lp_order_dram': [5, 3, 0, 4, 2, 1], 'tiling_choices': [1, 1, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 1, 0], 'input_dnn_all': [[1, {'ch_out': [112, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1]], 'edp': 0.8098605796246192}, {'pe_array': 3, 'input_lp_order_gb': [4, 1, 5, 2, 0, 3, 6], 'input_lp_order_dram': [4, 3, 2, 0, 1, 6, 5], 'tiling_choices': [0, 0, 1, 2, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 0, 1, 1], 'input_dnn_all': [[1, {'ch_out': [112, 0], 'ch_in': [112, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 1.6777002827994465}, {'pe_array': 2, 'input_lp_order_gb': [6, 2, 4, 5, 1, 0, 3], 'input_lp_order_dram': [3, 5, 2, 1, 6, 0, 4], 'tiling_choices': [0, 0, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 0, 1, 1], 'input_dnn_all': [[1, {'ch_out': [672, 0], 'ch_in': [112, 0], 'batch': [1, 0], 'col_out': [16, 0], 'row_out': [16, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 14.019746742811991}, {'pe_array': 0, 'input_lp_order_gb': [5, 1, 2, 3, 0, 4], 'input_lp_order_dram': [4, 1, 5, 3, 0, 2], 'tiling_choices': [0, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 1, 1], 'input_dnn_all': [[2, {'ch_out': [672, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1]], 'edp': 2.056773372416008}, {'pe_array': 3, 'input_lp_order_gb': [1, 4, 0, 6, 3, 2, 5], 'input_lp_order_dram': [1, 5, 2, 3, 6, 4, 0], 'tiling_choices': [0, 0, 0, 1, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 0, 0, 0], 'input_dnn_all': [[1, {'ch_out': [184, 0], 'ch_in': [672, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 4.736287610450802}, {'pe_array': 2, 'input_lp_order_gb': [6, 2, 5, 3, 0, 1, 4], 'input_lp_order_dram': [3, 4, 0, 1, 6, 5, 2], 'tiling_choices': [0, 3, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 0, 0, 0], 'input_dnn_all': [[1, {'ch_out': [1104, 0], 'ch_in': [184, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 10.43826899403654}, {'pe_array': 3, 'input_lp_order_gb': [5, 3, 2, 1, 4, 0], 'input_lp_order_dram': [1, 3, 4, 5, 2, 0], 'tiling_choices': [1, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 1, 1], 'input_dnn_all': [[1, {'ch_out': [1104, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1]], 'edp': 3.120187668825599}, {'pe_array': 3, 'input_lp_order_gb': [1, 2, 6, 4, 5, 3, 0], 'input_lp_order_dram': [1, 0, 2, 4, 6, 3, 5], 'tiling_choices': [0, 0, 1, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 0, 0, 0], 'input_dnn_all': [[1, {'ch_out': [184, 0], 'ch_in': [1104, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 14.479974317768532}, {'pe_array': 2, 'input_lp_order_gb': [4, 6, 3, 5, 2, 1, 0], 'input_lp_order_dram': [0, 5, 4, 3, 6, 2, 1], 'tiling_choices': [0, 3, 0, 1, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 0, 1, 0, 1], 'input_dnn_all': [[1, {'ch_out': [1104, 0], 'ch_in': [184, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 10.50988765713924}, {'pe_array': 3, 'input_lp_order_gb': [1, 3, 5, 2, 0, 4], 'input_lp_order_dram': [4, 5, 3, 0, 2, 1], 'tiling_choices': [5, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 1, 1], 'input_dnn_all': [[1, {'ch_out': [1104, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [5, 0], 'col_kernel': [5, 0]}, 1, 1]], 'edp': 2.999984757644801}, {'pe_array': 3, 'input_lp_order_gb': [1, 5, 3, 6, 0, 4, 2], 'input_lp_order_dram': [2, 1, 0, 5, 6, 3, 4], 'tiling_choices': [4, 0, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 1, 1, 0, 0], 'input_dnn_all': [[1, {'ch_out': [352, 0], 'ch_in': [1104, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 37.75886887615728}, {'pe_array': 3, 'input_lp_order_gb': [4, 6, 1, 2, 0, 3, 5], 'input_lp_order_dram': [5, 0, 4, 6, 2, 1, 3], 'tiling_choices': [1, 0, 0, 0, 0, 0, 0], 'tiling_choices_order': [1, 1, 1, 0, 0, 1, 0], 'input_dnn_all': [[1, {'ch_out': [1504, 0], 'ch_in': [352, 0], 'batch': [1, 0], 'col_out': [8, 0], 'row_out': [8, 0], 'row_kernel': [1, 0], 'col_kernel': [1, 0]}, 0, 1]], 'edp': 35.819751964257726}]


layer=2
pe_array=opt_hw[layer]['pe_array']
pe_array_dim_choices=0
tiling_choices= opt_hw[layer]['tiling_choices']
tiling_choices_order=opt_hw[layer]['tiling_choices_order']


tiling1=fpga_tiling_generator(input_dnn,tmp_hw_spec)
tiling_space_1=tiling1.tiling_space_partition(pe_array,layer,pe_array_dim_choices)
print(tiling_space_1)
tmp_hw_spec_list=resource_allocator(input_dnn,tmp_hw_spec)
tiling_string=tiling1.tiling_translation(layer,pe_array,pe_array_dim_choices,tiling_choices,tiling_choices_order)
print(tmp_hw_spec_list[0][layer])
print(tmp_hw_spec_list[1][layer])
print(tiling_string)
exit()
score=[]
max_trials=500
out_of_limit_num=0
for _ in range(max_trials):
    layer=randint(0,9)
    #pick a pe array
    pe_array=randint(0,3)
    #complete the rest of the lp_order: local buffer(rf), global buffer(gb), dram 
    #input_lp_order_rf=list(range(7))
    #shuffle(input_lp_order_rf)
    if input_dnn[layer][2]==0 or input_dnn[layer][2]==2:
        input_lp_order_gb=list(range(7))
        shuffle(input_lp_order_gb)
        input_lp_order_dram=list(range(7))
        shuffle(input_lp_order_dram)
    elif input_dnn[layer][2]==1:
        input_lp_order_gb=list(range(6))
        shuffle(input_lp_order_gb)
        input_lp_order_dram=list(range(6))
        shuffle(input_lp_order_dram)
    #translate the lp_order to string format
    lp_order_string=dram_invariant_looporder(pe_array,input_lp_order_dram, input_lp_order_gb,input_dnn[layer][2])

    #choose the applicable tiling space
    #!!!ATTENTION HERE!!!
    #
    #1. you need to specify the pe_array_dim_choices style, to favor different trade off sizes among each pe dimension;
    #           currently it is under 10, i.e. 0-9
    pe_array_dim_choices=randint(0,max_dim_choices-1)
    tiling_space_1=tiling1.tiling_space_partition(pe_array,layer,pe_array_dim_choices) 
    #print(len(tiling_space_1))
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
    print('estimating....')
    p_tmp_hw_spec={\
        'gb_vol':tmp_hw_spec_list[0][layer], \
        'rf_vol':512, \
        'num_pe':tmp_hw_spec_list[1][layer], \
        'num_rf':tmp_hw_spec_list[1][layer]\
    }



    #ATTENTION first element returned from life_eval is not a value, instead a tuple containing energy and latency
    if life_eval(tiling_string,1,p_tmp_hw_spec,input_dnn[layer][2],group_num=input_dnn[layer][3],df_order=lp_order_string)[1]:
        score.append(life_eval(tiling_string,1,p_tmp_hw_spec,input_dnn[layer][2],group_num=input_dnn[layer][3],df_order=lp_order_string)[0])
        print("current score: ", score[-1], 'Best score: ', sorted(score, reverse=True)[0])
    else:
        print('out of limit')
        out_of_limit_num +=1
print(out_of_limit_num/max_trials)








