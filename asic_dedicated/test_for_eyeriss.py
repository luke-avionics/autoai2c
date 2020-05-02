import argparse
import os
import sys
import math
import copy
import _thread

#in general df_order read from bottom to top from left to right, data passed out or in after refresh

#rf level dedicate the rf volume AND how many tims of resue of elements in noc level before being kicked out
#so even though some rf elements are serially (temprorily stored) because of the constraints of rf size they still as a
#whole dedicate the reuse of items in noc
sys.path.append('.')
sys.path.append('..')
#from simulator_support_dse import *
from simulator_eyeriss_scaletoli_debug import *
from cnn_load import *


def gb_out_consumption(df_order, df_config_dict):
    consumption=1    
    for i in range(len(df_order)):
        if df_order[i] =="ref_gb_out":
            return consumption
        elif "ch_out" in df_order[i] or "batch" in df_order[i] or "col_out" in df_order[i] or "row_out" in df_order[i] :
            consumption*=df_config_dict[df_order[i]]
    return consumption



def gb_in_consumption(df_order, df_config_dict, stride):
    consumption=1    
    for i in range(len(df_order)):
        if df_order[i] =="ref_gb_in":
            return consumption
        elif "col_out" in df_order[i] or  "row_out" in df_order[i] or "ch_in" in df_order[i] or "batch" in df_order[i]:
            consumption*=df_config_dict[df_order[i]]
    consumption*= stride
    return consumption


def gb_we_consumption(df_order, df_config_dict):
    consumption=1    
    for i in range(len(df_order)):
        if df_order[i] =="ref_gb_we":
            return consumption
        elif "kernel" in df_order[i] or "batch" in df_order[i] or "ch" in df_order[i]:
            consumption*=df_config_dict[df_order[i]]
    return consumption



def ref_location_optimization(df_order, df_config_dict,mode):
    if "ref_gb_out" not in df_order:
        df_order=["ref_gb_out"]+df_order
        df_order=["ref_gb_in"]+df_order
        df_order=["ref_gb_we"]+df_order

    if "ref_rf_out" not in df_order:
        df_order=["ref_rf_out"]+df_order
        df_order=["ref_rf_in"]+df_order
        df_order=["ref_rf_we"]+df_order
    if mode==0:
        #gb out
        ref_rf_out_idx=df_order.index('ref_gb_out')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('out_dram' in df_order[i] or 'batch_dram' in df_order[i]) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'dram' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_gb_out')
        del df_order[ref_rf_out_idx]

        #gb kernel
        ref_rf_out_idx=df_order.index('ref_gb_we')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('kernel_dram' in df_order[i] or 'ch_out_dram' in df_order[i] or 'ch_in_dram' in df_order[i] or 'batch_dram' in df_order[i] ) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'dram' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_gb_we')
        del df_order[ref_rf_out_idx]

        #gb input
        ref_rf_out_idx=df_order.index('ref_gb_in')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('kernel_dram' in df_order[i] or 'col_out_dram' in df_order[i] or 'row_out_dram' in df_order[i] or 'ch_in_dram' in df_order[i] or 'batch_dram' in df_order[i]) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'dram' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_gb_in')
        del df_order[ref_rf_out_idx]



        #rf out
        ref_rf_out_idx=df_order.index('ref_rf_out')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('out_gb' in df_order[i] or 'batch_gb' in df_order[i]) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'gb' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_rf_out')
        del df_order[ref_rf_out_idx]

        #rf kernel
        ref_rf_out_idx=df_order.index('ref_rf_we')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('kernel_gb' in df_order[i] or 'ch_out_gb' in df_order[i] or 'ch_in_gb' in df_order[i] or 'batch_gb' in df_order[i] ) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'gb' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_rf_we')
        del df_order[ref_rf_out_idx]

        #rf input
        ref_rf_out_idx=df_order.index('ref_rf_in')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('kernel_gb' in df_order[i] or 'col_out_gb' in df_order[i] or 'row_out_gb' in df_order[i] or 'ch_in_gb' in df_order[i] or 'batch_gb' in df_order[i]) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'gb' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_rf_in')
        del df_order[ref_rf_out_idx]
    elif mode==1:
        #gb out
        ref_rf_out_idx=df_order.index('ref_gb_out')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('out_dram' in df_order[i] or 'batch_dram' in df_order[i]) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'dram' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_gb_out')
        del df_order[ref_rf_out_idx]

        #gb kernel
        ref_rf_out_idx=df_order.index('ref_gb_we')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('kernel_dram' in df_order[i] or 'ch_out_dram' in df_order[i]  or 'batch_dram' in df_order[i] ) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'dram' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_gb_we')
        del df_order[ref_rf_out_idx]

        #gb input
        ref_rf_out_idx=df_order.index('ref_gb_in')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('kernel_dram' in df_order[i] or 'col_out_dram' in df_order[i] or 'row_out_dram' in df_order[i] or 'ch_out_dram' in df_order[i] or 'batch_dram' in df_order[i]) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'dram' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_gb_in')
        del df_order[ref_rf_out_idx]



        #rf out
        ref_rf_out_idx=df_order.index('ref_rf_out')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('out_gb' in df_order[i] or 'batch_gb' in df_order[i]) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'gb' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_rf_out')
        del df_order[ref_rf_out_idx]

        #rf kernel
        ref_rf_out_idx=df_order.index('ref_rf_we')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('kernel_gb' in df_order[i] or 'ch_out_gb' in df_order[i] or 'batch_gb' in df_order[i] ) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'gb' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_rf_we')
        del df_order[ref_rf_out_idx]

        #rf input
        ref_rf_out_idx=df_order.index('ref_rf_in')
        first_gb_out_idx=ref_rf_out_idx
        for i in range(len(df_order)):
            if(('kernel_gb' in df_order[i] or 'col_out_gb' in df_order[i] or 'row_out_gb' in df_order[i] or 'ch_out_gb' in df_order[i] or 'batch_gb' in df_order[i]) and df_config_dict[df_order[i]]!=1):
                first_gb_out_idx=i
                break
            elif 'gb' in df_order[i]:
                first_gb_out_idx=i
        df_order.insert(first_gb_out_idx,'ref_rf_in')
        del df_order[ref_rf_out_idx]
    return df_order
df_order=['row_out_noc', 'col_out_noc', 'ch_out_noc', 'batch_gb',  'col_out_gb', 'ch_out_gb', 'row_out_gb', 'row_kernel_gb', 'col_kernel_gb', 'col_kernel_dram', 'ch_out_dram',  'row_kernel_dram', 'col_out_dram', 'batch_dram', 'row_out_dram']
df_dict={'ch_out_noc': 16, 'col_out_noc': 1, 'row_out_noc': 8, 'ch_in_gb': 128, 'ch_out_gb': 1, 'col_kernel_gb': 1, 'row_kernel_gb': 3, 'col_out_gb': 1, 'row_out_gb': 1, 'batch_gb': 1, 'ch_in_dram': 1, 'ch_out_dram': 16, 'col_out_dram': 56, 'row_out_dram': 7, 'col_kernel_dram': 3, 'row_kernel_dram': 1, 'batch_dram': 1}
print(df_order)
print(ref_location_optimization(df_order,df_dict,1))
exit()
def memory_consumption(df_order, df_config_dict,stride):
    df_order,df_config_dict=copy.deepcopy(df_order),copy.deepcopy(df_config_dict)
    #we consumption
    we_df_order=[]
    we_consumption=16
    for i in df_order:
        if not 'rf' in i:
            break
        if ('ch' in i or 'kernel' in i or 'we' in i):
            we_df_order.append(i)
    ref_we_rf=we_df_order.index('ref_rf_we')
    we_df_order=we_df_order[0:ref_we_rf]
    for i in we_df_order:
        we_consumption*=df_config_dict[i]

    #out consumption
    out_df_order=[]
    out_consumption=16
    for i in df_order:
        if not 'rf' in i:
            break
        if ('out' in i):
            out_df_order.append(i)
    ref_out_rf=out_df_order.index('ref_rf_out')
    out_df_order=out_df_order[0:ref_out_rf]
    for i in out_df_order:
        out_consumption*=df_config_dict[i]



    #in consumption
    in_df_order=[]
    in_consumption=16
    for i in df_order:
        if not 'rf' in i:
            break
        if ('in' in i or 'row' in i or 'col' in i) and (df_config_dict[i] !=1):
            in_df_order.append(i)
    ref_in_rf=in_df_order.index('ref_rf_in')
    in_df_order=in_df_order[0:ref_in_rf]
    #decide to follow kernel or out
    row_num=1
    col_num=1
    if 'row_out_rf' in in_df_order:
        row_num=df_config_dict['row_out_rf']
        row_num=row_num*stride
        if 'row_kernel_rf' in in_df_order:
            row_num+=df_config_dict['row_kernel_rf']
    elif 'row_kernel_rf' in in_df_order:
        row_num=df_config_dict['row_kernel_rf']
        row_num=row_num
    if 'col_out_rf' in in_df_order:
        col_num=df_config_dict['col_out_rf']
        col_num=col_num*stride
        if 'col_kernel_rf' in in_df_order:
            col_num+=df_config_dict['col_kernel_rf']
    elif 'col_kernel_rf' in in_df_order:
        col_num=df_config_dict['col_kernel_rf']
        col_num=col_num
    in_consumption=in_consumption*col_num*row_num
    for i in in_df_order:
        if 'ch_in' in i:
            in_consumption*=df_config_dict[i]
    return [we_consumption,out_consumption,in_consumption]



def gb_memory_consumption(df_order, df_config_dict,stride,rf_consumption):
    df_order,df_config_dict=copy.deepcopy(df_order),copy.deepcopy(df_config_dict)
    tmp_df_order=[]
    noc_order=[]
    for i in df_order:
        if 'noc' in i:
            noc_order.append(i)
    noc_we=1
    noc_out=1
    noc_in=1
    for i in noc_order:
        if 'kernel' in i or 'ch' in i:
            noc_we*=df_config_dict[i]
            if 'in' in i:
                noc_in*=df_config_dict[i]
        if 'out' in i:
            noc_out*=df_config_dict[i]
            if 'row' in i or 'col' in i:
                noc_in*=df_config_dict[i]

    for i in df_order:
        if 'gb' in i:
            tmp_df_order.append(i)
    df_order=tmp_df_order
    #we consumption
    we_df_order=[]
    we_consumption=1
    for i in df_order:
        if ('ch' in i or 'kernel' in i or 'we' in i):
            we_df_order.append(i)
    ref_we_gb=we_df_order.index('ref_gb_we')
    we_df_order=we_df_order[0:ref_we_gb]
    for i in we_df_order:
        we_consumption*=df_config_dict[i]

    #out consumption
    out_df_order=[]
    out_consumption=1
    for i in df_order:
        if ('out' in i):
            out_df_order.append(i)
    ref_out_gb=out_df_order.index('ref_gb_out')
    out_df_order=out_df_order[0:ref_out_gb]
    for i in out_df_order:
        out_consumption*=df_config_dict[i]



    #in consumption
    in_df_order=[]
    in_consumption=1
    for i in df_order:
        if ('in' in i or 'row' in i or 'col' in i) and (df_config_dict[i] !=1):
            in_df_order.append(i)
    ref_in_gb=in_df_order.index('ref_gb_in')
    in_df_order=in_df_order[0:ref_in_gb]
    #decide to follow kernel or out
    row_num=1
    col_num=1
    if 'row_out_gb' in in_df_order:
        row_num=df_config_dict['row_out_gb']
        row_num=row_num*stride
        if 'row_kernel_gb' in in_df_order:
            row_num+=df_config_dict['row_kernel_gb']
    elif 'row_kernel_gb' in in_df_order:
        row_num=df_config_dict['row_kernel_gb']
        row_num=row_num
    if 'col_out_gb' in in_df_order:
        col_num=df_config_dict['col_out_gb']
        col_num=col_num*stride
        if 'col_kernel_gb' in in_df_order:
            col_num+=df_config_dict['col_kernel_gb']
    elif 'col_kernel_gb' in in_df_order:
        col_num=df_config_dict['col_kernel_gb']
        col_num=col_num
    in_consumption=in_consumption*col_num*row_num
    for i in in_df_order:
        if 'ch_in' in i:
            in_consumption*=df_config_dict[i]
    return [we_consumption*rf_consumption[0]*noc_we,out_consumption*rf_consumption[1]*noc_out,in_consumption*rf_consumption[2]*noc_in]


def examples_arch():
    return None

def init(dram_vol, dram_bw, gb_vol, gb_bw, noc_bw,
                       rf_vol, rf_bw, num_rf, num_adder, num_mul, num_pe,
                       bits_adder, e_adder, bits_mul, e_mul, freq_pe, cycles_add, cycles_mul,
                    #    bw_dram_to_gb, bw_gb_to_noc, bw_noc_to_rf, bw_rf_to_alu,
                       ebit_dram_to_gb, ebit_gb_to_noc, ebit_noc_to_rf, ebit_rf_to_alu,
                       e_dram_to_gb, e_gb_to_noc, e_noc_to_rf, e_rf_to_alu,
                       freq_dram, freq_gb, freq_noc, freq_rf,
                       t_dram_to_gb, t_gb_to_noc, t_noc_to_rf, t_rf_to_alu):
                       
    hw_config1 = plt_config1(dram_vol, dram_bw, gb_vol, gb_bw, noc_bw,
                       rf_vol, rf_bw, num_rf, num_adder, num_mul, num_pe,
                       bits_adder, e_adder, bits_mul, e_mul, freq_pe, cycles_add, cycles_mul,
                    #    bw_dram_to_gb, bw_gb_to_noc, bw_noc_to_rf, bw_rf_to_alu,
                       ebit_dram_to_gb, ebit_gb_to_noc, ebit_noc_to_rf, ebit_rf_to_alu,
                       e_dram_to_gb, e_gb_to_noc, e_noc_to_rf, e_rf_to_alu,
                       freq_dram, freq_gb, freq_noc, freq_rf,
                       t_dram_to_gb, t_gb_to_noc, t_noc_to_rf, t_rf_to_alu)
    return hw_config1




#---------------------------------------- below is what you can change ------------------------------------
def sample_energy(input_input_df_dict,input_stride,hw_spec,mode,input_df_order=None):
    df_order = ['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out','row_out_rf', 'ref_rf_in','batch_rf',\
            'ref_rf_we','col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc',\
            'ref_gb_we','ch_out_gb','ref_gb_in','ch_in_gb',\
            'ref_gb_out','col_out_dram', 'ch_out_dram', 'batch_dram']
    input_df_dict=copy.deepcopy(input_input_df_dict)

    if input_df_order:
        df_order=list(input_df_order)  
    #check if there is pre_defined ref_locs 
    pre_defined_ref=False
    for i in df_order:
        if 'ref' in i:
            pre_defined_ref=True
            break 
    if not pre_defined_ref:
        for i in range(len(df_order)):
            if 'gb' in df_order[i]:
                break
        df_order.insert(i,'ref_gb_in') 
        df_order.insert(i,'ref_gb_out')
        df_order.insert(i,'ref_gb_we')
        df_order.insert(0,'ref_rf_in')
        df_order.insert(0,'ref_rf_out')
        df_order.insert(0,'ref_rf_we')
    if mode==1:
        for i in df_order:
            if 'ch_in' in i:
                print('you picked the depth-wsie convolution, but there is input_channel in the loop definition, pls go check and remove it')
                raise

    df_config_dict=input_df_dict
    df_order=ref_location_optimization(df_order,df_config_dict,mode)
    #print('=========\n',df_order)
    all_refresh_locs = ['ref_gb_in','ref_gb_out','ref_gb_we','ref_rf_in','ref_rf_out','ref_rf_we']
    df_config_dict['ref_rf_out']=64
    df_config_dict['ref_rf_in']=16
    df_config_dict['ref_rf_we']=64
    df_config_dict['ref_gb_in']=df_config_dict['ref_gb_out']=df_config_dict['ref_gb_we']=64
    
    stride=input_stride
    # this part is filled in by the "smart" domain specific experts (Yang Zhao)
    dram_vol = float('inf') # the dram volume (bits)
    #gb_vol=2785280
    #rf_vol =4160*1.23*55
    #shiDianNao
    #rf_vol=92*8
    #gb_vol=2048000
    #dram_vol=1651200
 
    #num_rf = 168 # the number of RF  # set by Yang 7/12/2019
    #num_pe = 168 # the number of PE  # set by Yang 7/12/2019
    #shiDianNao
    #num_rf = 64
    #num_pe = 64
    gb_vol=hw_spec['gb_vol']
    rf_vol=hw_spec['rf_vol']
    num_rf=hw_spec['num_rf']
    num_pe=hw_spec['num_pe']

    num_adder = 4 # the number of adders in each PE  # set by Yang 7/12/2019, 16-bit adder = 4 * 4-bit adder ? actually
    num_mul = 16 # the number of multipliers in each PE  # set by Yang 7/12/2019, 16-bit mul = 16 * 4-bit adder ?
    bits_adder = 4 # the precision for the adder, don't change this one
    bits_mul = 4 # the precision for the multiplier, don't change this one
    #65 nm
#    e_adder = 1.0/68 # unit energy for each adder operation
#    e_mul = 1.0/17 # unit energy for each multiplier operation  # set by Yang 7/12/2019, 1 MAC = 1 = 4 * E_add + 16 * E_mul
    #28 nm
    e_adder = 1.0/68*1.6
    e_mul=1.0/17*1.6
    freq_pe = 250e6 # frequency for the PE  # set by Yang 7/12/2019
    cycles_add = 1.0 # cycles needed for each adder operation  # set by Yang 7/12/2019
    cycles_mul = 2.0 # cycles needed for each multiplier operation  # set by Yang 7/12/2019

    dram_bw = 64 # the bitwidth for dram (bits)  # set by Yang 7/12/2019, not sure
    gb_bw = 64 # the bitwidth for global buffer (bits)
    noc_bw = 144*num_pe # the bitwidth for noc (bits)
    #tpu ws
    noc_bw = 144*num_pe # the bitwidth for noc (bits)
    rf_bw = 64 # the bitwidth for rf (bits)


    # communication for dram->gb

    ebit_dram_to_gb = 200/16 # energy/bit for the dram->gb data communication
    e_dram_to_gb = 0 # setup energy for the dram->gb data communication
    t_dram_to_gb = 0 #setup time for dram->gb
    
    # gb->noc

    ebit_gb_to_noc = 3.78/8/2
    e_gb_to_noc = 0
    t_gb_to_noc = 0
    # noc->rf

    ebit_noc_to_rf = 0.89/8*2/3
    e_noc_to_rf = 0
    t_noc_to_rf = 0
    # rf->alu

    ebit_rf_to_alu = 0.89/8/2
    e_rf_to_alu = 0
    t_rf_to_alu = 0

#    [we_consumption_rf,out_consumption_rf,in_consumption_rf]=memory_consumption(df_order,df_config_dict,stride)
#    total_rf_consumption=sum([we_consumption_rf,out_consumption_rf,in_consumption_rf])
#    we_rf_size,out_rf_size,in_rf_size=rf_vol*we_consumption_rf/total_rf_consumption,rf_vol*out_consumption_rf/total_rf_consumption,rf_vol*in_consumption_rf/total_rf_consumption
#    we_rf_unit,out_rf_unit,in_rf_unit=we_rf_size*0.03/2048,out_rf_size*0.03/2048,in_rf_size*0.03/2048
#    ebit_rf_to_alu = (we_rf_unit*we_consumption_rf+out_rf_unit*out_consumption_rf+in_rf_unit*in_consumption_rf)/total_rf_consumption

#    [we_consumption_gb, out_consumption_gb, in_consumption_gb]=gb_memory_consumption(df_order,df_config_dict,stride,[we_consumption_rf,out_consumption_rf,in_consumption_rf])
#    total_gb_consumption=sum([we_consumption_gb, out_consumption_gb, in_consumption_gb])
#    we_gb_size,out_gb_size,in_gb_size=gb_vol*we_consumption_gb/total_gb_consumption,gb_vol*out_consumption_gb/total_gb_consumption,gb_vol*in_consumption_gb/total_gb_consumption
#    we_gb_unit,out_gb_unit,in_gb_unit=we_gb_size*we_gb_size*-4.8862e-14+we_gb_size*5.9817e-7+0.2459,\
#                                      out_gb_size*out_gb_size*-4.8862e-14+out_gb_size*5.9817e-7+0.2459,\
#                                      in_gb_size*in_gb_size*-4.8862e-14+in_gb_size*5.9817e-7+0.2459
#    ebit_gb_to_noc=(we_gb_unit*we_consumption_gb+out_gb_unit*out_consumption_gb+in_gb_unit*in_consumption_gb)/total_gb_consumption

    # working frequency for dram, gb, noc, rf
    freq_dram = 90e6
    freq_gb = 250e6
    freq_noc = 250e6
    freq_rf = 250e6
    
    
    hw_config1 = init(dram_vol, dram_bw, gb_vol, gb_bw, noc_bw,
                       rf_vol, rf_bw, num_rf, num_adder, num_mul, num_pe,
                       bits_adder, e_adder, bits_mul, e_mul, freq_pe, cycles_add, cycles_mul,
                    #    bw_dram_to_gb, bw_gb_to_noc, bw_noc_to_rf, bw_rf_to_alu,
                       ebit_dram_to_gb, ebit_gb_to_noc, ebit_noc_to_rf, ebit_rf_to_alu,
                       e_dram_to_gb, e_gb_to_noc, e_noc_to_rf, e_rf_to_alu,
                       freq_dram, freq_gb, freq_noc, freq_rf,
                       t_dram_to_gb, t_gb_to_noc, t_noc_to_rf, t_rf_to_alu)
    

    
    bits_weight = 16
    bits_activation = 16

    bw_gb_to_noc_dict = {'in':64, 'out':64, 'we':64}
    bw_rf_to_alu_dict = {'in':16, 'out':16, 'we':16}
    if mode==0:
        Energy_breakdown , opr_conv, opr_rf, opr_gb, num_active_pes = hw_config1.conv_df(stride, df_order, df_config_dict, bits_activation, bits_weight, bw_gb_to_noc_dict, bw_rf_to_alu_dict)
    elif mode==1:
        Energy_breakdown, opr_conv, opr_rf, opr_gb, num_active_pes = hw_config1.dw_conv_df(stride, df_order,
                                                                                        df_config_dict, bits_activation,
                                                                                        bits_weight, bw_gb_to_noc_dict,
                                                                                        bw_rf_to_alu_dict)
    # print (Energy_breakdown)
    
    E_comp = Energy_breakdown[0][0]
    #print('comp energy', E_comp)
    E_dram_to_gb = Energy_breakdown[1][0]
    E_gb_to_noc = Energy_breakdown[2][0]
    #print('comp_ene',E_comp)
    #print('sram_access', E_gb_to_noc/ebit_gb_to_noc/1e6/8)
    #print('dram_access', E_dram_to_gb/ebit_dram_to_gb/1e6/8/1.5)
    E_noc_to_rf = Energy_breakdown[3][0]
    E_rf_to_alu = Energy_breakdown[4][0]
    E_sum_up = E_comp + E_dram_to_gb + E_gb_to_noc + E_noc_to_rf + E_rf_to_alu
    # print ('latency(s): '+str(opr_conv.time))
    # print ('energy: '+str(opr_conv.energy))
    # print ('energy (sum up): '+ str(E_sum_up))
    # print ('\n')
    # print ('energy breakdown')
    # print ('E_comp: '+ str(E_comp/opr_conv.energy))    E_comp=8*10^8 (pj? .... \op    ?\op)
    # print ('E_dram_to_gb: ' +str(E_dram_to_gb/opr_conv.energy))
    # print ('E_gb_to_noc: '+ str(E_gb_to_noc/opr_conv.energy))
    # print ('E_noc_to_rf: '+ str(E_noc_to_rf/opr_conv.energy))
    # print ('E_rf_to_alu: ' + str(E_rf_to_alu/opr_conv.energy))
    # print ('\n')
    # print ('resource usage')
    # print ('gb: '+ str(opr_gb.plt.leaf_find_renum(opr_gb.consume_list[0], 'volume').val)+'/'+str(gb_vol))
    # print ('rf: '+ str(opr_rf.plt.leaf_find_renum(opr_rf.consume_list[0], 'volume').val)+'/'+str(rf_vol))
    #---------------help me check end ----------------------
#    print({'E_dram_to_gb':E_dram_to_gb/opr_conv.energy,\
#                                           'E_gb_to_noc':E_gb_to_noc/opr_conv.energy,\
#                                           'E_noc_to_rf':E_noc_to_rf/opr_conv.energy,\
#                                           'E_rf_to_alu':E_rf_to_alu/opr_conv.energy})
    return opr_conv.energy,opr_conv.time,{'E_dram_to_gb':E_dram_to_gb/opr_conv.energy,\
                                           'E_gb_to_noc':E_gb_to_noc/opr_conv.energy,\
                                           'E_noc_to_rf':E_noc_to_rf/opr_conv.energy,\
                                           'E_rf_to_alu':E_rf_to_alu/opr_conv.energy}























