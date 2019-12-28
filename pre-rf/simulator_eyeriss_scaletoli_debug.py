import argparse
import os
import sys
import math
import copy
import pickle
import _thread

# from inso_utils import *
# from cnn_load import *

__author__ = "insomnia Pengfei Xu"


def div_up(a, b):
    return math.ceil(float(a) / b)


def div_down(a, b):
    return math.floor(float(a) / b)


# resource number and resource name
class renum(object):
    def __init__(self, name, val):
        self.val = val
        self.name = name
        self.type = 'resource'


# tree-based platform definition:
# for combined platform: renum_list is NULL, this node has son nodes;
# for basic platform: son_plt_list is NULL, this node has no son nodes (it is a leaf node);
class plt(object):
    def __init__(self, name, renum_list, son_plt_list, repeat):
        self.renum_list = renum_list
        self.son_plt_list = son_plt_list
        self.name = name
        self.repeat = repeat
        self.all_leaf = self.get_leaf()
        self.type = 'platform'

    def get_leaf(self):
        if (self.son_plt_list == []):  # leaf_node
            return [self]
        elif (self.renum_list == []):  # not leaf node
            son_leaf = []
            for son in self.son_plt_list:
                son_leaf += son.get_leaf()
            return son_leaf
        else:
            print(self.name)
            print('in get_laef function')
            print('either son_plt_list or renum_list need to be []')
            sys.exit(-1)

    def find_leaf(self, leaf_list, leaf_name):
        # actually self is not used in the function, it is equivelant with def find_leaf(leaf_list,renum_name)
        for leaf in leaf_list:
            if (leaf.name == leaf_name):
                return leaf
        # print (leaf_name)
        # print ('in find_leaf function, the name is not found')
        return -1

    def leaf_find_renum(self, leaf, renum_name):
        # actually self is not used in the function, it is equivelant with def leaf_find_renum(leaf,renum_name)
        if (leaf.renum_list == []):  # not leaf node
            print(leaf.name)
            print('in find_renum function')
            print('this one is not a leaf node')
            sys.exit(-1)
        elif (leaf.son_plt_list == []):  # leaf node
            for renum_item in leaf.renum_list:
                if (renum_item.name == renum_name):
                    return renum_item
            # print (renum_name)
            # print ('in find_renum function, the name is not found')
            return -1
        else:
            print(leaf.name)
            print('in find_renum function')
            print('either son_plt_list or renum_list need to be []')
            sys.exit(-1)

    def find_renum(self, leaf_name, renum_name):
        leaf = self.find_leaf(self.all_leaf, leaf_name)
        return self.leaf_find_renum(leaf, renum_name)


class opr(object):
    def __init__(self, plt, consume_list, energy, time):
        if (len(plt.all_leaf) < len(consume_list)):
            print('number of leaf nodes should be greater than the length of the resource consumption list')
            sys.exit(-1)
        success = True
        for i, leaf_consumption in enumerate(consume_list):
            leaf = plt.find_leaf(plt.all_leaf, leaf_consumption.name)
            if leaf_consumption.repeat > leaf.repeat:

                raise Exception('in {} the resource consumption {} exceeeds the platform limit {}'. \
                                format(leaf_consumption.name, leaf_consumption.repeat, leaf.repeat))
                # print('in {} the resource consumption {} exceeeds the platform limit {}'.\
                #     format(leaf_consumption.name, leaf_consumption.repeat, leaf.repeat))
                success = False
            for renum_name in ['num_adder', 'num_mul', 'volume', 'bitwidth']:
                if plt.leaf_find_renum(leaf, renum_name).val < plt.leaf_find_renum(leaf_consumption, renum_name).val:

                    raise Exception('in {}-{} the resource consumption {} exceeeds the platform limit {}'. \
                                    format(leaf_consumption.name, renum_name,
                                           plt.leaf_find_renum(leaf_consumption, renum_name).val,
                                           plt.leaf_find_renum(leaf, renum_name).val))
                    # print('in {}-{} the resource consumption {} exceeeds the platform limit {}'.\
                    #     format(leaf_consumption.name, renum_name, plt.leaf_find_renum(leaf_consumption, renum_name).val, plt.leaf_find_renum(leaf, renum_name).val))
                    success = False
        self.plt = plt if success else -4
        self.consume_list = consume_list if success else -4
        self.energy = energy if success else -4
        self.time = time if success else -4
        self.success = success


# consider pipeline stage
# def pipeline_merge(opr_list1, opr_list2, plt1):
#     len1 = len(opr_list1)
#     len2 = len(opr_list2)
#     return opr_list

def temp_merge(opr1, opr2, plt1):
    if isinstance(opr1, int) or isinstance(opr2, int) or opr1.plt == -4 or opr2.plt == -4:
        return -9
    energy = opr1.energy + opr2.energy
    time = opr1.time + opr2.time
    consume_list = []
    for i, leaf in enumerate(plt1.all_leaf):
        leaf1 = plt1.find_leaf(opr1.consume_list, leaf.name)
        leaf2 = plt1.find_leaf(opr2.consume_list, leaf.name)
        if leaf1 == -1 and leaf2 == -1:
            continue
        elif leaf1 == -1 and leaf2 != -1:
            consume_list.append(leaf2)
        elif leaf1 != -1 and leaf2 == -1:
            consume_list.append(leaf1)
        elif leaf1 != -1 and leaf2 != -1:
            repeat = max(leaf1.repeat, leaf2.repeat)
            renum_list = []
            for renum_name in ['num_adder', 'num_mul', 'volume', 'bitwidth']:
                renum_new = renum(name=renum_name, val=max(plt1.leaf_find_renum(leaf1, renum_name).val,
                                                           plt1.leaf_find_renum(leaf2, renum_name).val))
                renum_list.append(renum_new)
            new_leaf = plt(name=leaf.name, renum_list=renum_list, son_plt_list=[], repeat=repeat)
            consume_list.append(new_leaf)
        else:
            print('error in temp merge function')
            sys.exit(-1)
    # try:
    new_opr = opr(plt1, consume_list, energy, time)
    return new_opr
    # except:
    #     return -9


def spatial_merge(opr1, opr2, plt1):
    if isinstance(opr1, int) or isinstance(opr2, int) or opr1.plt == -4 or opr2.plt == -4:
        return -8
    energy = opr1.energy + opr2.energy
    time = max(opr1.time, opr2.time)
    consume_list = []
    for i, leaf in enumerate(plt1.all_leaf):
        leaf1 = plt1.find_leaf(opr1.consume_list, leaf.name)
        leaf2 = plt1.find_leaf(opr2.consume_list, leaf.name)
        if leaf1 == -1 and leaf2 == -1:
            continue
        elif leaf1 == -1 and leaf2 != -1:
            consume_list.append(leaf2)
        elif leaf1 != -1 and leaf2 == -1:
            consume_list.append(leaf1)
        elif leaf1 != -1 and leaf2 != -1:
            # repeat = max(leaf1.repeat, leaf2.repeat)
            renum_list_new = []
            flag = True
            for renum_name in ['num_adder', 'num_mul', 'volume', 'bitwidth']:
                val_new = plt1.leaf_find_renum(leaf1, renum_name).val + plt1.leaf_find_renum(leaf2, renum_name).val
                renum_new = renum(name=renum_name, val=val_new)
                renum_list_new.append(renum_new)
                if val_new > plt1.leaf_find_renum(leaf, renum_name).val:
                    flag = False
            if flag:
                repeat_new = max(leaf1.repeat, leaf2.repeat)
                new_leaf = plt(name=leaf.name, renum_list=renum_list_new, son_plt_list=[], repeat=repeat_new)
            else:
                repeat_new = leaf1.repeat + leaf2.repeat
                renum_list_new = []
                for renum_name in ['num_adder', 'num_mul', 'volume', 'bitwidth']:
                    val_new = max(plt1.leaf_find_renum(leaf1, renum_name).val,
                                  plt1.leaf_find_renum(leaf2, renum_name).val)
                    renum_new = renum(name=renum_name, val=val_new)
                    renum_list_new.append(renum_new)
                new_leaf = plt(name=leaf.name, renum_list=renum_list_new, son_plt_list=[], repeat=repeat_new)
            consume_list.append(new_leaf)
        else:
            print('error in temp merge function')
            sys.exit(-1)
    # try:
    new_opr = opr(plt1, consume_list, energy, time)
    return new_opr
    # except:
    #     return -8


def opr_sum(opr_list, plt1, option):
    if len(opr_list) == 0:
        new_opr = -1
    elif len(opr_list) == 1:
        new_opr = opr_list[0]
    else:
        new_opr = opr_list[0]
        for i in range(len(opr_list) - 1):
            if option == 'temp':
                new_opr = temp_merge(new_opr, opr_list[i + 1], plt1)
            elif option == 'spatial':
                new_opr = spatial_merge(new_opr, opr_list[i + 1], plt1)
            else:
                print('no such options')
                sys.exit(-1)
    return new_opr


# digital accelerator components
def def_mem(name, mem_volume, bitwidth, repeat):
    vol = renum('volume', mem_volume)
    bw = renum('bitwidth', bitwidth)
    num_adder = renum('num_adder', 0)
    num_mul = renum('num_mul', 0)
    mem = plt(name, [num_adder, num_mul, vol, bw], [], repeat)
    return mem


# 4-bit adder and 4-bit multiplier
def def_pe(name, num_adder, num_mul, repeat):
    num_a = renum('num_adder', num_adder)
    num_m = renum('num_mul', num_mul)
    vol = renum('volume', 0)
    bw = renum('bitwidth', 0)
    pe = plt(name, [num_a, num_m, vol, bw], [], repeat)
    return pe


# digital accelerator template 1 (based on eyeriss)
# the hw platform has memory of 4 hierarchies, computation resources
# memory: dram, gb, noc, rf, gb can be allocted as input, output, and weights
# computation resource: multipliers and adders
# for each memory hierarchy, read and write need to be configured as time division
class plt_config1(object):
    def __init__(self, dram_vol, dram_bw, gb_vol, gb_bw, noc_bw,
                 rf_vol, rf_bw, num_rf, num_adder, num_mul, num_pe,
                 bits_adder, e_adder, bits_mul, e_mul, freq_pe, cycles_add, cycles_mul,
                 #    bw_dram_to_gb, bw_gb_to_noc, bw_noc_to_rf, bw_rf_to_alu,
                 ebit_dram_to_gb, ebit_gb_to_noc, ebit_noc_to_rf, ebit_rf_to_alu,
                 e_dram_to_gb, e_gb_to_noc, e_noc_to_rf, e_rf_to_alu,
                 freq_dram, freq_gb, freq_noc, freq_rf,
                 t_dram_to_gb, t_gb_to_noc, t_noc_to_rf, t_rf_to_alu):
        if num_rf != num_pe:
            raise Exception('num_rf: {} need to equal num_pe: {}'.format(num_rf, num_pe))
        dram = def_mem('dram', dram_vol, dram_bw, repeat=1)
        gb = def_mem('gb', gb_vol, gb_bw, repeat=1)
        noc = def_mem('noc', float('inf'), noc_bw, repeat=1)
        rf = def_mem('rf', rf_vol, rf_bw, repeat=num_rf)
        pe = def_pe('pe', num_adder, num_mul, repeat=num_pe)
        plt1 = plt('acc1', [], [pe, rf, noc, gb, dram], repeat=1)
        self.plt = plt1
        self.bits_adder = bits_adder  # set it to 4 bits
        self.bits_mul = bits_mul  # set it to 4 bits
        self.e_adder = e_adder
        self.e_mul = e_mul
        self.freq_pe = freq_pe
        self.cycles_add = cycles_add
        self.cycles_mul = cycles_mul
        # self.bw_dram_to_gb = bw_dram_to_gb
        # self.bw_gb_to_noc = bw_gb_to_noc
        # self.bw_noc_to_rf = bw_noc_to_rf
        # self.bw_rf_to_alu = bw_rf_to_alu
        self.ebit_dram_to_gb = ebit_dram_to_gb
        self.ebit_gb_to_noc = ebit_gb_to_noc
        self.ebit_noc_to_rf = ebit_noc_to_rf
        self.ebit_rf_to_alu = ebit_rf_to_alu
        self.e_dram_to_gb = e_dram_to_gb
        self.e_gb_to_noc = e_gb_to_noc
        self.e_noc_to_rf = e_noc_to_rf
        self.e_rf_to_alu = e_rf_to_alu
        self.freq_dram = freq_dram
        self.freq_gb = freq_gb
        self.freq_noc = freq_noc
        self.freq_rf = freq_rf
        self.t_dram_to_gb = t_dram_to_gb
        self.t_gb_to_noc = t_gb_to_noc
        self.t_noc_to_rf = t_noc_to_rf
        self.t_rf_to_alu = t_rf_to_alu

    def add(self, bits):  # adder_precision = 4 bits
        num_adder = div_up(bits, self.bits_adder)
        energy = self.e_adder * num_adder
        time = float(self.cycles_add) / self.freq_pe
        add_consumption = def_pe('pe', num_adder=num_adder, num_mul=0, repeat=1)
        # try:
        add_opr = opr(self.plt, [add_consumption], energy, time)
        return add_opr
        # except:
        #     return -9

    def mul(self, bits1, bits2):
        a1 = div_up(bits1, self.bits_mul)
        a2 = div_up(bits2, self.bits_mul)
        num_mul = a1 * a2
        time = float(self.cycles_mul) / self.freq_pe
        energy = self.e_mul * num_mul
        mul_consumption = def_pe('pe', num_adder=0, num_mul=num_mul, repeat=1)
        # try:
        mul_opr = opr(self.plt, [mul_consumption], energy, time)
        return mul_opr
        # except:
        #     return -9

    def comm(self, bits, bitwidth, src_name, dst_name, e_setup, ebit, t_setup, freq_src, freq_dst):
        src_consumption = def_mem(name=src_name, mem_volume=bits, bitwidth=bitwidth, repeat=1)
        dst_consumption = def_mem(name=dst_name, mem_volume=bits, bitwidth=bitwidth, repeat=1)
        consumption_list = [src_consumption, dst_consumption] if dst_name != 'alu' else [src_consumption]
        energy = e_setup + bits * ebit
        time = t_setup + div_up(bits, bitwidth) / freq_src + div_up(bits, bitwidth) / freq_dst
        # try:
        comm_opr = opr(self.plt, consumption_list, energy, time)
        return comm_opr
        # except:
        #     return -9

    # holding data in the memory without refreshing it, here we think the GB and rf is always occpupied
    # we think this part has no energy cost
    def occupy_volume(self, bits, mem_name):
        mem_consumption = def_mem(name=mem_name, mem_volume=bits, bitwidth=0, repeat=1)
        consumption_list = [mem_consumption]
        energy = 0
        time = 0
        occupy_opr = opr(self.plt, consumption_list, energy, time)
        return occupy_opr

    def dram_to_gb(self, bits, bw_dram_to_gb):  # occupied bitwidth
        # print("dram_to_gb")
        return self.comm(bits, bw_dram_to_gb, 'dram', 'gb', self.e_dram_to_gb, self.ebit_dram_to_gb, self.t_dram_to_gb,
                         self.freq_dram, self.freq_gb)

    def gb_to_noc(self, bits, bw_gb_to_noc):
        # print("gb_to_noc")
        return self.comm(bits, bw_gb_to_noc, 'gb', 'noc', self.e_gb_to_noc, self.ebit_gb_to_noc, self.t_gb_to_noc,
                         self.freq_gb, self.freq_noc)

    def noc_to_rf(self, bits, bw_noc_to_rf):
        # print("noc_to_rf")
        return self.comm(bits, bw_noc_to_rf, 'noc', 'rf', self.e_noc_to_rf, self.ebit_noc_to_rf, self.t_noc_to_rf,
                         self.freq_noc, self.freq_rf)

    def rf_to_alu(self, bits, bw_rf_to_alu):
        # print("rf_to_alu")
        return self.comm(bits, bw_rf_to_alu, 'rf', 'alu', self.e_rf_to_alu, self.ebit_rf_to_alu, self.t_rf_to_alu,
                         self.freq_rf, float('inf'))

    # examples of df_config_dict: {'rf_out':8,'batch_rf':4,'ch_out_rf':16,'rf_in':8, ...,'ch_out':64}, bottom-up sequence
    # for key in loop orders, the value is the tiling factor, for key in refresh locations, the value is the bitwidth
    # noc level loops are in parallel
    def conv_df(self, stride, df_order_in, df_config_dict_in, bits_activation, bits_weight, bw_gb_to_noc_dict,
                bw_rf_to_alu_dict):
        all_dims = ['batch', 'ch_out', 'ch_in', 'row_out', 'col_out', 'row_kernel', 'col_kernel']
        all_lvls = ['dram', 'gb', 'noc', 'rf']
        all_orders = ['batch_dram', 'batch_gb', 'batch_noc', 'batch_rf',
                      'ch_out_dram', 'ch_out_gb', 'ch_out_noc', 'ch_out_rf',
                      'ch_in_dram', 'ch_in_gb', 'ch_in_noc', 'ch_in_rf',
                      'row_out_dram', 'row_out_gb', 'row_out_noc', 'row_out_rf',
                      'col_out_dram', 'col_out_gb', 'col_out_noc', 'col_out_rf',
                      'row_kernel_dram', 'row_kernel_gb', 'row_kernel_noc', 'row_kernel_rf',
                      'col_kernel_dram', 'col_kernel_gb', 'col_kernel_noc', 'col_kernel_rf']
        out_related = ['batch', 'ch_out', 'row_out', 'col_out']
        in_related = ['batch', 'ch_in', 'row_out', 'col_out', 'row_kernel', 'col_kernel']
        we_related = ['ch_out', 'ch_in', 'row_kernel', 'col_kernel']
        all_data_types = ['in', 'out', 'we']  # input, output and weight
        all_refresh_locs = ['ref_gb_in', 'ref_gb_out', 'ref_gb_we', 'ref_rf_in', 'ref_rf_out', 'ref_rf_we']
        # print (len(df_order_in))
        df_order = copy.deepcopy(df_order_in)
        df_config_dict = copy.deepcopy(df_config_dict_in)
        for name in all_orders:
            if ('noc' in name) and (name not in df_order):
                df_order.append(name)
                df_config_dict[name] = 1.0
        num_active_pes = 1.0
        for i, df in enumerate(df_order):
            if df not in df_config_dict:
                # print('df_list and df_config_dict should be consistent')
                print(i)
                print(df)
                raise Exception('df_list {} and df_config_dict {} should be consistent'.format(i, df))
            if ('noc' in df) and (df in all_orders):
                num_active_pes *= df_config_dict[df]
        prod_out = 1.0
        prod_we = 1.0
        prod_plane = 1.0
        prod_inrow = 1.0
        prod_incol = 1.0
        prod_krow = 1.0
        prod_kcol = 1.0
        # cur_in = prod_plane*(prod_krow + stride*(prod_inrow-1))*(prod_kcol + stride*(prod_incol-1))
        # cur_out = prod_out
        # cur_we = prod_we

        # try:
        basic_add = self.add(bits_activation)
        basic_mul = self.mul(bits_activation, bits_weight)
        mac_one_pe = temp_merge(basic_add, basic_mul, self.plt)
        mac = opr_sum((int)(num_active_pes) * [mac_one_pe], self.plt, 'spatial')
        rf_to_alu_in_one_pe = self.rf_to_alu(bits_activation, bw_rf_to_alu_dict['in'])
        rf_to_alu_in = opr_sum((int)(num_active_pes) * [rf_to_alu_in_one_pe], self.plt, 'spatial')
        rf_to_alu_out_one_pe = self.rf_to_alu(bits_activation, bw_rf_to_alu_dict['out'])
        rf_to_alu_out = opr_sum((int)(num_active_pes) * [rf_to_alu_out_one_pe], self.plt, 'spatial')
        rf_to_alu_we_one_pe = self.rf_to_alu(bits_weight, bw_rf_to_alu_dict['we'])
        rf_to_alu_we = opr_sum((int)(num_active_pes) * [rf_to_alu_we_one_pe], self.plt, 'spatial')
        opr_first = opr_sum([rf_to_alu_in, rf_to_alu_we, mac, rf_to_alu_out], self.plt,
                            'temp')  # no psum read (only psum write) for the first,
        opr_other = temp_merge(rf_to_alu_out, opr_first, self.plt)  # psum read/write for the rest
        # print("basic_add_energy="+ str(basic_add.energy))
        # print("basic_mul_energy="+ str(basic_mul.energy))
        # print("mac_one_pe_energy="+ str(mac_one_pe.energy))
        # print("rf_to_alu_in_one_pe="+ str(rf_to_alu_in_one_pe.energy))
        # print("rf_to_alu_out_one_pe="+ str(rf_to_alu_out_one_pe.energy))
        # print("rf_to_alu_we_one_pe="+ str(rf_to_alu_we_one_pe.energy))
        E_comp = [0.0, 0.0]
        E_dram_to_gb = [0.0, 0.0]
        E_gb_to_noc = [0.0, 0.0]
        E_noc_to_rf = [0.0, 0.0]
        E_rf_to_alu = [0.0, 0.0]
        E_breakdown = [E_comp, E_dram_to_gb, E_gb_to_noc, E_noc_to_rf, E_rf_to_alu]
        E_breakdown[0][0] += mac.energy
        E_breakdown[0][1] += mac.energy
        E_breakdown[4][0] += rf_to_alu_in.energy + rf_to_alu_we.energy + rf_to_alu_out.energy
        E_breakdown[4][1] += rf_to_alu_in.energy + rf_to_alu_we.energy + 2 * rf_to_alu_out.energy
        # print (df_order)
        rf_volume_used = 0
        gb_volume_used = 0
        for df in df_order:
            if df in all_refresh_locs:
                # print (df)
                # print ('fuck')
                bw = df_config_dict[df]

                if 'in' in df:
                    bits1 = prod_plane * (prod_krow + stride * (prod_inrow - 1)) * (
                            prod_kcol + stride * (prod_incol - 1)) * bits_activation
                    bits2 = prod_plane * df_config_dict['batch_noc'] * df_config_dict['ch_in_noc'] \
                            * (prod_krow * df_config_dict['row_kernel_noc'] + stride * (
                            prod_inrow * df_config_dict['row_out_noc'] - 1)) \
                            * (prod_kcol * df_config_dict['col_kernel_noc'] + stride * (
                            prod_incol * df_config_dict['col_out_noc'] - 1)) * bits_activation
                    bw_gb_to_noc = bw_gb_to_noc_dict['in']
                elif 'we' in df:
                    bits1 = prod_we * bits_weight
                    bits2 = bits1 * df_config_dict['ch_in_noc'] * df_config_dict['ch_out_noc'] * df_config_dict[
                        'row_kernel_noc'] * df_config_dict['col_kernel_noc']
                    bw_gb_to_noc = bw_gb_to_noc_dict['we']
                elif 'out' in df:
                    bits1 = prod_out * bits_activation
                    bits2 = bits1 * df_config_dict['batch_noc'] * df_config_dict['ch_out_noc'] * df_config_dict[
                        'row_out_noc'] * df_config_dict['col_out_noc']
                    bw_gb_to_noc = bw_gb_to_noc_dict['out']
                else:
                    print(df)
                    print('error in function conv_df about in, out, we')
                    sys.exit(-1)

                if 'rf' in df:
                    rf_volume_used += bits1
                    comm_opr1_one_pe = self.noc_to_rf(bits1, df_config_dict[df])
                    comm_opr1 = opr_sum((int)(num_active_pes) * [comm_opr1_one_pe], self.plt, 'spatial')
                    comm_opr2 = self.gb_to_noc(bits2, bw_gb_to_noc)
                    comm_opr = temp_merge(comm_opr1, comm_opr2, self.plt)
                    if 'out' in df:
                        opr_first = temp_merge(opr_first, comm_opr, self.plt)
                        opr_other = opr_sum([comm_opr, opr_other, comm_opr], self.plt, 'temp')
                        E_breakdown[3][0] += comm_opr1.energy
                        E_breakdown[3][1] += 2 * comm_opr1.energy
                        E_breakdown[2][0] += comm_opr2.energy
                        E_breakdown[2][1] += 2 * comm_opr2.energy
                    else:
                        opr_first = temp_merge(opr_first, comm_opr, self.plt)
                        opr_other = temp_merge(opr_other, comm_opr, self.plt)
                        E_breakdown[3][0] += comm_opr1.energy
                        E_breakdown[3][1] += comm_opr1.energy
                        E_breakdown[2][0] += comm_opr2.energy
                        E_breakdown[2][1] += comm_opr2.energy
                elif 'gb' in df:
                    gb_volume_used += bits2
                    comm_opr = self.dram_to_gb(bits2, df_config_dict[df])
                    if 'out' in df:
                        opr_first = temp_merge(opr_first, comm_opr, self.plt)
                        opr_other = opr_sum([comm_opr, opr_other, comm_opr], self.plt, 'temp')
                        E_breakdown[1][0] += comm_opr.energy
                        E_breakdown[1][1] += 2 * comm_opr.energy
                    else:
                        opr_first = temp_merge(opr_first, comm_opr, self.plt)
                        opr_other = temp_merge(opr_other, comm_opr, self.plt)
                        E_breakdown[1][0] += comm_opr.energy
                        E_breakdown[1][1] += comm_opr.energy
                else:
                    print(df)
                    print('error in function conv_df about rf, noc, gb')
                    sys.exit(-1)


            elif (df in all_orders) and ('noc' in df):
                continue
            elif (df in all_orders) and ('noc' not in df):
                # print (df)
                # print (opr_first.time)
                # print (opr_other.time)
                opr_first = opr_sum([opr_first] + ((int)(df_config_dict[df] - 1)) * [opr_other], self.plt, 'temp')
                opr_other = opr_first
                for i in range(5):
                    e_first = E_breakdown[i][0]
                    e_other = E_breakdown[i][1]
                    E_breakdown[i][0] = e_first + (df_config_dict[df] - 1) * e_other
                    E_breakdown[i][1] = E_breakdown[i][0]
                for name in out_related:
                    if name in df:
                        prod_out *= df_config_dict[df]
                for name in we_related:
                    if name in df:
                        prod_we *= df_config_dict[df]
                if ('batch' in df) or ('ch_in' in df):
                    prod_plane *= df_config_dict[df]
                if ('row_out' in df):
                    prod_inrow *= df_config_dict[df]
                if ('col_out' in df):
                    prod_incol *= df_config_dict[df]
                if ('row_kernel' in df):
                    prod_krow *= df_config_dict[df]
                if ('col_kernel' in df):
                    prod_kcol *= df_config_dict[df]
                # print ('fuck')
            else:
                print(df)
                print('error in function conv_df')
                sys.exit(-1)
        opr_conv = opr_first
        # check if rf and gb volume is enough
        # print (gb_volume_used)
        opr_rf = self.occupy_volume(rf_volume_used, 'rf')
        # print (opr_rf.energy)
        opr_gb = self.occupy_volume(gb_volume_used, 'gb')
        return E_breakdown, opr_conv, opr_rf, opr_gb, num_active_pes
        # except:
        # pass


#     the seqence can be changed inside each row, the sequence for different rows is fixed row1<row2<row3<row4 (rf->noc->gb->dram)
#     row1 :col_kernel_rf, row_kernel_rf, col_out_rf, row_out_rf, ch_in_rf, ch_out_rf, batch_rf,
#     <
#     row2: col_kernel_noc, row_kernel_noc, col_out_noc, row_out_noc, ch_in_noc, ch_out_noc, batch_noc,
#     <
#     row3: col_kernel_gb, row_kernel_gb, col_out_gb, row_out_gb, ch_in_gb, ch_out_gb, batch_gb,
#     <
#     row4: col_kernel_dram, row_kernel_dram, col_out_dram, row_out_dram, ch_in_dram, ch_out_dram, batch_dram,

#     ref_gb_in,ref_gb_out,ref_gb_we, ref_rf_in,ref_rf_out,ref_rf_we can be inserted in row1, row3, row4,
#     but ref_rf_* <= ref_gb_*, where * can be {in,out,we}

# unit energy configuration for eyeriss
class one_config():
    def __init__(self, dram_bw, gb_vol, gb_bw, noc_bw, rf_vol, rf_bw, \
                 num_rf, num_adder, num_mul, num_pe, \
                 freq_pe, freq_dram, freq_gb, freq_noc, freq_rf, \
                 stride, df_order, df_config_dict, \
                 bits_activation, bits_weight, bw_gb_to_noc_dict, bw_rf_to_alu_dict):
        # configuration for the simulator which hw designer cannot change, it is for domain experts to fix
        dram_vol = float('inf')  # the dram volume (bits)
        bits_adder = 4  # the precision for the adder, don't change this one
        bits_mul = 4  # the precision for the multiplier, don't change this one
        e_adder = 1.0 / 68  # unit energy for each adder operation
        e_mul = 1.0 / 17  # unit energy for each multiplier operation  # set by Yang 7/12/2019, 1 MAC = 1 = 4 * E_add + 16 * E_mul
        cycles_add = 1.0  # cycles needed for each adder operation  # set by Yang 7/12/2019
        cycles_mul = 2.0  # cycles needed for each multiplier operation  # set by Yang 7/12/2019
        # dram->gb
        ebit_dram_to_gb = 12.5  # energy/bit for the dram->gb data communication
        e_dram_to_gb = 0  # setup energy for the dram->gb data communication
        t_dram_to_gb = 0  # setup time for dram->gb
        # gb->noc
        ebit_gb_to_noc = 0.375
        e_gb_to_noc = 0
        t_gb_to_noc = 0
        # noc->rf
        ebit_noc_to_rf = 0.125
        e_noc_to_rf = 0
        t_noc_to_rf = 0
        # rf->alu
        ebit_rf_to_alu = 0.0625
        e_rf_to_alu = 0
        t_rf_to_alu = 0
        try:
            hw_config1 = plt_config1(dram_vol, dram_bw, gb_vol, gb_bw, noc_bw,
                                     rf_vol, rf_bw, num_rf, num_adder, num_mul, num_pe,
                                     bits_adder, e_adder, bits_mul, e_mul, freq_pe, cycles_add, cycles_mul,
                                     #    bw_dram_to_gb, bw_gb_to_noc, bw_noc_to_rf, bw_rf_to_alu,
                                     ebit_dram_to_gb, ebit_gb_to_noc, ebit_noc_to_rf, ebit_rf_to_alu,
                                     e_dram_to_gb, e_gb_to_noc, e_noc_to_rf, e_rf_to_alu,
                                     freq_dram, freq_gb, freq_noc, freq_rf,
                                     t_dram_to_gb, t_gb_to_noc, t_noc_to_rf, t_rf_to_alu)
            E_breakdown, opr_conv, opr_rf, opr_gb, num_active_pes = hw_config1.conv_df(stride, df_order, df_config_dict,
                                                                                       bits_activation, bits_weight,
                                                                                       bw_gb_to_noc_dict,
                                                                                       bw_rf_to_alu_dict)
        except:
            # print ('this configuration is not correct')
            return
            # raise Exception('Wrong hardware configuration')
        # _, opr_conv,_,_ = hw_config1.conv_df(stride, df_order, df_config_dict, bits_activation, bits_weight, bw_gb_to_noc_dict ,bw_rf_to_alu_dict)
        if isinstance(opr_conv, int) or isinstance(opr_rf, int) or isinstance(opr_gb, int):
            # print ('this configuration is not correct')
            return
        self.hw_config = hw_config1
        self.conv_estimation = opr_conv
        self.Energy_breakdown = E_breakdown
        self.rf_estimation = opr_rf
        self.gb_estimation = opr_gb
        self.num_active_pes = num_active_pes

    def get_energy(self):
        try:
            return self.conv_estimation.energy
        except:
            return -1

    def get_latency(self):
        try:
            return self.conv_estimation.time
        except:
            return -1

    def get_energy_breakdown(self):
        try:
            return self.Energy_breakdown
        except:
            return -1

    def get_gb(self):
        try:
            return self.gb_estimation.plt.leaf_find_renum(self.gb_estimation.consume_list[0], 'volume').val
        except:
            return -1

    def get_rf(self):
        try:
            return self.rf_estimation.plt.leaf_find_renum(self.rf_estimation.consume_list[0], 'volume').val
        except:
            return -1

    def get_pes(self):
        try:
            return self.num_active_pes
        except:
            return -1


# search the refresh locations
class dse1():
    def __init__(self, save_path):
        self.stride = 1
        self.save_path = save_path
        # working frequency for dram, gb, noc, rf
        self.freq_dram = 90e6
        self.freq_gb = 250e6
        self.freq_noc = 250e6
        self.freq_pe = 250e6  # frequency for the PE  # set by Yang 7/12/2019
        self.freq_rf = 250e6

        self.dram_bw = 64  # the bitwidth for dram (bits)  # set by Yang 7/12/2019, not sure
        self.gb_bw = 64  # the bitwidth for global buffer (bits)
        self.noc_bw = 144 * 168  # the bitwidth for noc (bits)
        self.rf_bw = 64  # the bitwidth for rf (bits)

        self.num_adder = 4  # the number of adders in each PE  # set by Yang 7/12/2019, 16-bit adder = 4 * 4-bit adder ? actually
        self.num_mul = 16  # the number of multipliers in each PE  # set by Yang 7/12/2019, 16-bit mul = 16 * 4-bit adder ?

        self.gb_vol = 884736  # the global buffer volume (bits)  # set by Yang 7/12/2019, 108kB = 108*1024*8
        self.rf_vol = 4160  # the rf volume for rf (bits)  # set by Yang 7/12/2019, (448+24+48)B

        self.num_rf = 168  # the number of RF  # set by Yang 7/12/2019
        self.num_pe = 168  # the number of PE  # set by Yang 7/12/2019

        self.bw_gb_to_noc_dict = {'in': 64, 'out': 64, 'we': 64}
        self.bw_rf_to_alu_dict = {'in': 16, 'out': 16, 'we': 16}
        self.bits_weight = 16
        self.bits_activation = 16
        self.stride = 1
        self.df_config_dict = {'ch_out_rf': 16, 'ch_in_rf': 3, 'row_kernel_rf': 3, 'ref_rf_out': 64, 'row_out_rf': 13,
                               'ref_rf_in': 16, 'batch_rf': 4, \
                               'ref_rf_we': 64, 'col_kernel_noc': 3, 'ch_in_noc': 2, 'col_out_noc': 13, 'ch_out_noc': 2, \
                               'ref_gb_we': 64, 'ch_out_gb': 2, 'ref_gb_in': 64, 'ch_in_gb': 32, \
                               'ref_gb_out': 64, 'col_out_dram': 1, 'ch_out_dram': 4, 'batch_dram': 1
                               }

    def count_num(self, kw, list1):
        count = 0
        for a in list1:
            if kw in a:
                count += 1
        return count

    def search(self, print_freq):
        cost_best = float('inf')
        df_order_wo_ref = ['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'row_out_rf', 'batch_rf', \
                           'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc', \
                           'ch_out_gb', 'ch_in_gb', \
                           'col_out_dram', 'ch_out_dram', 'batch_dram'
                           ]
        count_rf = self.count_num('rf', df_order_wo_ref)
        count_noc = self.count_num('noc', df_order_wo_ref)
        # count_gb = self.count_num('gb',df_order_wo_ref)
        ori_len = len(df_order_wo_ref)
        search_space_def = ['ref_rf_we', 'ref_rf_in', 'ref_rf_out', 'ref_gb_we', 'ref_gb_in', 'ref_gb_out']
        search_space_dim = len(search_space_def)
        search_space = [(ref_rf_we_loc, ref_rf_in_loc, ref_rf_out_loc, ref_gb_we_loc, ref_gb_in_loc, ref_gb_out_loc)
                        for ref_rf_we_loc in range(count_rf + 1)
                        for ref_rf_in_loc in range(count_rf + 2)
                        for ref_rf_out_loc in range(count_rf + 3)
                        for ref_gb_we_loc in range(count_rf + count_noc + 4, ori_len + 4, 1)
                        for ref_gb_in_loc in range(count_rf + count_noc + 4, ori_len + 5, 1)
                        for ref_gb_out_loc in range(count_rf + count_noc + 4, ori_len + 6, 1)
                        ]
        converge_num = 0
        for j, design in enumerate(search_space):
            df_order = copy.deepcopy(df_order_wo_ref)
            for i in range(search_space_dim):
                df_order.insert(design[i], search_space_def[i])
            test_case = one_config(self.dram_bw, self.gb_vol, self.gb_bw, self.noc_bw, self.rf_vol, self.rf_bw, \
                                   self.num_rf, self.num_adder, self.num_mul, self.num_pe, \
                                   self.freq_pe, self.freq_dram, self.freq_gb, self.freq_noc, self.freq_rf, \
                                   self.stride, df_order, self.df_config_dict, \
                                   self.bits_activation, self.bits_weight, self.bw_gb_to_noc_dict,
                                   self.bw_rf_to_alu_dict)
            try:
                energy = test_case.get_energy()
                latency = test_case.get_latency()
                energy_breakdown = test_case.get_energy_breakdown()
                gb = test_case.get_gb()
                rf = test_case.get_rf()
                pes = test_case.get_pes()
                cost_obj = energy
                if -1 in [energy, latency, gb, rf, pes]:
                    # print (df_order)
                    continue
            except:
                continue
            if cost_obj < cost_best:
                cost_best = cost_obj
                self.best_test_case = test_case
                self.best_df_order = df_order
                f1 = open(self.save_path + '_best.obj', 'wb')
                pickle.dump(self, f1)
                f1.close()
            converge_num += 1
            f2 = open(self.save_path + '_itr_' + str(converge_num) + '.obj', 'wb')
            pickle.dump(self, f2)
            f2.close()
            if j % print_freq == 0 or cost_obj < cost_best:
                print(df_order)
                print('energy: ' + str(energy))
                print('latency: ' + str(latency))
                print('energy_breakdown: ' + str(energy_breakdown))
                print('gb volume: ' + str(gb) + '/' + str(self.gb_vol))
                print('rf volume: ' + str(rf) + '/' + str(self.rf_vol))
                print('num of pes: ' + str(pes))
        # self.best_df_order = best_df_order
        # self.best_test_case = best_test_case
        # # CONV5
        # df_order = ['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf',\
        #         'ref_rf_we', 'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc',\
        #         'ref_gb_we', 'ch_out_gb', 'ref_gb_in', 'ch_in_gb',\
        #         'ref_gb_out', 'col_out_dram', 'ch_out_dram', 'batch_dram'
        # ]

    def get_best(self):
        f2 = open(self.save_path + '_best.obj', 'rb')
        best_result = pickle.load(f2)
        return best_result


# eyeriss arhitecture scaled to dr.li
class dse2():
    def __init__(self, save_path):
        self.stride = 1
        self.save_path = save_path
        # working frequency for dram, gb, noc, rf
        self.freq_dram = 90e6
        self.freq_gb = 250e6
        self.freq_noc = 250e6
        self.freq_pe = 250e6  # frequency for the PE  # set by Yang 7/12/2019
        self.freq_rf = 250e6

        self.dram_bw = 64  # the bitwidth for dram (bits)  # set by Yang 7/12/2019, not sure
        self.gb_bw = 64  # the bitwidth for global buffer (bits)
        self.noc_bw = 144 * 512  # the bitwidth for noc (bits)
        self.rf_bw = 64  # the bitwidth for rf (bits)

        self.num_adder = 4  # the number of adders in each PE  # set by Yang 7/12/2019, 16-bit adder = 4 * 4-bit adder ? actually
        self.num_mul = 16  # the number of multipliers in each PE  # set by Yang 7/12/2019, 16-bit mul = 16 * 4-bit adder ?

        self.gb_vol = 884736.0 / 168 * 512  # the global buffer volume (bits)  # set by Yang 7/12/2019, 108kB = 108*1024*8
        self.rf_vol = 4160 * 10  # the rf volume for rf (bits)  # set by Yang 7/12/2019, (448+24+48)B

        self.num_rf = 512  # the number of RF  # set by Pengfei 7/23/2019
        self.num_pe = 512  # the number of PE  # set by Pengfei 7/23/2019

        self.bw_gb_to_noc_dict = {'in': 64, 'out': 64, 'we': 64}
        self.bw_rf_to_alu_dict = {'in': 16, 'out': 16, 'we': 16}
        self.bits_weight = 8
        self.bits_activation = 8

    def count_num(self, kw, list1):
        count = 0
        for a in list1:
            if kw in a:
                count += 1
        return count

    def rt_list(self, bound, step, mul):
        rt = [1]
        start = step if step != 0 else mul
        while start <= bound:
            rt.append(start)
            start = (start + step) * mul
        return rt

    def search(self, print_freq):
        cost_best = float('inf')
        # df_order_groundtruth = ['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf', 'ref_rf_we',
        #         'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc', \
        #         'ch_out_gb', 'ref_gb_in', 'ref_gb_we', 'ch_in_gb', 'ref_gb_out',\
        #         'col_out_dram', 'ch_out_dram', 'batch_dram']
        df_order_wo_ref = ['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'row_out_rf', 'batch_rf',
                           'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc', \
                           'ch_out_gb', 'ch_in_gb', \
                           'col_out_dram', 'ch_out_dram', 'batch_dram']
        count_rf = self.count_num('rf', df_order_wo_ref)
        count_noc = self.count_num('noc', df_order_wo_ref)
        count_gb = self.count_num('gb', df_order_wo_ref)
        ori_len = len(df_order_wo_ref)
        search_space1_def = ['ref_rf_we', 'ref_rf_in', 'ref_rf_out', 'ref_gb_we', 'ref_gb_in', 'ref_gb_out']
        search_space1_dim = len(search_space1_def)
        search_space1 = [(ref_rf_we_loc, ref_rf_in_loc, ref_rf_out_loc, ref_gb_we_loc, ref_gb_in_loc, ref_gb_out_loc)
                         # for ref_rf_we_loc in range(count_rf+1)
                         for ref_rf_we_loc in [5]
                         # for ref_rf_in_loc in range(count_rf+2)
                         for ref_rf_in_loc in [4]
                         # for ref_rf_out_loc in range(count_rf+3)
                         for ref_rf_out_loc in [3]
                         # for ref_gb_we_loc in range(count_rf+count_noc+4, count_rf+count_noc+count_gb+4,1)
                         for ref_gb_we_loc in [13]
                         # for ref_gb_in_loc in range(count_rf+count_noc+4, count_rf+count_noc+count_gb+5,1)
                         for ref_gb_in_loc in [13]
                         # for ref_gb_out_loc in range(count_rf+count_noc+4, count_rf+count_noc+count_gb+6,1)
                         for ref_gb_out_loc in [16]]
        # conv 64 3 3 224.0 224 1 1
        # conv 64 64 3 224.0 224.0 1 1
        # conv 128 64 3 112.0 112.0 1 1
        # conv 256 128 3 56.0 56.0 1 1
        # conv 512 256 3 28.0 28.0 1 1
        # conv 512 512 3 14.0 14.0 1 1
        self.ch_out = 512
        self.ch_in = 512
        self.out_size = 14
        self.kernel_size = 3
        self.batch_size = 1
        self.df_config_dict = {'ref_rf_out': self.rf_bw, 'ref_rf_in': self.rf_bw, 'ref_rf_we': self.rf_bw, \
                               'ref_gb_we': self.gb_bw, 'ref_gb_in': self.gb_bw, 'ref_gb_out': self.gb_bw}
        self.df_config_dict['row_kernel_rf'] = self.kernel_size
        self.df_config_dict['col_kernel_noc'] = self.kernel_size
        self.df_config_dict['batch_rf'] = 1
        self.df_config_dict['batch_dram'] = div_up(self.batch_size, self.df_config_dict['batch_rf'])
        search_space2 = [(ch_out_rf, ch_in_rf, row_out_rf, ch_in_noc, col_out_noc, ch_out_noc, ch_out_gb, ch_in_gb,
                          col_out_dram, ch_out_dram)
                         for ch_out_rf in range(1, self.ch_out + 1, 1)
                         for ch_in_rf in
                         self.rt_list(bound=min(self.ch_in, div_down(self.rf_vol / self.bits_weight, ch_out_rf)),
                                      step=2, mul=1)
                         for row_out_rf in [self.out_size]
                         for ch_in_noc in
                         self.rt_list(bound=min(div_up(self.ch_in, ch_in_rf), div_down(self.num_pe, self.kernel_size)),
                                      step=0, mul=2)
                         for col_out_noc in
                         self.rt_list(bound=min(self.out_size, div_down(self.num_pe, self.kernel_size * ch_in_noc)),
                                      step=0, mul=2)
                         for ch_out_noc in [min(div_up(self.ch_out, ch_out_rf),
                                                div_down(self.num_pe, self.kernel_size * ch_in_noc * col_out_noc))]
                         for ch_out_gb in self.rt_list(bound=div_up(self.ch_out, ch_out_rf * ch_out_noc), step=2, mul=1)
                         for ch_in_gb in [div_up(self.ch_in, ch_in_rf * ch_in_noc)]
                         for col_out_dram in [div_up(self.out_size, col_out_noc)]
                         for ch_out_dram in [div_up(self.ch_out, ch_out_gb * ch_out_noc * ch_out_rf)]
                         ]
        # print (len(search_space2))
        # sys.exit(-1)
        converge_num = 0
        for j, design in enumerate(search_space1):
            self.df_order = copy.deepcopy(df_order_wo_ref)
            for i in range(search_space1_dim):
                self.df_order.insert(design[i], search_space1_def[i])
            # print (df_order)
            # print (df_order_groundtruth)
            # sys.exit(-1)
            for i, ds2 in enumerate(search_space2):
                self.df_config_dict['ch_out_rf'] = ds2[0]
                self.df_config_dict['ch_in_rf'] = ds2[1]
                self.df_config_dict['row_out_rf'] = ds2[2]
                self.df_config_dict['ch_in_noc'] = ds2[3]
                self.df_config_dict['col_out_noc'] = ds2[4]
                self.df_config_dict['ch_out_noc'] = ds2[5]
                self.df_config_dict['ch_out_gb'] = ds2[6]
                self.df_config_dict['ch_in_gb'] = ds2[7]
                self.df_config_dict['col_out_dram'] = ds2[8]
                self.df_config_dict['ch_out_dram'] = ds2[9]
                test_case = one_config(self.dram_bw, self.gb_vol, self.gb_bw, self.noc_bw, self.rf_vol, self.rf_bw, \
                                       self.num_rf, self.num_adder, self.num_mul, self.num_pe, \
                                       self.freq_pe, self.freq_dram, self.freq_gb, self.freq_noc, self.freq_rf, \
                                       self.stride, self.df_order, self.df_config_dict, \
                                       self.bits_activation, self.bits_weight, self.bw_gb_to_noc_dict,
                                       self.bw_rf_to_alu_dict)
                try:
                    energy = test_case.get_energy()
                    latency = test_case.get_latency()
                    energy_breakdown = test_case.get_energy_breakdown()
                    gb = test_case.get_gb()
                    rf = test_case.get_rf()
                    pes = test_case.get_pes()
                    cost_obj = energy
                    if -1 in [energy, latency, gb, rf, pes]:
                        # print (df_order)
                        continue
                except:
                    cost_obj = float('inf')
                    continue
                if converge_num % print_freq == 0:
                    f2 = open(self.save_path + '_itr_' + str(converge_num) + '.obj', 'wb')
                    pickle.dump(self, f2)
                    f2.close()
                    print(self.df_order)
                    print('energy: ' + str(energy))
                    print('latency: ' + str(latency))
                    print('energy_breakdown: ' + str(energy_breakdown))
                    print('gb volume: ' + str(gb) + '/' + str(self.gb_vol))
                    print('rf volume: ' + str(rf) + '/' + str(self.rf_vol))
                    print('num of pes: ' + str(pes))
                if cost_obj < cost_best:
                    cost_best = cost_obj
                    self.best_test_case = test_case
                    self.best_df = self.df_config_dict
                    f1 = open(self.save_path + '_best.obj', 'wb')
                    pickle.dump(self, f1)
                    f1.close()
                    f2 = open(self.save_path + '_itr_' + str(converge_num) + '.obj', 'wb')
                    pickle.dump(self, f2)
                    f2.close()
                converge_num += 1
            # self.best_df_order = best_df_order
            # self.best_test_case = best_test_case
            # # CONV5
            # df_order = ['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf',\
            #         'ref_rf_we', 'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc',\
            #         'ref_gb_we', 'ch_out_gb', 'ref_gb_in', 'ch_in_gb',\
            #         'ref_gb_out', 'col_out_dram', 'ch_out_dram', 'batch_dram'
            # ]

    def get_best(self):
        f2 = open(self.save_path + '_best.obj', 'rb')
        best_result = pickle.load(f2)
        return best_result


# dr.li's architecture
class dse3():
    def __init__(self, save_path):
        self.bits_weight = 8
        self.bits_activation = 8
        self.bits_partialsum = 24
        self.stride = 1
        self.PE_MULT = 8  # added by Yang 7/23/2019
        self.PE_IC = 8  # added by Yang 7/23/2019
        self.PE_OC = 8  # added by Yang 7/23/2019
        self.save_path = save_path
        # working frequency for dram, gb, noc, rf
        self.freq_dram = 100e6
        self.freq_gb = 100e6
        self.freq_noc = 100e6
        self.freq_pe = 100e6  # frequency for the PE  # set by Yang 7/12/2019
        self.freq_rf = 100e6

        # self.dram_bw = 64 # the bitwidth for dram (bits)  # set by Yang 7/12/2019, not sure
        # self.gb_bw = 64 # the bitwidth for global buffer (bits)
        self.noc_bw = 144 * 512  # the bitwidth for noc (bits)
        # self.rf_bw = 64 # the bitwidth for rf (bits)

        self.num_adder = 4  # the number of adders in each PE  # set by Yang 7/12/2019, 16-bit adder = 4 * 4-bit adder ? actually
        self.num_mul = 16  # the number of multipliers in each PE  # set by Yang 7/12/2019, 16-bit mul = 16 * 4-bit adder ?

        self.num_rf = self.PE_MULT * self.PE_IC * self.PE_OC  # the number of RF  # set by Yang 7/23/2019
        self.num_pe = self.PE_MULT * self.PE_IC * self.PE_OC  # the number of PE  # set by Yang 7/23/2019

        self.rf_vol = self.bits_activation + self.bits_partialsum  # set by Yang 7/23/2019
        self.gb_vol = 300 * 1024 * 8  # the global buffer volume (bits)  # set by Yang 7/23/2019, 200KB

        # self.rf_vol = float('inf')  # set by Yang 7/23/2019
        # self.gb_vol = float('inf') # the global buffer volume (bits)  # set by Yang 7/23/2019, 200KB

        # self.dram_bw = 64 # the bitwidth for dram (bits)  # set by Yang 7/12/2019, not sure
        # self.gb_bw = 64 # the bitwidth for global buffer (bits)
        # self.noc_bw = self.PE_MULT * self.PE_IC * self.bits_activation + self.PE_IC * self.PE_OC * self.bits_weight \
        #               + self.PE_MULT * self.PE_OC * self.bits_partialsum  # the bitwidth for noc (bits) # set by Yang 7/23/2019
        # self.rf_bw = 64 # the bitwidth for rf (bits)

        self.bw_gb_to_noc_dict = {'in': self.PE_MULT * self.PE_IC, 'out': self.PE_MULT * self.PE_OC,
                                  'we': self.PE_IC * self.PE_OC}
        self.bw_rf_to_alu_dict = {'in': self.bits_activation, 'out': self.bits_partialsum, 'we': self.bits_weight}

    def count_num(self, kw, list1):
        count = 0
        for a in list1:
            if kw in a:
                count += 1
        return count

    def rt_list(self, bound, step, mul):
        rt = [1]
        start = step if step != 0 else mul
        while start <= bound:
            rt.append(start)
            start = (start + step) * mul
        return rt

    def search(self, print_freq):
        cost_best = float('inf')
        df_order_groundtruth = ['col_kernel_rf', 'row_kernel_rf', 'ref_rf_we', 'ref_rf_in', 'ref_rf_out', \
                                'col_out_noc', 'ch_in_noc', 'ch_out_noc', 'ref_gb_out', \
                                'ch_in_gb', 'row_out_gb', 'ch_out_gb', 'ref_gb_in', 'ref_gb_we', \
                                'col_out_dram', 'row_out_dram']
        df_order_wo_ref = ['col_kernel_rf', 'row_kernel_rf', \
                           'col_out_noc', 'ch_in_noc', 'ch_out_noc', \
                           'ch_in_gb', 'row_out_gb', 'ch_out_gb', \
                           'col_out_dram', 'row_out_dram']
        count_rf = self.count_num('rf', df_order_wo_ref)
        count_noc = self.count_num('noc', df_order_wo_ref)
        count_gb = self.count_num('gb', df_order_wo_ref)
        ori_len = len(df_order_wo_ref)
        search_space1_def = ['ref_rf_we', 'ref_rf_in', 'ref_rf_out', 'ref_gb_we', 'ref_gb_in', 'ref_gb_out']
        search_space1_dim = len(search_space1_def)
        search_space1 = [(ref_rf_we_loc, ref_rf_in_loc, ref_rf_out_loc, ref_gb_we_loc, ref_gb_in_loc, ref_gb_out_loc)
                         # for ref_rf_we_loc in range(count_rf+1)
                         for ref_rf_we_loc in [2]
                         # for ref_rf_in_loc in range(count_rf+2)
                         for ref_rf_in_loc in [3]
                         # for ref_rf_out_loc in range(count_rf+3)
                         for ref_rf_out_loc in [4]
                         # for ref_gb_we_loc in range(count_rf+count_noc+4, count_rf+count_noc+count_gb+4,1)
                         for ref_gb_we_loc in [11]
                         # for ref_gb_in_loc in range(count_rf+count_noc+4, count_rf+count_noc+count_gb+5,1)
                         for ref_gb_in_loc in [11]
                         # for ref_gb_out_loc in range(count_rf+count_noc+4, count_rf+count_noc+count_gb+6,1)
                         for ref_gb_out_loc in [8]]
        self.ch_out = 8
        self.ch_in = 64
        self.out_size = 16
        self.kernel_size = 1
        self.batch_size = 1
        self.df_config_dict = {}
        search_space2 = [(col_kernel_rf, row_kernel_rf, col_out_noc, ch_in_noc, ch_out_noc, ch_in_gb, row_out_gb,
                          ch_out_gb, col_out_dram, row_out_dram)
                         for col_kernel_rf in [self.kernel_size]
                         for row_kernel_rf in [self.kernel_size]
                         for col_out_noc in self.rt_list(bound=min(self.out_size, self.num_pe), step=0, mul=2)
                         for ch_in_noc in
                         self.rt_list(bound=min(self.ch_in, div_down(self.num_pe, col_out_noc)), step=0, mul=2)
                         for ch_out_noc in [min(self.ch_out, div_down(self.num_pe, ch_in_noc * col_out_noc))]
                         for ch_in_gb in [div_up(self.ch_in, ch_in_noc)]
                         for row_out_gb in self.rt_list(bound=self.out_size, step=2, mul=1)
                         for ch_out_gb in [div_up(self.ch_out, ch_out_noc)]
                         for col_out_dram in [div_up(self.out_size, col_out_noc)]
                         for row_out_dram in [div_up(self.out_size, row_out_gb)]
                         ]
        # print (len(search_space2))
        # sys.exit(-1)
        converge_num = 0
        for j, design in enumerate(search_space1):
            self.df_order = copy.deepcopy(df_order_wo_ref)
            for i in range(search_space1_dim):
                self.df_order.insert(design[i], search_space1_def[i])
            # print (df_order)
            # print (df_order_groundtruth)
            # sys.exit(-1)
            for i, ds2 in enumerate(search_space2):
                self.df_config_dict['col_kernel_rf'] = ds2[0]
                self.df_config_dict['row_kernel_rf'] = ds2[1]
                self.df_config_dict['col_out_noc'] = ds2[2]
                self.df_config_dict['ch_in_noc'] = ds2[3]
                self.df_config_dict['ch_out_noc'] = ds2[4]
                self.df_config_dict['ch_in_gb'] = ds2[5]
                self.df_config_dict['row_out_gb'] = ds2[6]
                self.df_config_dict['ch_out_gb'] = ds2[7]
                self.df_config_dict['col_out_dram'] = ds2[8]
                self.df_config_dict['row_out_dram'] = ds2[9]
                self.rf_bw = ds2[3] * ds2[2] * self.bits_activation  # the bitwidth for rf (bits)
                self.gb_bw = ds2[2] * ds2[3] * self.bits_activation + \
                             ds2[2] * ds2[4] * self.bits_partialsum + \
                             ds2[3] * ds2[4] * self.bits_weight  # the bitwidth for global buffer (bits)
                self.dram_bw = self.gb_bw  # the bitwidth for dram (bits)  # set by Yang 7/12/2019, not sure
                self.df_config_dict['ref_rf_out'] = 1
                self.df_config_dict['ref_rf_in'] = 1
                self.df_config_dict['ref_rf_we'] = 1
                self.df_config_dict['ref_gb_out'] = 1
                self.df_config_dict['ref_gb_in'] = 1
                self.df_config_dict['ref_gb_we'] = 1

                test_case = one_config(self.dram_bw, self.gb_vol, self.gb_bw, self.noc_bw, self.rf_vol, self.rf_bw, \
                                       self.num_rf, self.num_adder, self.num_mul, self.num_pe, \
                                       self.freq_pe, self.freq_dram, self.freq_gb, self.freq_noc, self.freq_rf, \
                                       self.stride, self.df_order, self.df_config_dict, \
                                       self.bits_activation, self.bits_weight, self.bw_gb_to_noc_dict,
                                       self.bw_rf_to_alu_dict)
                try:
                    energy = test_case.get_energy()
                    latency = test_case.get_latency()
                    energy_breakdown = test_case.get_energy_breakdown()
                    gb = test_case.get_gb()
                    rf = test_case.get_rf()
                    pes = test_case.get_pes()
                    cost_obj = energy
                    if -1 in [energy, latency, gb, rf, pes]:
                        # print (df_order)
                        continue
                except:
                    cost_obj = float('inf')
                    continue
                if converge_num % print_freq == 0 or cost_obj < cost_best:
                    f2 = open(self.save_path + '_itr_' + str(converge_num) + '.obj', 'wb')
                    pickle.dump(self, f2)
                    f2.close()
                    print(self.df_order)
                    print('energy: ' + str(energy))
                    print('latency: ' + str(latency))
                    print('energy_breakdown: ' + str(energy_breakdown))
                    print('gb volume: ' + str(gb) + '/' + str(self.gb_vol))
                    print('rf volume: ' + str(rf) + '/' + str(self.rf_vol))
                    print('num of pes: ' + str(pes))
                if cost_obj < cost_best:
                    cost_best = cost_obj
                    self.best_test_case = test_case
                    self.best_df = self.df_config_dict
                    f1 = open(self.save_path + '_best.obj', 'wb')
                    pickle.dump(self, f1)
                    f1.close()
                converge_num += 1
            # self.best_df_order = best_df_order
            # self.best_test_case = best_test_case
            # # CONV5
            # df_order = ['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf',\
            #         'ref_rf_we', 'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc',\
            #         'ref_gb_we', 'ch_out_gb', 'ref_gb_in', 'ch_in_gb',\
            #         'ref_gb_out', 'col_out_dram', 'ch_out_dram', 'batch_dram'
            # ]

    def get_best(self):
        f2 = open(self.save_path + '_best.obj', 'rb')
        best_result = pickle.load(f2)
        return best_result
