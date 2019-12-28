import argparse
import os
import sys
import math
import copy
import pickle
import _thread
__author__      = "Insomnia Pengfei Xu"




class cnn_model(object):#for other networks than vgg, prev_layers and next_layers , and ops should
    #be stored in the cnn_struct file
    #e.g.conv bn(true or false) relu (true or false) ch_out ch_in kernel_size size_out size_in stride padding prev=[2,4,6] next=[7,8,10,11] ops = ['get']
    def __init__(self,vgg_file_name):
        f1 = open(vgg_file_name, 'r')
        lines = f1.readlines()
        layers = []
        for i,line in enumerate(lines):
            info = line.split(' ')
            next_layers=[]
            ops = 'get'
            if (i==0):
                prev_layers =[]
            else:
                prev_layers = [layers[i-1]]
            if (info[0]=='conv'):
                bn = int(info[1])
                relu = int(info[2])
                ch_out = int(info[3])
                ch_in = int(info[4])
                kernel_size = int(info[5])
                size_out = int(float(info[6]))
                size_in = int(float(info[7]))
                stride = int(info[8])
                padding = int(info[9])
                # the order for the layer information: conv bn(true or false) relu (true or false) ch_out ch_in kernel_size size_out size_in stride padding
                new_layer = conv(prev_layers,next_layers,ops,bn, relu , ch_out, ch_in,kernel_size, size_out, size_in, stride,padding)
            elif (info[0]=='fc'):
                bn = int(info[1])
                relu = int(info[2])
                ch_out = int(info[3])
                ch_in = int(info[4])
                bias_or_not = int(info[5])
                # the order for the layer informataion: fc bn relu ch_out ch_in have_bias
                new_layer = fc(prev_layers,next_layers,ops,bn, relu,ch_out,ch_in,bias_or_not)
            elif (info[0]=='pooling'):
                kernel_size = int(info[1])
                size_out = int(float(info[2]))
                size_in = int(float(info[3]))
                new_layer = max_pooling(prev_layers,next_layers,ops,kernel_size, size_out, size_in)
            layers.append(new_layer)
        total_params = 0
        for i,layer in enumerate(layers):
            if (i==len(layers)-1):
                next_layers=[]
            else:
                next_layers=[layers[i+1]]
            layers[i].put_next_layers(next_layers)
            layers[i].update()
            total_params+=layers[i].num_params
            if (i<len(layers)-1):
                layers[i+1].param_ptr = total_params
        self.cnn = vgg_file_name.replace('_cnn_struct.txt','')
        self.layers=layers
        f1.close()
        self.num_pixels = 224*224*3
        self.total_params = total_params
    def print_info(self):
        print ('cnn_model='+self.cnn)
        print ('num_pixels='+str(self.num_pixels))
        print ('total_params='+str(self.total_params))
        print ('\n')
        for i in range(len(self.layers)):
            print (i)
            if self.layers[i].type !='conv':
                continue
            self.layers[i].print_info()
    def optimize_df(self,tech,df,batch_size,num_pe,smart_ex, search_mode, cnn,use_sparsity,use_min_req,op):
        sram_all_layers = 0
        rf_all_layers = 0
        for i in range(len(self.layers)):
            #if (self.layers[i].type=='conv' and self.layers[i].comp_stage == 16):
            if (self.layers[i].type=='conv' or (self.layers[i].type=='fc' and df!='RS_val')):
                print (type(self.layers[i]))
                print (self.layers[i].type)
                self.layers[i].optimize_df(tech,df,batch_size,num_pe,smart_ex, search_mode, cnn,use_sparsity,use_min_req,op)
        #         if self.layers[i].sram_this_layer > sram_all_layers:
        #             sram_all_layers = self.layers[i].sram_this_layer
        #         if self.layers[i].rf_this_layer > rf_all_layers:
        #             rf_all_layers = self.layers[i].rf_this_layer
        # self.sram_storage = sram_all_layers
        # self.rf_storage = rf_all_layers
    # each dram_loop will update all the sram storage
    # each sram_loop will load part sram data to pe array (compute)
    # struct of id_list: [layer][dram_loop][sram_loop][pe]
    def tiling(self):
        self.id_list_input = []
        self.id_list_output = []
        self.id_list_weight = []
        # for layer in self.layers:
        #     if layer.type !='conv' and layer.type!='fc':
        #         continue
        #     layer.tiling_data('65nm','RS',False)


class cnn_layer(object):
    def __init__(self,prev_layers,next_layers,ops):
        self.prev_layers = prev_layers
        self.ops = ops# operations to converting the layers in prev_layers into one layer
        #possible parameters are 'get':1-to-1; 'concat':n-to-1; 'add':n-to-1.
        self.next_layers = next_layers
        self.type=''
        self.param_ptr =0
        self.batch_size = 1
        # self.passes = 8
        #self.op_constraint = 'fix_data_flow_weight_staionary'# possible options are fix_data_flow, minimize_energy, maximize_throughput,
    def update(self):
        #computation state: denoting data dependency
        # the layer with the same comp_state have no dependency with each other
        # the computation stage for the first layer is set to be 1
        if (self.prev_layers==[]):
            self.comp_stage = 0
        else:
            llist = self.prev_layers
            max_val = llist[0].comp_stage
            for l in llist:
                if (l.comp_stage >max_val):
                    max_val= l.comp_stage
            self.comp_stage= max_val+1
    def print_info(self):
        print (self.type)
        print ('computation_stage='+str(self.comp_stage))
        print ('parameters_ptr='+str(self.param_ptr))
        print ('num_pe='+str(self.num_pe))
        for choice in ['input','output','weight']:
            print ('rf_storage_'+ choice+'='+ str(self.rf_storage[choice]))
            print ('sram_storage_'+choice+'='+ str(self.sram_storage[choice]))
            print ('dram_to_sram_'+choice+'='+ str(self.dram_to_sram[choice]))
            print ('sram_to_pe_'+choice+'='+ str(self.sram_to_pe[choice]))
            print ('to_pe_'+choice+'='+ str(self.to_pe[choice]))
            print ('pe_to_pe_'+choice+'='+ str(self.pe_to_pe[choice]))
        print('')
    def put_next_layers(self,next_layers):
        self.next_layers = next_layers
    # each dram_loop will update all the sram storage
    # each sram_loop will load part(or all) sram data to local (pe)
    # struct of id_list: [layer][dram_loop][sram_loop][pe]
    def tiling_data(self,tech,df,smart_ex):
        self.rf_storage = {}
        self.min_sram_storage = {}
        self.sram_storage = {}
        self.dram_to_sram = {}
        self.to_pe = {}
        self.pe_to_pe ={}
        self.sram_to_pe = {}
        self.to_rf = {}
        self.dram_loop ={}
        self.T1 = {'input':1.0,'weight':1.0,'output':1.0}
        self.dram_loop_dont_care ={}
        self.T2 = {'input':1.0,'weight':1.0,'output':1.0}
        self.sram_loop ={}
        self.T3 = {'input':1.0,'weight':1.0,'output':1.0}
        self.sram_loop_dont_care ={}
        self.T4 = {'input':1.0,'weight':1.0,'output':1.0}
        self.rf_loop ={}
        self.T5 = {'input':1.0,'weight':1.0,'output':1.0}
        self.rf_loop_dont_care ={}
        self.T6 = {'input':1.0,'weight':1.0,'output':1.0}
        self.T7 = {'input':1.0,'weight':1.0,'output':1.0} # parallel related
        self.reuse_dram={}
        self.reuse_sram ={}
        self.reuse_noc ={}
        self.reuse_rf = {}
        if self.type in ['conv','fc']:
            for opt in ['input','output','weight']:
                option = self.type +'_'+ opt
                if option == 'conv_input':
                    store_order = ['batch','ch_in','input_row','input_col']
                    self.store_order_input = store_order
                elif option == 'conv_output':
                    store_order = ['batch','ch_out','output_row','output_col']
                    self.store_order_output = store_order
                elif option == 'conv_weight':
                    store_order = ['ch_out','ch_in','kernel_row','kernel_col']
                    self.store_order_weight = store_order
                elif option == 'fc_input':
                    store_order = ['batch','ch_in']
                    self.store_order_input = store_order
                elif option == 'fc_output':
                    store_order = ['batch','ch_out']
                    self.store_order_output = store_order
                elif option == 'fc_weight':
                    store_order = ['ch_out','ch_in']
                    self.store_order_weight = store_order
                else:
                    print ('error, unexpected options!!!!!!!!!!')
                tiling_dict = self.tiling_dict
                ori_loop =[]
                for loop1 in self.loop_order:
                    loop1 = loop1.replace('output','input') if (opt == 'input') else loop1
                    if abs(tiling_dict[loop1]-1)>1e-5:
                        ori_loop.append(loop1)
                dram_loop = []
                dram_loop_dont_care = []
                sram_loop = []
                sram_loop_dont_care =[]
                rf_loop = []
                rf_loop_dont_care=[]
                dram_loop, dram_loop_dont_care, sram_loop, sram_loop_dont_care, rf_loop, rf_loop_dont_care = get_bound(ori_loop,store_order)
                reuse_dram = 1
                reuse_sram = 1
                reuse_noc = 1
                reuse_rf = 1
                num_dram_loop = 1 
                num_sram_loop = 1
                num_sram_loop_related = 1
                num_rf_loop = 1
                num_rf_loop_related = 1
                num_dram_loop_dont_care = 1
                num_pe = 1
                num_parall_related = 1
                num_sram_loop_dont_care = 1
                for loop1 in self.loop_parallel:
                    loop1 = loop1.replace('output','input') if opt =='input' else loop1
                    if loop1.replace('_dram','').replace('_sram','').replace('_rf','').replace('_pe','') in store_order:
                        if tiling_dict[loop1]==0:
                            print(loop1)
                        num_parall_related*= tiling_dict[loop1]
                        self.T7[opt] *= tiling_dict[loop1]
                        #print (loop1+' '+str(tiling_dict[loop1]))
                    num_pe *= tiling_dict[loop1]
                for loop1 in rf_loop_dont_care:
                    self.T6[opt] *= tiling_dict[loop1]
                for loop1 in rf_loop:
                    self.T5[opt] *= tiling_dict[loop1]
                    num_rf_loop *= tiling_dict[loop1]
                    if loop1.replace('_dram','').replace('_sram','').replace('_rf','').replace('_pe','') in store_order:
                        num_rf_loop_related *= tiling_dict[loop1]
                for loop1 in sram_loop:
                    self.T3[opt] *= tiling_dict[loop1]
                    loop1 = loop1.replace('output','input') if (opt=='input') else loop1
                    num_sram_loop *= tiling_dict[loop1]
                    if loop1.replace('_dram','').replace('_sram','').replace('_pe','').replace('_rf','') in store_order:
                        num_sram_loop_related *= tiling_dict[loop1]
                    else:
                        reuse_sram *= tiling_dict[loop1]
                for loop1 in sram_loop_dont_care:
                    self.T4[opt] *= tiling_dict[loop1]
                    loop1 = loop1.replace('output','input') if (opt=='input') else loop1
                    num_sram_loop_dont_care *= tiling_dict[loop1]
                for loop1 in dram_loop:
                    self.T1[opt] *= tiling_dict[loop1]
                    loop1 = loop1.replace('output','input') if (opt=='input') else loop1
                    num_dram_loop *= tiling_dict[loop1]
                    if loop1.replace('_dram','').replace('_sram','').replace('_pe','').replace('_rf','') not in store_order:
                        reuse_dram *= tiling_dict[loop1]                    
                    #print (tiling_dict[loop1])
                for loop1 in dram_loop_dont_care:
                    self.T2[opt] *= tiling_dict[loop1]
                    loop1 = loop1.replace('output','input') if (opt=='input') else loop1
                    num_dram_loop_dont_care *= tiling_dict[loop1]
                rf_storage = num_rf_loop_related*num_pe
                #print ('num_parallel_related = '+str(num_parall_related))
                sram_storage = num_rf_loop_related * num_parall_related * num_sram_loop_related
                dram_to_sram = sram_storage * num_dram_loop
                to_pe = rf_storage * num_dram_loop * num_dram_loop_dont_care * num_sram_loop if (num_sram_loop!= 1) else rf_storage * num_dram_loop#include sram to pe and pe to pe
                to_rf = num_dram_loop * num_dram_loop_dont_care * num_sram_loop * num_sram_loop_dont_care* num_pe *num_rf_loop
                # if (opt =='weight'):
                #     print (sram_loop)
                #     print (num_sram_loop)
                #     sys.exit(-1)
                sram_to_pe = float(to_pe*num_parall_related)/num_pe
                pe_to_pe = to_pe-sram_to_pe
                if opt == 'input':
                    store_order = ['batch','ch_in','input_row','input_col']#hard related
                    soft_related = ['kernel_row','kernel_col']# soft related
                    tiling_dict_min_sram = {}
                    tiling_dict_min_rf={}
                    tiling_dict_min_rf_soft ={}
                    tiling_dict_min_sram_soft={}
                    for input_dim in store_order:
                        tiling_dict_min_rf[input_dim]=1
                        tiling_dict_min_sram[input_dim]=1
                    for soft_dim in soft_related:
                        tiling_dict_min_rf_soft[soft_dim] = 1
                        tiling_dict_min_sram_soft[soft_dim] = 1
                    #print (sram_loop)
                    #print (sram_loop_dont_care)
                    #print (rf_loop)
                    #print (rf_loop_dont_care)
                    #sys.exit(-1)
                    for input_dim in store_order:
                        for level in ['_rf','_sram','_dram','_pe']:
                            tiling_dict_min_rf[input_dim] *= tiling_dict[input_dim+level] if ((input_dim+level) in rf_loop) else 1
                            tiling_dict_min_sram[input_dim] *= tiling_dict[input_dim+level] if ((input_dim+level) in rf_loop or (input_dim+level) in sram_loop or (input_dim+level).replace('input','output') in self.loop_parallel) else 1
                    for soft_dim in soft_related:
                        for level in ['_rf','_sram','_dram','_pe']:
                            tiling_dict_min_rf_soft[soft_dim] *= tiling_dict[soft_dim+level] if ((soft_dim+level) in rf_loop or (soft_dim+level) in rf_loop_dont_care) else 1
                            tiling_dict_min_sram_soft[soft_dim] *= tiling_dict[soft_dim+level] if ((soft_dim+level) in rf_loop or (soft_dim+level) in sram_loop or (soft_dim+level).replace('input','output') in self.loop_parallel or (soft_dim+level) in sram_loop_dont_care or (soft_dim+level) in rf_loop_dont_care) else 1
                    if option =='conv_input':
                        tiling_dict_min_rf['input_row'] = (tiling_dict_min_rf['input_row']-1)*self.stride + tiling_dict_min_rf_soft['kernel_row']
                        tiling_dict_min_rf['input_col'] = (tiling_dict_min_rf['input_col']-1)*self.stride + tiling_dict_min_rf_soft['kernel_col']
                        #print (tiling_dict_min_rf['input_row'])
                        tiling_dict_min_sram['input_row'] = (tiling_dict_min_sram['input_row']-1)*self.stride + tiling_dict_min_sram_soft['kernel_row']
                        tiling_dict_min_sram['input_col'] = (tiling_dict_min_sram['input_col']-1)*self.stride + tiling_dict_min_sram_soft['kernel_col']
                        #print (tiling_dict_min_sram['input_col'])
                    elif option =='max_pooling_input':
                        tiling_dict_min_rf['input_row'] = (tiling_dict_min_rf['input_row']-1)*self.kernel_size
                        tiling_dict_min_rf['input_col'] = (tiling_dict_min_rf['input_col']-1)*self.kernel_size
                        tiling_dict_min_sram['input_row'] = (tiling_dict_min_sram['input_row']-1)*self.kernel_size
                        tiling_dict_min_sram['input_col'] = (tiling_dict_min_sram['input_col']-1)*self.kernel_size
                    elif option =='fc_input':
                        pass
                    else:
                        print ('no such options, error!')
                        sys.exit(-1)
                    #print (tiling_dict_min_rf)
                    #print (tiling_dict_min_sram)
                    #print (tiling_dict_min_rf_soft)
                    #print (tiling_dict_min_sram_soft)
                    #print ('')
                    #print ('dram to sram '+opt)
                    rf_storage = num_pe
                    sram_storage = 1
                    for input_dim in store_order:
                        rf_storage *= tiling_dict_min_rf[input_dim]
                        sram_storage *= tiling_dict_min_sram[input_dim]
                        # print (tiling_dict_min_sram)
                        #print (tiling_dict_min_sram[input_dim])
                    dram_to_sram = sram_storage * num_dram_loop
                    # print (num_dram_loop)
                    # sys.exit(-1)
                    to_pe = rf_storage * num_dram_loop * num_dram_loop_dont_care * num_sram_loop if (num_sram_loop!= 1) else rf_storage * num_dram_loop #include sram to pe and pe to pe
                    sram_to_pe = float(to_pe*num_parall_related)/num_pe
                    pe_to_pe = to_pe-sram_to_pe
                    #print (num_to_sram)
                    #print (sram_storage)
                    #print (num_parall_related)
                    #print (num_rf_loop_related)
                    #print (num_sram_loop_dont_care)
                    #print ('dram to sram '+opt)
                    #print ('')
                dict_itr_sram={}
                dict_itr_dram={}
                #skip the loops in sram_loop_dont_care and dram_loop_dont_care
                self.num_pe = num_pe
                self.sram_storage[opt] = sram_storage
                self.rf_storage[opt] = rf_storage
                self.dram_to_sram[opt] = dram_to_sram
                self.to_pe[opt] = to_pe
                self.pe_to_pe[opt] = pe_to_pe
                self.to_rf[opt] = to_rf *2 if opt=='output' else to_rf
                self.sram_to_pe[opt] = 2*sram_to_pe if opt == 'output' else sram_to_pe
                self.dram_loop[opt] = dram_loop
                self.dram_loop_dont_care[opt] = dram_loop_dont_care
                self.sram_loop[opt] = sram_loop
                self.sram_loop_dont_care[opt] = sram_loop_dont_care
                self.rf_loop[opt] = rf_loop
                self.rf_loop_dont_care[opt] = rf_loop_dont_care
                self.reuse_noc[opt] = float(num_pe)/num_parall_related
                self.reuse_dram[opt] = reuse_dram
                self.reuse_sram[opt] = reuse_sram*2 if opt == 'output' else reuse_sram
                self.reuse_rf[opt] = -1
            if df == 'OSC':
                self.rf_storage['weight'] = 2*self.tiling_dict['ch_out_pe']
                self.sram_storage['weight'] = 2*self.tiling_dict['ch_out_pe']
            #rf storage (each rf) and sram storage: KB
            #area = sram_storage *k_sram + rf_storage*num_pe*k_rf + num_pe*k_pe
            if tech == '65nm': # 2 Byte for input ,output, weight
                self.rf_storage['input'] = 2*float(self.rf_storage['input'])/1024/self.num_pe
                self.rf_storage['output'] = 2*float(self.rf_storage['output'])/1024/self.num_pe
                self.rf_storage['weight'] = 2*float(self.rf_storage['weight'])/1024/self.num_pe
                self.sram_storage['input'] = 2*float(self.sram_storage['input'])/1024
                self.sram_storage['output'] = 2*float(self.sram_storage['output'])/1024
                self.sram_storage['weight'] = 2*float(self.sram_storage['weight'])/1024
            elif tech == '28nm':# 1 Byte for input and weight, 2 Byte for output
                self.rf_storage['input'] = float(self.rf_storage['input'])/1024/self.num_pe
                self.rf_storage['output'] = 2*float(self.rf_storage['output'])/1024/self.num_pe
                self.rf_storage['weight'] = float(self.rf_storage['weight'])/1024/self.num_pe
                self.sram_storage['input'] = float(self.sram_storage['input'])/1024
                self.sram_storage['output'] = 2*float(self.sram_storage['output'])/1024
                self.sram_storage['weight'] = float(self.sram_storage['weight'])/1024
                if smart_ex and df in ['NLR','OSA']:
                    self.sram_storage['weight'] *=0.5
        elif self.type == 'max_pooling':
            for opt in ['input','output','weight']:
                option = self.type +'_'+ opt
                if option == 'max_pooling_input':
                    store_order = ['batch','ch','input_row','input_col']
                    self.store_order_input = store_order
                elif option == 'max_pooling_output':
                    store_order = ['batch','ch','output_row','output_col']
                    self.store_order_output = store_order
                elif option == 'max_pooling_weight':
                    store_order = []
                    self.store_order_weight = store_order
                else:
                    print ('error, unexpected options!!!!!!!!!!')


class conv(cnn_layer):
    # the order for the layer information: conv bn(true or false) relu (true or false) ch_out ch_in kernel_size size_out size_in stride padding
    def __init__(self,prev_layers,next_layers,ops,
                 bn, relu , ch_out, ch_in,
                 kernel_size, size_out, size_in, stride,
                 padding):
        cnn_layer.__init__(self,prev_layers,next_layers,ops)
        self.type='conv'
        self.bn = bn
        self.relu = relu
        self.ch_out = ch_out
        self.ch_in = ch_in
        self.kernel_size = kernel_size
        self.size_out = size_out
        self.size_in = size_in
        self.stride = stride
        self.padding = padding
        num_params = ch_out*ch_in*kernel_size*kernel_size+ch_out
        if (bn==1):
            #print 'bn'
            num_params+= ch_out*2
        self.num_params=num_params
    def print_info(self):
        cnn_layer.print_info(self)
        print ('have_batch_normalizaiton='+str(self.bn))
        print ('have_relu='+str(self.relu))
        print ('batch_size='+str(self.batch_size))
        print ('ch_out='+str(self.ch_out))
        print ('ch_in='+str(self.ch_in))
        print ('kernel_size='+str(self.kernel_size))
        print ('size_out='+str(self.size_out))
        print ('size_in='+str(self.size_in))
        print ('stride='+str(self.stride))
        print ('padding='+str(self.padding))
        print ('num_params='+str(self.num_params))
        print (self.tiling_dict)
        print (self.loop_order)
        print ('\n')

class fc(cnn_layer):
    # the order for the layer informataion: fc bn relu ch_out ch_in have_bias
    def __init__(self, prev_layers,next_layers,ops,
                bn, relu,ch_out,ch_in,bias_or_not):
        cnn_layer.__init__(self,prev_layers,next_layers,ops)
        self.type='fc'
        self.bn = bn
        self.bias_or_not = bias_or_not
        self.relu = relu
        self.ch_out = ch_out
        self.ch_in = ch_in
        self.kernel_size = 1
        self.size_out = 1
        self.size_in = 1
        num_params = ch_in*ch_out
        if (bias_or_not==1):
            num_params+=ch_out
        if (bn==1):
            num_params+=ch_out*2
        self.num_params=num_params
    def print_info(self):
        cnn_layer.print_info(self)
        print ('have_batch_normalization='+str(self.bn))
        print ('have_relu='+str(self.relu))
        print ('have_bias='+str(self.bias_or_not))
        print ('batch_size='+str(self.batch_size))
        print ('ch_out='+str(self.ch_out))
        print ('ch_in='+str(self.ch_in))
        print ('num_params='+str(self.num_params))
        print (self.tiling_dict)
        print (self.loop_order)
        print ('\n')

class max_pooling(cnn_layer):
    def __init__(self, prev_layers,next_layers,ops,
                kernel_size, size_out, size_in):
        cnn_layer.__init__(self,prev_layers, next_layers,ops)
        self.type = 'max_pooling'
        self.kernel_size = kernel_size
        self.size_out = size_out
        self.size_in = size_in
        self.num_params = 0
        llist = self.prev_layers
        self.ch = llist[0].ch_out
    def print_info(self):
        cnn_layer.print_info(self)
        print ('kernel_size='+str(self.kernel_size))
        print ('batch_size='+str(self.batch_size))
        print ('ch='+str(self.ch))
        print ('size_out='+str(self.size_out))
        print ('size_in='+str(self.size_in))
        print ('num_params='+str(self.num_params))
        print (self.tiling_dict)
        print (self.loop_order)
        print ('\n')
