import random
import numpy as np
import test_for_eyeriss as simnas
import time
from itertools import combinations,permutations
import copy
from  multiprocessing import Queue
import multiprocessing
import math

default_hw={ \
    'gb_vol':108*1024*8, \
    'rf_vol':6893, \
    'num_pe':168, \
    'num_rf':168
}

##############################
#shared util funcs
#############################

def life_eval(actions,stride,hw_spec,mode,group_num=1,df_order=None):
    #function to query chip_estimator and get energy+latency feedback

    #actions: tiling factors for a specific loop-order
    #stride: the stride number for this CONV layer operation
    #hw_spec: hw specs for evaluation
    #df_order: loop-order for evaluation 
    #           !!!!if not provided PLS provide it in chip_estimator
    #           !!!!legacy functionality, so always try to provide specific loop-order here
    try:
        if mode!=2 and group_num!=1:
            print('You did not choose group convolution, please set group num to 1')
            raise
        #input isolation
        input_actions=dict(actions)
        if df_order:
            input_df_order=list(df_order)
        else:
            input_df_order=None
        ene_results=simnas.sample_energy(input_actions,stride,hw_spec,mode,input_df_order=input_df_order)
        penalty=-(ene_results[0]*1e-8*ene_results[1]*100)*group_num*group_num
        buffer_not_exceed=True
        #print(ene_results[0],ene_results[1])
    #if design hw constraint exceeded, 
    #if exceeded return extremely large penalty
    except Exception as e:
        if 'resource' in str(e):
            pass
        else:
            print('error:',e)
            print(actions)
            print(df_order)         
        penalty=-9e12                                  #very strong penalty to over budget
        buffer_not_exceed=False
    return penalty, buffer_not_exceed



    




#noc_template to be considered 
noc_template=[['col_kernel_noc','row_kernel_noc','ch_in_noc','ch_out_noc'], \
                      ['col_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      ['row_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      ['row_out_noc','col_out_noc','ch_out_noc'], \
                      ]

noc_template_dw=[['col_kernel_noc','row_kernel_noc','ch_out_noc'], \
                      ['col_kernel_noc','col_out_noc','ch_out_noc'], \
                      ['row_kernel_noc','col_out_noc','ch_out_noc'], \
                      ['row_out_noc','col_out_noc','ch_out_noc'], \
                      ]




#######################
#layer level util func
#######################
#find the factors of a number
def r_factors(x):
    #find the factors of a number
    factor_list=[]
    for i in range(1, x + 1):
        if x % i == 0:
            factor_list.append(i)
    return factor_list
def diff_cal(factors):
    diff_sum=0
    for i in range(1,len(factors)):
        diff_sum+=abs(factors[i]-factors[i-1])
    return diff_sum
        
def factor_n(x,n=3,flexible_factor=1):
    #return the factor combo of length n for number x
    #flexible number:
    #               return factor combo of length n for number [x,flexible_factor)
    #               with requirement that the factors in in factor combo can not differ too much which is bad for resource partition


    #force one if n==1
    if n==1:
        flexible_factor=1
    #initialize max diff among factors and if this is original input or not
    diff_sum_min=math.inf
    input=True
    result=[]
    for _ in range(flexible_factor):
        #return factors of x
        factor_list=r_factors(x)
        num=factor_list[-1]
        tmp_list=[]
        for i in factor_list:
            for _ in range(n):
                tmp_list.append(i)
        # Get all combinations of factor_list
        # and length n
        comb = combinations(tmp_list, n) 
        for i in list(comb):
            mult=1
            for f in i:
                mult*=f
            if mult==num and (i not in result):               
                if input:
                    result.append(i)
                else:
                    if diff_cal(i)<diff_sum_min:
                        result.append(i)
                        diff_sum_min=diff_cal(i)
        if input:
            for i in result:
                tmp_diff_sum=diff_cal(i)
                if tmp_diff_sum<diff_sum_min:
                    diff_sum_min=tmp_diff_sum
        x+=1
        input=False
    return result

def permute_factor(input_factor_list):
    #permute the order within each factor in the factor_list
    #input  isolation
    factor_list=copy.deepcopy(input_factor_list)
    result=[]
    for f in factor_list:
        perm = permutations(f)     
        # Print the obtained permutations                        
        for i in list(perm): 
            if i not in result:                             
                result.append(i)
    return result





 




    
#####################    
#threading util
####################

def multi_p(func,args,output_q,num_worker_threads,dump_yard):
    #routine to distribute workers to multi cores
    #BETTER leave it

    #length of args has to be the multiple of num_worker_threads
    args=list(args)
    run_ites=int((len(args))//num_worker_threads)
    for run_ite in range(run_ites):
        processes = [multiprocessing.Process(target=func, args=([args[i]])) for i in range(run_ite*num_worker_threads,(run_ite+1)*num_worker_threads)]
        #print(len(processes))
        #print('queue size: ',score_pair.qsize())
        for p in processes:
            p.start()
        while not output_q.empty():
            pair=output_q.get()
            dump_yard.append(pair)
        for p in processes:
            p.join()
    while not output_q.empty():
        pair=output_q.get()
        dump_yard.append(pair)
    return None



###################################
#combined level
###################################





def data_type_sizes(input_dnn):
    input_size=0
    output_size=0
    kernel_size=0
    for layer in input_dnn:
        if layer[2]==0:
            input_size+=layer[1]['ch_in'][0]*layer[1]['row_out'][0]*layer[1]['col_out'][0]*layer[0]
            kernel_size+=layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*layer[1]['row_kernel'][0]
            output_size+=layer[1]['ch_out'][0]*layer[1]['row_out'][0]*layer[1]['col_out'][0]
        elif layer[2]==1:
            input_size+=layer[1]['ch_out'][0]*layer[1]['row_out'][0]*layer[1]['col_out'][0]*layer[0]
            kernel_size+=layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*layer[1]['row_kernel'][0]
            output_size+=layer[1]['ch_out'][0]*layer[1]['row_out'][0]*layer[1]['col_out'][0]
        elif layer[2]==2:
            input_size+=layer[1]['ch_in'][0]*layer[1]['row_out'][0]*layer[1]['col_out'][0]*layer[0]*layer[3]
            kernel_size+=layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*layer[1]['row_kernel'][0]*layer[3]
            output_size+=layer[1]['ch_out'][0]*layer[1]['row_out'][0]*layer[1]['col_out'][0]*layer[3]
    return input_size, output_size, kernel_size
    
def resource_allocator(input_dnn,tmp_hw_spec):
    gb=tmp_hw_spec['gb_vol']
    pe=tmp_hw_spec['num_pe']
    para=[]
    comp=[]
    gb_all=[]
    pe_all=[]
    for layer in input_dnn:
        if layer[2]==0:
            para.append(layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]+layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0]+layer[0]*layer[1]['ch_in'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])
            comp.append(layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]*layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])
        elif layer[2]==1:
            para.append(layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]+layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0]+layer[0]*layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])
            comp.append(layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]*layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])
        elif layer[2] == 2:
            para.append((layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]+layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0]+layer[0]*layer[1]['ch_in'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])*layer[3])
            comp.append((layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]*layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0]/layer[3])*layer[3])
    for i in para:
        gb_all.append(math.ceil(i/sum(para)*gb))
    for i in comp:
        pe_all.append(math.ceil(i/sum(comp)*pe))
    return gb_all,pe_all
    
class fpga_tiling_generator():
    def __init__(self, input_dnn, hw_spec):
        self.pe_array_dim_num=1
        input_dnn=copy.deepcopy(input_dnn)
        self.input_dnn=input_dnn
        hw_spec=copy.deepcopy(hw_spec)
        [self.gb_all,self.dsp_limit]=resource_allocator(input_dnn,hw_spec)

        ch_in=[]
        ch_out=[]
        row_out=[]
        col_out=[]
        col_kernel=[]
        row_kernel=[]
        #collect all dimension information
        for layer in input_dnn:
            if layer[2]==0 or layer[2]==2:
                ch_in.append(layer[1]['ch_in'][0])
                ch_out.append(layer[1]['ch_out'][0])
                row_out.append(layer[1]['row_out'][0])
                col_out.append(layer[1]['col_out'][0])
                row_kernel.append(layer[1]['row_kernel'][0])
                col_kernel.append(layer[1]['col_kernel'][0])
            elif layer[2]==1:
                ch_out.append(layer[1]['ch_out'][0])
                row_out.append(layer[1]['row_out'][0])
                col_out.append(layer[1]['col_out'][0])
                row_kernel.append(layer[1]['row_kernel'][0])
                col_kernel.append(layer[1]['col_kernel'][0])
        
        #weight stationary
        #noc standard:
        #row/col_kernel_noc=max(kernel_size)
        #increment channel until limit reached
        #rf gb standard. max out usage as possible
        #row stationary1 
        self.ws_noc=[]
        self.ws_rf_gb_tiling_choices=[]
        self.ws_rf_gb_tiling_choices_num=[]
        for i in range(len(input_dnn)):
            self.ws_noc.append([])
            self.ws_rf_gb_tiling_choices.append([])
            self.ws_rf_gb_tiling_choices_num.append([])
            if input_dnn[i][2]==0 or input_dnn[i][2]==2:
                ch_in_choices=sorted(r_factors(input_dnn[i][1]['ch_in'][0]))
            ch_out_choices=sorted(r_factors(input_dnn[i][1]['ch_out'][0]))
            col_kernel_choices=sorted(r_factors(input_dnn[i][1]['col_kernel'][0]))
            row_kernel_choices=sorted(r_factors(input_dnn[i][1]['row_kernel'][0]))
            
            
            pe_designs=[]
            str1=[]
            if input_dnn[i][2]==0 or input_dnn[i][2]==2:
                for m in ch_in_choices:
                    for l in ch_out_choices:
                        for rk in row_kernel_choices:
                            for ck in col_kernel_choices:
                                if m*l*rk*ck <= self.dsp_limit[i]:
                                    pe_designs.append(m*l*rk*ck)
                                    str1.append((m,l,ck,rk))
                pe_designs=[i[1] for i in sorted(zip(pe_designs,str1),reverse=True)]
            elif input_dnn[i][2]==1:
                for l in ch_out_choices:
                    for rk in row_kernel_choices:
                        for ck in col_kernel_choices:
                            if l*rk*ck <= self.dsp_limit[i]:
                                pe_designs.append(l*rk*ck)
                                str1.append((l,ck,rk))
                pe_designs=[i[1] for i in sorted(zip(pe_designs,str1),reverse=True)]
            #print(pe_designs[1:10])
            
            for pe_design in pe_designs[0:self.pe_array_dim_num]:
                self.ws_noc[-1].append({})
                self.ws_rf_gb_tiling_choices[-1].append({})
                self.ws_rf_gb_tiling_choices_num[-1].append([])
                if input_dnn[i][2]==0 or input_dnn[i][2]==2:
                    self.ws_noc[-1][-1]['ch_in_noc']=pe_design[0]
                    self.ws_noc[-1][-1]['ch_out_noc']=pe_design[1]
                    self.ws_noc[-1][-1]['col_kernel_noc']=pe_design[2]
                    self.ws_noc[-1][-1]['row_kernel_noc']=pe_design[3]
                elif input_dnn[i][2] == 1:
                    self.ws_noc[-1][-1]['ch_out_noc']=pe_design[0]
                    self.ws_noc[-1][-1]['col_kernel_noc']=pe_design[1]
                    self.ws_noc[-1][-1]['row_kernel_noc']=pe_design[2]
                if input_dnn[i][2] == 0 or input_dnn[i][2]==2:
                    self.ws_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_in'][0]//self.ws_noc[-1][-1]['ch_in_noc'],2)
                    self.ws_rf_gb_tiling_choices_num[-1][-1].append(len(self.ws_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']))
                
                self.ws_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_out'][0]//self.ws_noc[-1][-1]['ch_out_noc'],2)
                self.ws_rf_gb_tiling_choices_num[-1][-1].append(len(self.ws_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']))
                
                self.ws_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']=factor_n(input_dnn[i][1]['col_out'][0],2)
                self.ws_rf_gb_tiling_choices_num[-1][-1].append(len(self.ws_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']))
                
                self.ws_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']=factor_n(input_dnn[i][1]['row_out'][0],2)
                self.ws_rf_gb_tiling_choices_num[-1][-1].append(len(self.ws_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']))
            
                self.ws_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['col_kernel'][0]//self.ws_noc[-1][-1]['col_kernel_noc'],2)
                self.ws_rf_gb_tiling_choices_num[-1][-1].append(len(self.ws_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']))    

                self.ws_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['row_kernel'][0]//self.ws_noc[-1][-1]['row_kernel_noc'],2)
                self.ws_rf_gb_tiling_choices_num[-1][-1].append(len(self.ws_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']))                 

                self.ws_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']=factor_n(input_dnn[i][1]['batch'][0],2)
                self.ws_rf_gb_tiling_choices_num[-1][-1].append(len(self.ws_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']))

            
            
            
            
            
        #output stationary
        self.os_noc=[]
        self.os_rf_gb_tiling_choices=[]
        self.os_rf_gb_tiling_choices_num=[]
        for i in range(len(input_dnn)):
            self.os_noc.append([])
            self.os_rf_gb_tiling_choices.append([])
            self.os_rf_gb_tiling_choices_num.append([])
            ch_out_choices=sorted(r_factors(input_dnn[i][1]['ch_out'][0]))
            col_out_choices=sorted(r_factors(input_dnn[i][1]['col_out'][0]))
            row_out_choices=sorted(r_factors(input_dnn[i][1]['row_out'][0]))
            
            
            pe_designs=[]
            str1=[]
            for l in ch_out_choices:
                for j in list(combinations(col_out_choices,2)):
                        if l*j[0]*j[1] <= self.dsp_limit[i]:
                            pe_designs.append(j[0]*j[1]*l)
                            str1.append((l,j[0],j[1]))
                for k in col_out_choices:
                        if l*k*k <= self.dsp_limit[i]:
                            pe_designs.append(k*k*l)
                            str1.append((l,k,k))
            pe_designs=[i[1] for i in sorted(zip(pe_designs,str1),reverse=True)]
            #print(pe_designs[1:10])
            
            for pe_design in pe_designs[0:self.pe_array_dim_num]:
                self.os_noc[-1].append({})
                self.os_rf_gb_tiling_choices[-1].append({})
                self.os_rf_gb_tiling_choices_num[-1].append([])
                self.os_noc[-1][-1]['ch_out_noc']=pe_design[0]
                self.os_noc[-1][-1]['col_out_noc']=pe_design[1]
                self.os_noc[-1][-1]['row_out_noc']=pe_design[2]
                if input_dnn[i][2] == 0 or input_dnn[i][2]==2:
                    self.os_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_in'][0],2)
                    self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']))
                
                self.os_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_out'][0]//self.os_noc[-1][-1]['ch_out_noc'],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']))
                
                self.os_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']=factor_n(input_dnn[i][1]['col_out'][0]//self.os_noc[-1][-1]['col_out_noc'],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']))
                
                self.os_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']=factor_n(input_dnn[i][1]['row_out'][0]//self.os_noc[-1][-1]['row_out_noc'],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']))
            
                self.os_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['col_kernel'][0],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']))    

                self.os_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['row_kernel'][0],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']))                 

                self.os_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']=factor_n(input_dnn[i][1]['batch'][0],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']))


            
            

            
            
        #row stationary1 
        self.rs1_noc=[]
        self.rs1_rf_gb_tiling_choices=[]
        self.rs1_rf_gb_tiling_choices_num=[]
        for i in range(len(input_dnn)):
            self.rs1_noc.append([])
            self.rs1_rf_gb_tiling_choices.append([])
            self.rs1_rf_gb_tiling_choices_num.append([])
            if input_dnn[i][2]==0 or input_dnn[i][2]==2:
                ch_in_choices=sorted(r_factors(input_dnn[i][1]['ch_in'][0]))
            ch_out_choices=sorted(r_factors(input_dnn[i][1]['ch_out'][0]))
            col_out_choices=sorted(r_factors(input_dnn[i][1]['col_out'][0]))
            row_kernel_choices=sorted(r_factors(input_dnn[i][1]['row_kernel'][0]))
            
            
            pe_designs=[]
            str1=[]
            if input_dnn[i][2]==0 or input_dnn[i][2]==2:
                for m in ch_in_choices:
                    for l in ch_out_choices:
                        for j in col_out_choices:
                            for rk in row_kernel_choices:
                                if m*l*j*rk<= self.dsp_limit[i]:
                                    pe_designs.append(m*l*j*rk)
                                    str1.append((m,l,j,rk))
                pe_designs=[i[1] for i in sorted(zip(pe_designs,str1),reverse=True)]
            elif input_dnn[i][2]==1:
                for l in ch_out_choices:
                    for j in col_out_choices:
                        for rk in row_kernel_choices:
                            if l*j*rk <= self.dsp_limit[i]:
                                pe_designs.append(l*j*rk)
                                str1.append((l,j,rk))
                pe_designs=[i[1] for i in sorted(zip(pe_designs,str1),reverse=True)]
            #print(pe_designs[1:10])
            
            for pe_design in pe_designs[0:self.pe_array_dim_num]:
                self.rs1_noc[-1].append({})
                self.rs1_rf_gb_tiling_choices[-1].append({})
                self.rs1_rf_gb_tiling_choices_num[-1].append([])
                if input_dnn[i][2]==0 or input_dnn[i][2]==2:
                    self.rs1_noc[-1][-1]['ch_in_noc']=pe_design[0]
                    self.rs1_noc[-1][-1]['ch_out_noc']=pe_design[1]
                    self.rs1_noc[-1][-1]['col_out_noc']=pe_design[2]
                    self.rs1_noc[-1][-1]['row_kernel_noc']=pe_design[3]
                elif input_dnn[i][2]==1:
                    self.rs1_noc[-1][-1]['ch_out_noc']=pe_design[0]
                    self.rs1_noc[-1][-1]['col_out_noc']=pe_design[1]
                    self.rs1_noc[-1][-1]['row_kernel_noc']=pe_design[2]
                if input_dnn[i][2] == 0 or input_dnn[i][2]==2:
                    self.rs1_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_in'][0]//self.rs1_noc[-1][-1]['ch_in_noc'],2)
                    self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']))
                
                self.rs1_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_out'][0]//self.rs1_noc[-1][-1]['ch_out_noc'],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']))
                
                self.rs1_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']=factor_n(input_dnn[i][1]['col_out'][0]//self.rs1_noc[-1][-1]['col_out_noc'],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']))
                
                self.rs1_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']=factor_n(input_dnn[i][1]['row_out'][0],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']))
            
                self.rs1_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['col_kernel'][0],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']))    

                self.rs1_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['row_kernel'][0]//self.ws_noc[-1][-1]['row_kernel_noc'],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']))                 

                self.rs1_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']=factor_n(input_dnn[i][1]['batch'][0],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']))

            #noc_template to be considered 
# noc_template=[['col_kernel_noc','row_kernel_noc','ch_in_noc','ch_out_noc'], \
                      # ['col_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      # ['row_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      # ['row_out_noc','col_out_noc','ch_out_noc'], \
                      # ]
        
        
        #row stationary2
        self.rs2_noc=[]
        self.rs2_rf_gb_tiling_choices=[]
        self.rs2_rf_gb_tiling_choices_num=[]
        for i in range(len(input_dnn)):
            self.rs2_noc.append([])
            self.rs2_rf_gb_tiling_choices.append([])
            self.rs2_rf_gb_tiling_choices_num.append([])
            if input_dnn[i][2]==0 or input_dnn[i][2]==2:
                ch_in_choices=sorted(r_factors(input_dnn[i][1]['ch_in'][0]))
            ch_out_choices=sorted(r_factors(input_dnn[i][1]['ch_out'][0]))
            col_out_choices=sorted(r_factors(input_dnn[i][1]['col_out'][0]))
            col_kernel_choices=sorted(r_factors(input_dnn[i][1]['col_kernel'][0]))
            
            
            pe_designs=[]
            str1=[]
            if input_dnn[i][2]==0 or input_dnn[i][2]==2:
                for m in ch_in_choices:
                    for l in ch_out_choices:
                        for j in col_out_choices:
                            for ck in col_kernel_choices:
                                if m*l*j*ck<= self.dsp_limit[i]:
                                    pe_designs.append(m*l*j*ck)
                                    str1.append((m,l,j,ck))
                pe_designs=[i[1] for i in sorted(zip(pe_designs,str1),reverse=True)]
            elif input_dnn[i][2]==1:
                for l in ch_out_choices:
                    for j in col_out_choices:
                        for ck in col_kernel_choices:
                            if l*j*ck<= self.dsp_limit[i]:
                                pe_designs.append(l*j*ck)
                                str1.append((l,j,ck))
                pe_designs=[i[1] for i in sorted(zip(pe_designs,str1),reverse=True)]
            #print(pe_designs[1:10])
            
            for pe_design in pe_designs[0:self.pe_array_dim_num]:
                self.rs2_noc[-1].append({})
                self.rs2_rf_gb_tiling_choices[-1].append({})
                self.rs2_rf_gb_tiling_choices_num[-1].append([])
                if input_dnn[i][2]==0 or input_dnn[i][2]==2:
                    self.rs2_noc[-1][-1]['ch_in_noc']=pe_design[0]
                    self.rs2_noc[-1][-1]['ch_out_noc']=pe_design[1]
                    self.rs2_noc[-1][-1]['col_out_noc']=pe_design[2]
                    self.rs2_noc[-1][-1]['col_kernel_noc']=pe_design[3]
                elif input_dnn[i][2]==1:
                    self.rs2_noc[-1][-1]['ch_out_noc']=pe_design[0]
                    self.rs2_noc[-1][-1]['col_out_noc']=pe_design[1]
                    self.rs2_noc[-1][-1]['col_kernel_noc']=pe_design[2]
                if input_dnn[i][2]==0 or input_dnn[i][2]==2:
                    self.rs2_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_in'][0]//self.rs2_noc[-1][-1]['ch_in_noc'],2)
                    self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']))
                
                self.rs2_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_out'][0]//self.rs2_noc[-1][-1]['ch_out_noc'],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']))
                
                self.rs2_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']=factor_n(input_dnn[i][1]['col_out'][0]//self.rs2_noc[-1][-1]['col_out_noc'],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']))
                
                self.rs2_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']=factor_n(input_dnn[i][1]['row_out'][0],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']))
            
                self.rs2_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['col_kernel'][0]//self.ws_noc[-1][-1]['col_kernel_noc'],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']))    

                self.rs2_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['row_kernel'][0],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']))                 

                self.rs2_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']=factor_n(input_dnn[i][1]['batch'][0],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']))

        
    def choices_generation(self,pe_array):
        if pe_array==0:
            return self.ws_rf_gb_tiling_choices_num,self.ws_rf_gb_tiling_choices
    def tiling_translation(self,layer,pe_array,pe_array_dim_choices,rf_gb_choices,rf_gb_choices_order):
        #pe_array 0-3
        #pe_array_dim_choices 0-self.pe_array_dim_num(10)
        #rf_gb_choices n* 7
        #rf_gb_choices_order 0-1 * 7
        #single layer
        #weight stationary
        if pe_array==0:
            rf_gb_tiling_choices=self.ws_rf_gb_tiling_choices
            noc=self.ws_noc
        elif pe_array==1:
            rf_gb_tiling_choices=self.rs2_rf_gb_tiling_choices
            noc=self.rs2_noc
        elif pe_array==2:
            rf_gb_tiling_choices=self.rs1_rf_gb_tiling_choices
            noc=self.rs1_noc
        elif pe_array==3:
            rf_gb_tiling_choices=self.os_rf_gb_tiling_choices
            noc=self.os_noc
        if self.input_dnn[layer][2]==0 or self.input_dnn[layer][2]==2:
            ch_in_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]["ch_in_rf_gb_choices"][rf_gb_choices[0]]
            ch_out_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['ch_out_rf_gb_choices'][rf_gb_choices[1]]
            col_out_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['col_out_rf_gb_choices'][rf_gb_choices[2]]
            row_out_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['row_out_rf_gb_choices'][rf_gb_choices[3]]
            col_kernel_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['col_kernel_rf_gb_choices'][rf_gb_choices[4]]
            row_kernel_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['row_kernel_rf_gb_choices'][rf_gb_choices[5]]
            batch_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['batch_rf_gb_choices'][rf_gb_choices[6]]
            ch_in_tiling=list(permutations(ch_in_rf_gb_choice))[rf_gb_choices_order[0]]
            ch_out_tiling=list(permutations(ch_out_rf_gb_choice))[rf_gb_choices_order[1]]
            col_out_tiling=list(permutations(col_out_rf_gb_choice))[rf_gb_choices_order[2]]
            row_out_tiling=list(permutations(row_out_rf_gb_choice))[rf_gb_choices_order[3]]
            col_kernel_tiling=list(permutations(col_kernel_rf_gb_choice))[rf_gb_choices_order[4]]
            row_kernel_tiling=list(permutations(row_kernel_rf_gb_choice))[rf_gb_choices_order[5]]
            batch_tiling=list(permutations(batch_rf_gb_choice))[rf_gb_choices_order[6]]
        elif self.input_dnn[layer][2]==1:
            ch_out_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['ch_out_rf_gb_choices'][rf_gb_choices[0]]
            col_out_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['col_out_rf_gb_choices'][rf_gb_choices[1]]
            row_out_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['row_out_rf_gb_choices'][rf_gb_choices[2]]
            col_kernel_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['col_kernel_rf_gb_choices'][rf_gb_choices[3]]
            row_kernel_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['row_kernel_rf_gb_choices'][rf_gb_choices[4]]
            batch_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['batch_rf_gb_choices'][rf_gb_choices[5]]
            ch_out_tiling=list(permutations(ch_out_rf_gb_choice))[rf_gb_choices_order[0]]
            col_out_tiling=list(permutations(col_out_rf_gb_choice))[rf_gb_choices_order[1]]
            row_out_tiling=list(permutations(row_out_rf_gb_choice))[rf_gb_choices_order[2]]
            col_kernel_tiling=list(permutations(col_kernel_rf_gb_choice))[rf_gb_choices_order[3]]
            row_kernel_tiling=list(permutations(row_kernel_rf_gb_choice))[rf_gb_choices_order[4]]
            batch_tiling=list(permutations(batch_rf_gb_choice))[rf_gb_choices_order[5]]
        tiling_str=dict(noc[layer][pe_array_dim_choices])
        if self.input_dnn[layer][2] == 0 or self.input_dnn[layer][2]==2:
            tiling_str['ch_in_gb']=ch_in_tiling[0]
        tiling_str['ch_out_gb']=ch_out_tiling[0]
        tiling_str['col_kernel_gb']=col_kernel_tiling[0]
        tiling_str['row_kernel_gb']=row_kernel_tiling[0]
        tiling_str['col_out_gb']=col_out_tiling[0]
        tiling_str['row_out_gb']=row_out_tiling[0]
        tiling_str['batch_gb']=batch_tiling[0]
        if self.input_dnn[layer][2] == 0 or self.input_dnn[layer][2]==2:
            tiling_str['ch_in_dram']=ch_in_tiling[1]
        tiling_str['ch_out_dram']=ch_out_tiling[1]
        tiling_str['col_out_dram']=col_out_tiling[1]
        tiling_str['row_out_dram']=row_out_tiling[1]
        tiling_str['col_kernel_dram']=col_kernel_tiling[1]
        tiling_str['row_kernel_dram']=row_kernel_tiling[1]
        tiling_str['batch_dram']=batch_tiling[1]
        return tiling_str
    
    def tiling_space_partition(self, pe_array,layer,pe_array_dim_choices):
        if pe_array==0:
            return self.ws_rf_gb_tiling_choices_num[layer][pe_array_dim_choices]
        elif pe_array==1:
            return self.rs2_rf_gb_tiling_choices_num[layer][pe_array_dim_choices]
        elif pe_array==2:
            return self.rs1_rf_gb_tiling_choices_num[layer][pe_array_dim_choices]
        elif pe_array==3:
            return self.os_rf_gb_tiling_choices_num[layer][pe_array_dim_choices]
    



def _gcd(l):
    if len(l)==1:
        return l[0]
    def find_gcd(x, y): 
        while(y): 
            x, y = y, x % y 
      
        return x 

      
    num1=l[0] 
    num2=l[1] 
    gcd=find_gcd(num1,num2) 
      
    for i in range(2,len(l)): 
        gcd=find_gcd(gcd,l[i]) 
    return gcd


def dram_invariant_looporder(pe_array, input_lp_order_dram, input_lp_order_gb,mode):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................
    if mode==0 or mode==2:
        if not (len(input_lp_order_gb)==len(set(input_lp_order_gb)) and len(input_lp_order_dram)==len(set(input_lp_order_dram))):
            raise Exception('Please provide lp_order with no duplicate elements')
        input_rf=noc_template[pe_array]
        lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
        lp_order_template=['ch_out_gb','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb']
        lp_order_string=[]
        for i in range(len(lp_order_template)):
            lp_order_string.append(lp_order_template[input_lp_order_gb[i]])
        for i in range(len(lp_order_template_dram)):
            lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    elif mode==1:
        if not (len(input_lp_order_gb)==len(set(input_lp_order_gb)) and len(input_lp_order_dram)==len(set(input_lp_order_dram))):
            raise Exception('Please provide lp_order with no duplicate elements')
        input_rf=noc_template_dw[pe_array]
        lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
        lp_order_template=['ch_out_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb']
        lp_order_string=[]
        for i in range(len(lp_order_template)):
            lp_order_string.append(lp_order_template[input_lp_order_gb[i]])
        for i in range(len(lp_order_template_dram)):
            lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    return copy.deepcopy(input_rf)+copy.deepcopy(lp_order_string)
    
def performance_feedback(tiling1,pe_array,pe_array_dim_choices,param,tmp_hw_spec,mode,layer):
    lp_order_string=dram_invariant_looporder(pe_array, param[0:7], param[7:14],mode)
    tiling_string=tiling1.tiling_translation(layer,pe_array,pe_array_dim_choices,param[14:21],param[21:28])
    p=life_eval(tiling_string,1,tmp_hw_spec,df_order=lp_order_string)
    return 4*p
    
    
def translate_raw_param(tiling1,pe_array,pe_array_dim_choices,param,tmp_hw_spec,mode,layer):
    lp_order_string=dram_invariant_looporder(pe_array, param[0:7], param[7:14],mode)
    tiling_string=tiling1.tiling_translation(layer,pe_array,pe_array_dim_choices,param[14:21],param[21:28])
    return lp_order_string,tiling_string, tmp_hw_spec




def hw_consumption(lp_order_string,tiling_string,hw_spec,stride):
    noc_consumption=1
    for i in tiling_string:
        if 'noc' in i:
            noc_consumption*=tiling_string[i]
    tmp_hw_spec=copy.deepcopy(hw_spec)
    ep=tmp_hw_spec['gb_vol']
    tmp_hw_spec['gb_vol']=int(tmp_hw_spec['gb_vol']/2)
    mp=tmp_hw_spec['gb_vol']
    sp=0
    p=life_eval(tiling_string,1,hw_spec,df_order=lp_order_string)
    while (ep-sp)>100 or (not p[1]):
        if p[1]:
            tmp_hw_spec['gb_vol']=math.floor((sp+mp)/2)
            ep=mp
            mp=tmp_hw_spec['gb_vol']
            p=life_eval(tiling_string,stride,tmp_hw_spec,df_order=lp_order_string)
        else:
            tmp_hw_spec['gb_vol']=math.ceil((mp+ep)/2)
            sp=mp
            mp=tmp_hw_spec['gb_vol']
            p=life_eval(tiling_string,stride,tmp_hw_spec,df_order=lp_order_string)
    return noc_consumption/tmp_hw_spec['num_pe'],tmp_hw_spec['gb_vol']/hw_spec['gb_vol']
    
# df_order=['row_out_noc', 'col_out_noc', 'ch_out_noc', 'batch_gb',  'col_out_gb', 'ch_out_gb', 'row_out_gb', 'row_kernel_gb', 'col_kernel_gb', 'ch_in_gb','col_kernel_dram', 'ch_out_dram', 'ch_in_dram', 'row_kernel_dram', 'col_out_dram', 'batch_dram', 'row_out_dram']
# df_dict={'ch_out_noc': 16, 'col_out_noc': 1, 'row_out_noc': 8, 'ch_in_gb': 128, 'ch_out_gb': 1, 'col_kernel_gb': 1, 'row_kernel_gb': 3, 'col_out_gb': 1, 'row_out_gb': 1, 'batch_gb': 1, 'ch_in_dram': 1, 'ch_out_dram': 16, 'col_out_dram': 56, 'row_out_dram': 7, 'col_kernel_dram': 3, 'row_kernel_dram': 1, 'batch_dram': 1}
# hw_spec={'gb_vol': 2097152, 'rf_vol': 512, 'num_pe': 144, 'num_rf': 144}
#
# print(hw_consumption(df_order,df_dict,hw_spec,1))