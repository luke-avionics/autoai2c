from random import random, randint,shuffle
import numpy as np
import test_for_eyeriss as simnas
import time
from itertools import combinations,permutations
import copy
from  multiprocessing import Queue
import multiprocessing
import math
from itertools import permutations,combinations

default_hw={ \
    'gb_vol':108*1024*8, \
    'rf_vol':6893, \
    'num_pe':168, \
    'num_rf':168
}

##############################
#shared util funcs
#############################

def life_eval(actions,stride,hw_spec,df_order=None):
    #function to query chip_estimator and get energy+latency feedback

    #actions: tiling factors for a specific loop-order
    #stride: the stride number for this CONV layer operation
    #hw_spec: hw specs for evaluation
    #df_order: loop-order for evaluation 
    #           !!!!if not provided PLS provide it in chip_estimator
    #           !!!!legacy functionality, so always try to provide specific loop-order here
    try:
        #input isolation
        input_actions=dict(actions)
        if df_order:
            input_df_order=list(df_order)
        else:
            input_df_order=None
        ene_results=simnas.sample_energy(input_actions,stride,hw_spec,input_df_order=input_df_order)
        penalty=(ene_results[0]*1e-8, ene_results[1]*100)
        buffer_not_exceed=True
        #print(ene_results[0],ene_results[1])
    #if design hw constraint exceeded,
    #if exceeded return extremely large penalty
    except Exception as e:
        if 'resource' in str(e):
            print('error:', e)
            pass
        else:
            print('error:',e)
            print(actions)
            print(df_order)
        penalty=(9e12,9e12)                                  #very strong penalty to over budget
        buffer_not_exceed=False
    return penalty, buffer_not_exceed


#noc_template to be considered 
noc_template=[['col_kernel_noc','row_kernel_noc','ch_in_noc','ch_out_noc'], \
                      ['col_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      ['row_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      ['row_out_noc','col_out_noc','ch_out_noc'], \
                      ]

rf_noc_template=noc_template


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

