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

        
class ev_dict():
    def __init__(self,stride_list,net_arch,df_order,prop_m=0.5,max_pop=600,dying_rate=0.2,true_df_order=None,hw_spec=default_hw):
        self.prop_m=prop_m                                                              #mutation probability
        self.max_pop=max_pop
        self.dying_rate=dying_rate=0.2                                                          #the dying_rate is also aimed to allowing only elites to have children
        self.stride_list=stride_list
        self.net_arch=net_arch
        self.df_order=df_order
        self.true_df_order=true_df_order
        self.best_score=None
        self.hw_spec=hw_spec

        
        #generate factor list according to a certain config_dict
        self.arch_factor_list=arch_factor_list_dict(self.net_arch)
        self.space_size=1
        #calculate the space size 
        for layer in self.arch_factor_list:
            for data_dim in layer:
                self.space_size*=len(layer[data_dim])   

    def exhuastive_search(self):
        # !!!  can now only be used when passed with one single layer of CONV
        # !!!  assume 7 data dim

        best=-9000000000000.0
        print_freq=100
        for layer in self.arch_factor_list:
            data_dims=list(layer.keys())
            #number of choices for each dimension
            data_dim_limits=[]
            freq_ctr=0
            for d0 in layer[data_dims[0]]:
                for d1 in layer[data_dims[1]]:
                    for d2 in layer[data_dims[2]]:                  
                        for d3 in layer[data_dims[3]]:
                            for d4 in layer[data_dims[4]]:
                                for d5 in layer[data_dims[5]]:
                                    for d6 in layer[data_dims[6]]:
                                        freq_ctr+=1
                                        df_dict={}
                                        value_list=[d0,d1,d2,d3,d4,d5,d6]
                                        for i in range(7):
                                            value=value_list[i]
                                            key=data_dims[i]
                                            ctr=0
                                            for sub_key in self.df_order[0]:
                                                if key in sub_key:
                                                    df_dict[sub_key]=value[ctr]
                                                    ctr+=1 
                                        tmp=arch_life([df_dict],self.stride_list,self.hw_spec,df_order=self.true_df_order)[0]
                                        if tmp>best:
                                            best=tmp
                                        if freq_ctr%print_freq==0:
                                            print(best)
 
    def search(self,n=10000,mutate_init=False,init_multiplier=3):
        pop_list=[]
        #generate initial population
        if not mutate_init:
            for i in range(0,int(self.max_pop*init_multiplier)):
                pop_list.append(arch_random_pop(self.net_arch,self.df_order,self.arch_factor_list))
        else:
            pass
            
        #get the score for the initial population
        #print('evaluating the initial population')                            
        score_board=[]                                                     
        dump_yard=[]
        args1=list(range(0,len(pop_list)))
        output_q=Queue()
        def func1(i):
            #merged_child=merge_noc(pop_list[i],noc)
            merged_child=pop_list[i]
            try:
                output_q.put((i,arch_life(merged_child,self.stride_list,self.hw_spec,df_order=self.true_df_order)[0]),False)
            except Empty:
                raise Exception("There is no room in the queue in initial tiling factor evaluation stage")
        num_worker_threads=int(multiprocessing.cpu_count())
        multi_p(func1,args1,output_q,num_worker_threads,dump_yard)
        dump_yard=sorted(dump_yard)
        for idx in range(len(dump_yard)):
            score_board.append(dump_yard[idx][1])           
        pop_list,score_board=pop_ranking(pop_list,score_board)

        score1=0
        for _ in range(n):
            #if not saturate birth and mutate
            if len(pop_list) < self.max_pop:
                #distributing tasks to cores
                num_worker_threads=int(multiprocessing.cpu_count())
                #print('generating pop')
                #size of newly generated population
                size=int(2*self.dying_rate*(self.max_pop))                                   # right now birth control is through max_pop; can be done through current pop
                if size%2 !=0:
                    size+=1
                #no task imbalance among cores
                if num_worker_threads > (size/2):
                    num_worker_threads=int(size/2)
                if ((size/2)%num_worker_threads)!=0:
                    size=int(math.ceil((size/2.0)/num_worker_threads)*num_worker_threads*2)
                #print('new born size:',size)
                pos=list(range(0,size*2))   #encourage exploration
                shuffle(pos,random=random)
                pos=pos[0:size]

                score_pair=Queue()
                #pos needs to be changed
                def work(i):
                    new_child=arch_give_birth(pop_list[pos[i]],pop_list[pos[i+1]],self.net_arch)        #will try no birth only mutate next                          
                    new_child=arch_mutate(new_child,self.prop_m,self.arch_factor_list)            
                    #get the scores for the new born
                    #merged_child=merge_noc(new_child,noc)
                    merged_child=new_child
                    try:
                        score_pair.put((compress_dict(new_child),arch_life(merged_child,self.stride_list,self.hw_spec,df_order=self.true_df_order)[0]),False)
                    except Empty:
                        raise Exception("There is no room in the queue in tiling factor update stage")
                run_ites=int((len(pos)/2)//num_worker_threads)

                #how many runs we need 
                for run_ite in range(run_ites):
                    processes = [multiprocessing.Process(target=work, args=([i])) for i in range(2*run_ite*num_worker_threads,2*(run_ite+1)*num_worker_threads,2)]
                    #print(len(processes))
                    #print('queue size: ',score_pair.qsize())
                    for p in processes:
                        p.start()

                    while not score_pair.empty():
                        pair=score_pair.get()
                        pop_list.append(decompress_dict(copy.deepcopy(pair[0]),pop_list[-1]))
                        score_board.append(pair[1])

                    for p in processes:
                        p.join()
                #clear score_pair in the queue to global
                while not score_pair.empty():
                    pair=score_pair.get()
                    pop_list.append(decompress_dict(copy.deepcopy(pair[0]),pop_list[-1]))
                    score_board.append(pair[1]) 
            #else kill and birth and mutate
            else:
                #print(score_board)
                #print('killing')
                #rank
                pop_list,score_board=pop_ranking(pop_list,score_board)
                #kill
                pop_list = pop_list[0:int((self.max_pop)*(1-self.dying_rate))]
                score_board=score_board[0:int((self.max_pop)*(1-self.dying_rate))]
                score1=score_board[0]
                self.best_score=score1
                self.best_dict=copy.deepcopy(pop_list[0])
                self.best_layer_breakdown=arch_life(pop_list[0],self.stride_list,self.hw_spec,df_order=self.true_df_order)[1]
                #print(self.best_layer_breakdown)
                #print('highest score: ',score1)
                #print(pop_list[0])
                #make sure nothing is wrong in the sorting process
                #print(arch_life(pop_list[0],self.stride_list,df_order=self.true_df_order)[0]==score_board[0])
                #print('energy_breakdown: ',simnas.sample_energy(pop_list[0][0],self.stride_list[0])[2])





                                                            

 
