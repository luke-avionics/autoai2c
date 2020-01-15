# autoai2c

The following will be an example to generate AlexNet example (on ***eic-cpu server !!!***) 
*(does not include cycle parameter tuning, but these parameters should be generic)*

**The flow can be divided into the following three parts** 
1. find the most suitable hardware spec: GB_VOL RF_VOL PE_num RF_num (PE_num=RF_num)
2. for each layer of CONV find the best rf-noc-template for the found hw_spec above, to form the rf_noc_template pool
3. optimize looporder+tiling factor specifically for the found hw_spec and in the scope of rf_noc_template pool (stage 3+4) 



***HW spec*** 
1. Go to **pre_rf** folder
2. comment out the standalone(not in the function) print command in *ev_util.py* 
3. open *ev_combined_no_hw.py*, go to line 243 
4. un-comment the definition for input_dnn; (u can only do one layer here to save time, but always be consistent in throughout the process)
    - the number 0 in the value list does not mean anything 
    - Will be modfified accordingly by *gen_net_arch(df_order,dnn)* in *ev_util.py* 
5. go to line 408; make sure stride_list is [4,1,1,1,1] or according to specific DNN model 
6. go to line 466 and 467; make sure the hw_pool is indexed towards *most_demanding_layer* 
7. go to line 473; make sure entire hw_pool is in consideration (no indexing) 
8. go to line 477; make sure hw_search=True 
9. go to line 775; make sure work_load is distributed upto hw_pool[... , 40 ] 
10. run ev_combined_no_hw.py; be sure to keep the log. 
11. go to the last line of the log: 
    - like: hardware-scores:  [-173.80162416640002, -159.07083724800003, -158.56037734400002, -153.24673078613336, -161.43945891840002, -138.36864675840002, -160.03449907200005, -144.8454766592, -149.83802880000002, -139.7495863296, -143.0273212416, -154.50193776640003, -128.95242383360002, -148.69871616, -135.54062110719968, -130.50038271999978, -140.2366384128001, -124.77900195839982, -127.67043010559973, -137.8449489919999, -139.71273943039975, -130.2370781183997, -162.88507699199954, -154.8419512319997, -159.2962113535998, -136.7375626239995, -124.38423306239972, -134.2984425471996, -153.11986237439956, -124.73595064319983, -110.89143823359974, -147.878060714666, -143.105809749333, -151.77733611519983, -123.60115384319967, -119.56602623999939, -122.74980901546579, -140.60113817599915, -138.89937578666633, -123.25950873599976]
    - find the higest score and keep the index, as hw index
    
    
    
***rf-noc_rf_template for each layer*** 
1. go back to *ev_combined_no_hw.py*
2. input_dnn, input_stride_list(line 408) will be un-changed from the previous part
3. We will find the best noc_rf_template for the first layer 
    - go to line 466 and 467; make sure the *input_dnn, input_stride_list* is indexed towards ***0***
4. go to line 473; index *hw_pool* to best one found in the previous part (like line 476)
    - right now, we only consider the best one for simplicity
    - if want to add more, simply do it like in line 474
5. go to line 787; distribute the workload up to [ 0,hw_pool[0:1] ]; instead of up to 40 like previous part 
6. run *ev_combined_no_hw.py*; be sure to keep the log. 
7. go to the log file 
    - search for **Highest rf**, note down the template
    - in the end of the log, search for **best loop order**, and note it down
    - go to line 227 of **ev_util.py**, use sample_results_df() to translate results in string format
        - replace the first argument with **best loop order** you note down
        - replace the second argument with **Highest rf** you note down
        - note down the output as one element of **rf_noc_template** used later
8. go to back step ***3*** to change the index to the following layers  (0,1,2,3,4...)
    - you can change index and run for multiple layers at a single time (nohup... > log.n &), but be sure not to run more than 3 pieces at a single time, otherwise CPU will overload.
    - and do step ***7*** in the end all at once



***looporder+tiling factors (post_rf)***
1. go to **post_rf** folder
2. open *ev_util.py*, and go to line 219, add/replace **rf_noc_template** with the new one comprising the elements you note down in the previous part
    - one tip to encourage more exploration, is to gradually accumlate templates from mulitple runs and change the multiprocessing workload later
    - But **ignore** it for now 
3. open *ev_combined_no_hw.py*, go to line 566, make sure the layer indexing is commented out, we want to do all layer optimization here
4. go to line 251 and 421,  make sure *input_dnn, input_stride_list* are the same as the previous part in **pre_rf**.
5. go to line 575; make sure only the hw indices you note down in the first part are in consideration 
6. go to line 721; change the workload to cover the range up to 5.
    - the range is determined by the size of **rf_noc_template** you have, for now it is 5.
7. go to line 865; change the workload to cover the hw_spec range up to 1.
    - the range is determined by how many sets of hw specs you want to consider here, for now let's keep it 1. 
8. run *ev_combined_no_hw.py*; be sure to keep the log 
9. go to the log file: 
    - go to the end of the file, **hardware-scores** will denote the performance of each set of hw_spec you consider
    - find the highest score and search for the score, in format **current best score : < score you note down >**
        - its corresponding hw index in **hardware-scores** will also denote the ***final best set of hw_spec***
    - right above the line you searched and found, note down the **best loop order** and **best config dict**
    - search for **Highest rf**, and note down the template element(the inner list) whose index corresponding to the best peforming hw_spec's you find in **hardware-scores** 
    - go to line 281 of *ev_util.py*, use *arch_sample_results_df()* to get the final loop-orders for all layers in string format.
        - replace the first argument with the number of layers you consider
        - replace the second argument with **best loop order** you found
        - replace the third argument with the template element you found in **Highest rf**, not the entire **Highest rf** !!!!
        - run and get the final results, this will be the final results for ***best loop orders***
    - the **best config dict** you note down above will be the final results for the corresponding tiling factors






***The following will be general outline description of the code*** 
1. pre-rf and post-rf share most of the code except some setup in **ev_util.py** and **ev_combined_no_hw.py** 
    - pre-rf will decide the hardware specs and rf_noc_template pool to be fined tuned later in post-rf 
2. **ev_util.py** is where most of the utility functions reside while the rest of them can be found in corresponding **ev_combined_no_hw.py** 
3. **ev_dict_object.py** will be the capsulated code structure(using genetic) to optimize the **tiling factors**, given **hardware specs**, **rf_noc_template** and **looporders** 
    - It is a very clear illusration of 
        - genetic algorithm 
        - distribution of tasks to different cores 
    - The feedback is realized from direct querying chip_estimator( **test_for_eyeriss.py** ) 
        - see arch_life(...) and life_eval(...) for more info 
4. **ev_combined_no_hw.py** is built upon the code of **ev_dict_object.py**, to find the most suitable **hardware specs**, **rf_noc_template**, **looporders** and final fine tuning of **tiling factors** 
    - **hardware specs** will be firstly filtered to eliminate the unreasonable choices (too small or above the budget), and then chosen through the coarse optimization of the rest of the parameters 
        - it will be done on the most_demanding_layer identified earlier 
    - **rf_noc_template**: because this set of parameters are fixed for all layers, it will be tricky to combine it with the rest of the layerwise parameters --> without this set of parameters, we can simply find the best layerwise parameters for each layer due to the greedy nature. 
        - Therefore, I intend to separate them: Firstly, shrinking the thousands of choices of **rf_noc_template** to a pool with size: currently the number of input DNN layers. Then, do a exhaustive  iteration for on top of the optimization for the layerwise parameters 
        - To form the pool, I found the best **rf_noc_template** for each layer: see ***rf-noc_rf_template for each layer***  above for more info. 
    - **looporders** optimization is just a more complicated(messy) version of **ev_dict_object.py**, with core difference in feedback functions: 
        - random_life(...)   in **ev_combined_no_hw.py** 
        - fine_tune(...)     in **ev_combined_no_hw.py** 
        - But the whole structure is staged for accommodation of the above two optimization 
    - **tiling factors** fine_tuning: when everything is fixed, tiling factors are further optimized more thoroughly: increasing cycle parameter in fine_tune(...)
5. **test_for_eyeriss.py** is the config for chip_estimator, where we set technology dependent variables and the optimizer code query from. 
6. The rest of the code are mostly supporting library for the chip_estimator 





