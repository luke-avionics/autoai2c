# autoai2c

The following will be an example to generate AlexNet example (on eic-cpu server) 

**The flow can be divided into the following three parts** 
1. find the most suitable hardware spec: GB_VOL RF_VOL PE_num RF_num (PE_num=RF_num)
2. for each layer of CONV find the best rf-noc-template for the found hw_spec above, to form the rf_noc_template pool
3. optimize looporder+tiling factor specifically for the found hw_spec and in the scope of rf_noc_template pool (stage 3+4) 



***HW spec*** 
1. Go to pre_rf folder
2. comment out the standalone(not in the function) print command in *ev_util.py* 
3. open *ev_combined_no_hw.py*, go to line 243 
4. un-comment the definition for input_dnn; 
    - the number 0 in the value list does not mean anything 
    - Will be modfified accordingly by *gen_net_arch(df_order,dnn)* in *ev_util.py* 
5. go to line 466 and 467; make sure the hw_pool is indexed towards *most_demanding_layer* 
6. go to line 473; make sure entire hw_pool is in consideration (no indexing) 
7. go to line 477; make sure hw_search=True 
8. go to line 775; make sure work_load is distributed upto hw_pool[... , 40 ]
9. run ev_combined_no_hw.py; be sure to keep the log.


