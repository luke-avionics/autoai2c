from ev_util import *
from ev_dict_object import *




def ene_lat_qury(dnn, cycles=20):
    #hw settup
    hw_spec={ \
    'gb_vol':108*1024*8, \
    'rf_vol':512*8, \
    'num_pe':168, \
    'num_rf':168
    }
    df_order=['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf', 'ref_rf_we',\
                'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc',\
                'ref_gb_we', 'ch_out_gb', 'ref_gb_in',  'ch_in_gb', 'ref_gb_out', \
                'col_out_dram', 'ch_out_dram', 'batch_dram'\
             ]
    stride=dnn[0][0]



    #generate reference df_order
    ref_df_order=[]
    for i in df_order:
        if 'ref' not in i:
            ref_df_order.append(i)
    #generate net_arch
    net_arch=gen_net_arch(ref_df_order,dnn)
    #scae=led towards the cpu server: 320 threads
    ev_dict1=ev_dict([stride],net_arch,ref_df_order,max_pop=320,true_df_order=[df_order],hw_spec=hw_spec)
    #optimize for n cycles
    ev_dict1.search(n=cycles,init_multiplier=3)       #TODO: add search for n cycles or search for convergence?
    
    best_dict=ev_dict1.best_dict[0]
    #final query for energy and latency 
    input_actions=dict(best_dict)
    input_df_order=list(df_order)
    ene_results=simnas.sample_energy(input_actions,stride,hw_spec,input_df_order=input_df_order)
    #energy latency
    return ene_results[0], ene_results[1]




#example usage
dnn=[[4, {'ch_out':[96,0],'ch_in':[3,0],'batch':[4,0],'col_out':[55,0],'row_out':[55,0],'row_kernel':[11,0],'col_kernel':[11,0]}]]
results=ene_lat_qury(dnn)
print(results)
