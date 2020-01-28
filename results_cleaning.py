df_config_dict=[{'ch_out_rf': 3, 'ch_out_noc': 32, 'ch_out_gb': 1, 'ch_out_dram': 1, 'ch_in_rf': 3, 'ch_in_gb': 1, 'ch_in_dram': 1, 'batch_rf': 2, 'batch_gb': 2, 'batch_dram': 1, 'col_out_rf': 4, 'col_out_noc': 1, 'col_out_gb': 2, 'col_out_dram': 7, 'row_out_rf': 1, 'row_out_noc': 8, 'row_out_gb': 7, 'row_out_dram': 1, 'row_kernel_rf': 1, 'row_kernel_gb': 11, 'row_kernel_dram': 1, 'col_kernel_rf': 11, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 4, 'ch_out_noc': 8, 'ch_out_gb': 1, 'ch_out_dram': 8, 'ch_in_rf': 8, 'ch_in_gb': 1, 'ch_in_dram': 6, 'batch_rf': 2, 'batch_gb': 1, 'batch_dram': 2, 'col_out_rf': 9, 'col_out_noc': 3, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_out_noc': 9, 'row_out_gb': 1, 'row_out_dram': 3, 'row_kernel_rf': 1, 'row_kernel_gb': 5, 'row_kernel_dram': 1, 'col_kernel_rf': 5, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 3, 'ch_out_noc': 16, 'ch_out_gb': 1, 'ch_out_dram': 8, 'ch_in_rf': 16, 'ch_in_gb': 16, 'ch_in_dram': 1, 'batch_rf': 2, 'batch_gb': 1, 'batch_dram': 2, 'col_out_rf': 13, 'col_out_noc': 1, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_out_noc': 13, 'row_out_gb': 1, 'row_out_dram': 1, 'row_kernel_rf': 1, 'row_kernel_gb': 3, 'row_kernel_dram': 1, 'col_kernel_rf': 3, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 3, 'ch_out_noc': 16, 'ch_out_gb': 1, 'ch_out_dram': 8, 'ch_in_rf': 16, 'ch_in_gb': 4, 'ch_in_dram': 3, 'batch_rf': 1, 'batch_gb': 4, 'batch_dram': 1, 'col_out_rf': 13, 'col_out_noc': 1, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_out_noc': 13, 'row_out_gb': 1, 'row_out_dram': 1, 'row_kernel_rf': 1, 'row_kernel_gb': 3, 'row_kernel_dram': 1, 'col_kernel_rf': 3, 'col_kernel_gb': 1, 'col_kernel_dram': 1}, {'ch_out_rf': 8, 'ch_out_noc': 16, 'ch_out_gb': 1, 'ch_out_dram': 2, 'ch_in_rf': 4, 'ch_in_gb': 24, 'ch_in_dram': 2, 'batch_rf': 1, 'batch_gb': 1, 'batch_dram': 4, 'col_out_rf': 13, 'col_out_noc': 1, 'col_out_gb': 1, 'col_out_dram': 1, 'row_out_rf': 1, 'row_out_noc': 13, 'row_out_gb': 1, 'row_out_dram': 1, 'row_kernel_rf': 1, 'row_kernel_gb': 3, 'row_kernel_dram': 1, 'col_kernel_rf': 1, 'col_kernel_gb': 3, 'col_kernel_dram': 1}]


df_order=[['row_kernel_rf', 'col_out_rf', 'row_out_rf', 'batch_rf', 'ref_rf_we', 'ch_out_rf', 'col_kernel_rf', 'ref_rf_out', 'ref_rf_in', 'ch_in_rf', 'row_out_noc', 'col_out_noc', 'ch_out_noc', 'row_kernel_gb', 'col_kernel_gb', 'ref_gb_in', 'ref_gb_out', 'ch_out_gb', 'col_out_gb', 'batch_gb', 'ch_in_gb', 'ref_gb_we', 'row_out_gb', 'row_kernel_dram', 'col_kernel_dram', 'col_out_dram', 'ch_out_dram', 'row_out_dram', 'ch_in_dram', 'batch_dram'],\
['row_kernel_rf', 'col_out_rf', 'row_out_rf', 'batch_rf', 'ref_rf_we', 'ch_out_rf', 'col_kernel_rf', 'ref_rf_out', 'ref_rf_in', 'ch_in_rf', 'row_out_noc', 'col_out_noc', 'ch_out_noc', 'row_kernel_gb', 'ref_gb_we', 'col_kernel_gb', 'ref_gb_in', 'batch_gb', 'col_out_gb', 'ch_in_gb', 'ref_gb_out', 'row_out_gb', 'ch_out_gb', 'ch_in_dram', 'row_kernel_dram', 'batch_dram', 'col_out_dram', 'ch_out_dram', 'col_kernel_dram', 'row_out_dram'],\
 ['row_kernel_rf', 'col_out_rf', 'row_out_rf', 'batch_rf', 'ref_rf_we', 'ch_out_rf', 'col_kernel_rf', 'ref_rf_out', 'ref_rf_in', 'ch_in_rf', 'row_out_noc', 'col_out_noc', 'ch_out_noc', 'col_kernel_gb', 'col_out_gb', 'ref_gb_out', 'ref_gb_we', 'ch_out_gb', 'batch_gb', 'row_out_gb', 'row_kernel_gb', 'ref_gb_in', 'ch_in_gb', 'ch_in_dram', 'row_kernel_dram', 'row_out_dram', 'ch_out_dram', 'col_kernel_dram', 'col_out_dram', 'batch_dram'],\
['row_kernel_rf', 'col_out_rf', 'row_out_rf', 'batch_rf', 'ref_rf_we', 'ch_out_rf', 'col_kernel_rf', 'ref_rf_out', 'ref_rf_in', 'ch_in_rf', 'row_out_noc', 'col_out_noc', 'ch_out_noc', 'col_kernel_gb', 'row_kernel_gb', 'batch_gb', 'ref_gb_out', 'ref_gb_we', 'col_out_gb', 'ref_gb_in', 'ch_in_gb', 'row_out_gb', 'ch_out_gb', 'row_kernel_dram', 'col_kernel_dram', 'row_out_dram', 'col_out_dram', 'ch_in_dram', 'batch_dram', 'ch_out_dram'],\
 ['row_kernel_rf', 'col_out_rf', 'row_out_rf', 'batch_rf', 'ref_rf_we', 'ch_out_rf', 'col_kernel_rf', 'ref_rf_out', 'ref_rf_in', 'ch_in_rf', 'row_out_noc', 'col_out_noc', 'ch_out_noc', 'row_kernel_gb', 'col_kernel_gb', 'ref_gb_in', 'ref_gb_out', 'ref_gb_we', 'col_out_gb', 'row_out_gb', 'ch_out_gb', 'ch_in_gb', 'batch_gb', 'ch_in_dram', 'row_out_dram', 'batch_dram', 'row_kernel_dram', 'ch_out_dram', 'col_kernel_dram', 'col_out_dram']]



def clean_results(df_config_dict,df_order):
    size_consistency=True
    tmp_len=len(df_order[0])    
    for layer in df_order:
        if tmp_len!=len(layer):
            size_consistency=False
            break
    print('size check: ', size_consistency)
    new_df_order=[]
    new_df_config_dict=[]
    for i in range(len(df_config_dict)):
        new_df_order.append([])
        new_df_config_dict.append({})

    for item in df_order[0]:
         for i in range(len(df_config_dict)):
                        
        
