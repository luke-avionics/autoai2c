#include "fpga_module.h"



void buffer_bram(fixed *src, fixed* dest, unsigned long long src_size, int buffer_size){
    int i,j=0;
    int num_buffers = (int) (src_size/buffer_size);
    for (i=0;i<num_buffers;i++){
    	int chunk_size=buffer_size;
        if ((i+1)*buffer_size>src_size)
            chunk_size = src_size-i*buffer_size;
        // Burst read of v1 and v2 vector from DDR memory
        // Utilize multiple interfaces to access data concurrently
        for (int j = 0 ; j < chunk_size ; j++){
        #pragma HLS PIPELINE
        	dest[i][j] = src[i*buffer_size + j];
        }
    }
}


//first is dram the second bram
void write_b2_dram(fixed *src, fixed* dest, unsigned long long src_size, int buffer_size){
    int i,j=0;
    int num_buffers = (int) (src_size/buffer_size);
    for (i=0;i<num_buffers;i++){
    	int chunk_size=buffer_size;
        if ((i+1)*buffer_size>src_size)
            chunk_size = src_size-i*buffer_size;
        // Burst read of v1 and v2 vector from DDR memory
        // Utilize multiple interfaces to access data concurrently
        for (int j = 0 ; j < chunk_size ; j++){
        #pragma HLS PIPELINE
            src[i*buffer_size + j] = dest[i][j]; 
        }
    }    

}


//current state
// all weights buffered up
// every single channel of output buffered up





//conv(l4,param_l5,l5,1,1,256,128,3,56,56,1,1);
int conv(fixed* conv_in, fixed* params, fixed* conv_out, int have_bn, int have_relu, int ch_out, int ch_in, int kernel_size, int size_out, int size_in, int stride, int padding){
    int P;
    

    //buffering params
    int tmp=0;
    unsigned long long params_index=0;


    int num_buffers=73;
    unsigned long long num_params=295680;
    fixed buffered_params[73][BUFFER_SIZE];
 
    buffer_bram(params, &buffered_params[0][0],num_params,4096);
    /*
    int i,j=0;
    for (i=0;i<num_buffers;i++){
    	int chunk_size=BUFFER_SIZE;
        if ((i+1)*BUFFER_SIZE>num_params)
            chunk_size = num_params-i*BUFFER_SIZE;
        // Burst read of v1 and v2 vector from DDR memory
        // Utilize multiple interfaces to access data concurrently
        for (int j = 0 ; j < chunk_size ; j++){
        #pragma HLS PIPELINE
        	buffered_params[i][j] = params[i*BUFFER_SIZE + j];
        }
    }
    */



    if (padding==0){
        P =0;
    }
    else{
        P = ((int)kernel_size - 1)/2;
    }

    //initialize out buffer
    fixed buffered_out[1][4096]={0};


    int id_img, id_ch_in, id_ch_out, id_output_row, id_output_col, id_kernel_row, id_kernel_col;
    int start_col, start_row, idx_input, output_idx;
    fixed result;
    fixed input_val;
    id_img = 0;
        for (id_ch_out = 0; id_ch_out < ch_out; id_ch_out++) {


            //refresh output channel
            unsigned int output_idx_offset=id_ch_out * size_out * size_out;
                

            for (id_output_row=0;id_output_row<size_out;id_output_row++){
                for (id_output_col=0;id_output_col<size_out;id_output_col++) {
                    result = 0;
                    start_row = id_output_row * stride - P;
                    start_col = id_output_col * stride - P;
                    //TODO: check the output_idx here!!
                    output_idx =id_img * ch_out * size_out * size_out  + id_output_row * size_out + id_output_col;
                    for (id_kernel_row = 0; id_kernel_row < kernel_size; id_kernel_row++) {
                        for (id_kernel_col = 0; id_kernel_col < kernel_size; id_kernel_col++) {
                            if (start_row + id_kernel_row < 0 or start_col + id_kernel_col < 0 or start_row + id_kernel_row >= size_in or start_col + id_kernel_col >= size_in) {
                                    result += 0;
                            }
                            else{
                                for (id_ch_in = 0; id_ch_in < ch_in; id_ch_in++) {
                                    idx_input = id_img * ch_in * size_in * size_in + id_ch_in *size_in * size_in + (start_row + id_kernel_row) * size_in + (start_col + id_kernel_col);
                                    input_val = conv_in[idx_input];
                                    //std::cout<<"fuck "<<input_val<<std::endl;
                                    params_index=id_ch_out * ch_in * kernel_size * kernel_size + id_ch_in * kernel_size * kernel_size + id_kernel_row * kernel_size + id_kernel_col;
                                    tmp= (int)(params_index-(int)(params_index/BUFFER_SIZE)*BUFFER_SIZE);
                                    result += input_val * buffered_params[(int)(params_index/BUFFER_SIZE)][tmp];
                                    //result += input_val * params[id_ch_out * ch_in * kernel_size * kernel_size + id_ch_in * kernel_size * kernel_size + id_kernel_row * kernel_size + id_kernel_col];
                                }
                            }
                        }
                    }
                    params_index=ch_out*ch_in*kernel_size*kernel_size+id_ch_out;
                    tmp= (int)(params_index-(int)(params_index/BUFFER_SIZE)*BUFFER_SIZE);
                    result += buffered_params[(int)(params_index/BUFFER_SIZE)][tmp];
                    //result += params[ch_out*ch_in*kernel_size*kernel_size+id_ch_out];
                    if (have_bn==1){
                        params_index=ch_out*ch_in*kernel_size*kernel_size+ch_out+2*id_ch_out;
                        tmp= (int)(params_index-(int)(params_index/BUFFER_SIZE)*BUFFER_SIZE);
                        result *= buffered_params[(int)(params_index/BUFFER_SIZE)][tmp];
                        //result *= params[ch_out*ch_in*kernel_size*kernel_size+ch_out+2*id_ch_out];

                        params_index=ch_out*ch_in*kernel_size*kernel_size+ch_out+2*id_ch_out+1;
                        tmp= (int)(params_index-(int)(params_index/BUFFER_SIZE)*BUFFER_SIZE);
                        result += buffered_params[(int)(params_index/BUFFER_SIZE)][tmp];
                        //result += params[ch_out*ch_in*kernel_size*kernel_size+ch_out+2*id_ch_out+1];
                    }
                    if (result <0 and have_relu==1){
                    	buffered_out[0][output_idx] =0;
                        //conv_out[output_idx]=0;
                    }
                    else{
                    	buffered_out[0][output_idx] =result;
                        //conv_out[output_idx]=result;
                    }
                }
            }



            //refresh output channel
            write_b2_dram(conv_out+output_idx_offset, &buffered_out[0][0], 56*56,BUFFER_SIZE);
        }
	return 0;
}




int fc(fixed* fc_in, fixed* params, fixed * fc_out, int have_bn, int have_relu, int have_bias, int ch_out, int ch_in){
    int i,j;
    fixed result;
    for (i=0;i<ch_out;i++){
        result =0;
        for(j=0;j<ch_in;j++){
            result += fc_in[j]*params[i*ch_in+j];
        }
        if (have_bias==1 and have_bn==1){
            result += params[ch_out*ch_in+i];
            result *= params[ch_out*ch_in+ch_out+2*i];
            result += params[ch_out*ch_in+ch_out+2*i+1];
        }
        else if(have_bias==0 and have_bn==1){
            result *= params[ch_out*ch_in+2*i];
            result += params[ch_out*ch_in+2*i+1];
        }
        else if (have_bias==1 and have_bn==0){
            result+=params[ch_out*ch_in+i];
        }
        if (result <0 and have_relu==1){
        	fc_out[i] =0;
        }
        else{
        	fc_out[i] =result;
        }
    }
    return 0;
}



int max_pooling(fixed* pooling_in, int kernel_size, fixed* pooling_out, int ch, int size_out, int size_in){
//max pooling with stride = 1
    int id_img, id_ch, id_img_row, id_img_col,i,j;
    int output_idx, input_idx;
    fixed max_val;
    fixed input_val;
    id_img=0;
    for (id_ch=0;id_ch<ch;id_ch++){
        for(id_img_row=0;id_img_row < size_out ;id_img_row++){
                for(id_img_col=0; id_img_col < size_out;id_img_col++){
                    output_idx = id_img*ch*size_out*size_out+id_ch*size_out*size_out+id_img_row*size_out+id_img_col;
                    input_idx = id_img*ch*size_in*size_in+id_ch*size_in*size_in+id_img_row*kernel_size*size_in+id_img_col*kernel_size;
                    max_val = pooling_in[input_idx];
                    for (i=0;i<kernel_size;i++){
                        for(j=0;j<kernel_size;j++){
                            input_val = pooling_in[input_idx+i*size_in+j];
                            if (input_val>max_val) max_val=input_val;
                        }
                    }
                    pooling_out[output_idx] = max_val;
                }
            }
    }
    return 0;
}




int get_label(int num_classes, fixed* vec_in){
    int i;
    int label = 0;
    fixed max_val = vec_in[0];
    for (i=0;i<num_classes;i++){
        if (vec_in[i]>max_val){
            max_val = vec_in[i];
            label = i;
        }
    }
    return label;
}



