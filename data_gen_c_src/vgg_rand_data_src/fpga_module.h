#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ctime>
#include <ap_fixed.h>
#include "stdio.h"
#include "sds_utils.h"
#define BUFFER_SIZE 4096
//#pragma SDS data zero_copy(in1[0:dim*dim], in2[0:dim*dim], out[0:dim*dim])
//typedef float fixed;
typedef ap_fixed<8,6,AP_RND > fixed;
//for pytorch, the feature map (4D) is batch_size x channel x row x col
//for pytorch, the weighs(4D) are ch_out x ch_in x kernel_row x kernel_col
//for pytorch, the bias(1D) are ch_out
//for pytorch, the bn parameters are 2*ch_out, (w1,b1,w2,b2,w3,b3.....)
#pragma SDS data zero_copy(conv_in[0:ch_in*size_in*size_in], params[0:ch_in*ch_out*kernel_size*kernel_size+ch_out+have_bn*2*ch_out],conv_out[0:ch_out*size_out*size_out])
int conv(fixed* conv_in, fixed* params, fixed* conv_out, int have_bn, int have_relu, int ch_out, int ch_in, int kernel_size, int size_out, int size_in, int stride, int padding);
#pragma SDS data zero_copy(fc_in[0:ch_in], params[0:ch_in*ch_out+have_bias*ch_out+have_bn*2*ch_out], fc_out[0:ch_out])
int fc(fixed* fc_in, fixed* params, fixed * fc_out, int have_bn, int have_relu, int have_bias, int ch_out, int ch_in);
#pragma SDS data zero_copy(pooling_in[0:ch*size_in*size_in], pooling_out[0:ch*size_out*size_out])
int max_pooling(fixed* pooling_in, int kernel_size, fixed* pooling_out, int ch, int size_out, int size_in);
#pragma SDS data zero_copy(vec_in[0:num_classes])
int get_label(int num_classes, fixed* vec_in);
