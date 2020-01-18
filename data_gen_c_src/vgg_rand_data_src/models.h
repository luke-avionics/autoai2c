#include "fpga_module.h"
#include <ctime>
#include "sds_utils.h"

//vgg11_bn for imagenet
#define num_params 132868840
#define num_pixels 150528
int cnn_inference(fixed *img, fixed *params);//return the inference result for 1 image
float cnn(int num_batch, int bsize, bool print_or_not);// return the accuracy for num_batch*100 images
