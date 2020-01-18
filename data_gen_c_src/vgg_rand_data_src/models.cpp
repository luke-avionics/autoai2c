#include "models.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>



float cnn(int num_batch, int bsize, bool print_or_not){
    //load binary file for trained weights
    int i,j,k;
    fixed * img;
    float * img_full;
    int f_we=open("/mnt/vgg11_bn_trained_params.bin", O_RDONLY);
    float *weights_full;
    if (f_we == -1) {
        perror("Error opening file for reading");
        exit(EXIT_FAILURE);
    }
    weights_full = (float *) mmap(0, sizeof(float)*(132868842), PROT_READ, MAP_SHARED, f_we, 0);
    if (weights_full == MAP_FAILED) {
        close(f_we);
        perror("Error mmapping the file");
        exit(EXIT_FAILURE);
    }
    int print_freq = 13286884;
    for (i = 0; i <132868842;i++){
        if (i%print_freq==0 or i>= 132868840){
            std::cout<<weights_full[i]<<std::endl;
        }
    }
    close(f_we);
    std::cout<<"Finished loading full precision weights"<<std::endl;


    int f_we_q=open("/mnt/vgg11_bn_trained_params_quantization.bin", O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (f_we_q==-1) {
        std::cout<<"Error opening file for writing"<<std::endl;
        exit(EXIT_FAILURE);
    }
    int result1 = lseek(f_we_q, sizeof(fixed)*132868842-1, SEEK_SET);
    if (result1 == -1) {
        close(f_we_q);
        std::cout<<"Error calling lseek() to stretch the file"<<std::endl;
        exit(EXIT_FAILURE);
    }
    result1 = write(f_we_q, "", 1);
    if (result1 != 1) {
        close(f_we_q);
        std::cout<<"Error writing last byte of the file"<<std::endl;
        exit(EXIT_FAILURE);
    }
    fixed * weights = (fixed *)mmap(0, sizeof(fixed)*132868842, PROT_READ | PROT_WRITE, MAP_SHARED, f_we_q, 0);
    if (weights == MAP_FAILED) {
        close(f_we_q);
        std::cout<<"Error mmapping the file"<<std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::cout<<"start type casting"<<std::endl;
    
    for (i = 0; i <132868842;i++) {
        weights[i]=(fixed) (weights_full[i]);
        if (i%print_freq==0 or i>= 132868840){
            std::cout<<weights[i]<<std::endl;
        }
    }
    


    //loop for multiple images
    const char* img_file=NULL;
    std::string tmp;
    FILE *f_img;
    int correct, total, label;
    correct =0;
    total = 0;
    float acc;
    int predicted;
    double arm_freq = 666.66*1000000.0;
    double t_total =0;
    double t_img =0;

    sds_utils::perf_counter hw_ctr;
    char* s1;
    char* s2;
    char* s3 = ".bin";
    std::cout<<"starting batch"<<std::endl;
    for (i=0;i<num_batch;i++){

        /*
    	//image directory
        if (i<10) s1 ="/home/root/sd/work/data_03_imagenet/test_imagenet_bin/batch000";
        else if (i<100) s1 ="/home/root/sd/work/data_03_imagenet/test_imagenet_bin/batch00";
        else s1 ="/home/root/sd/work/data_03_imagenet/test_imagenet_bin/batch0";
        */

        for (j=0;j<bsize;j++){

            /*
            if(j<10) s2="_000";
            else s2="_00";
            tmp = std::string(s1) + std::to_string(i)+ std::string(s2)+ std::to_string(j)+ std::string(s3);
            img_file = tmp.c_str();
            */

            //150529 image size 
            //replace image file here to some sampled ones, or just bin file
            const char* img_file = "/mnt/test_image.bin";
            f_img = fopen(img_file,"rb");
            if (!f_img){
                std::cout<<"Cannot open the file "<<img_file<<std::endl;
            }
            else{
                if (print_or_not) std::cout<<"procssing "<<img_file<<"............."<<std::endl;
                //input image mem alloc
                img = (fixed *) sds_alloc(150528*sizeof(fixed));
                img_full = (float*) sds_alloc(150529*sizeof(float));
                //input image load
                fread(img_full,sizeof(float),150529,f_img);
                label = (int)(img_full[150528]);// the label for the image
                for (k=0;k<150528;k++){
                    //if (k%300==0 or k == 150528)std::cout<<img_full[k]<<std::endl;
                	//type casting to fixed
                	img[k]=(fixed)(img_full[k]);
                }
                total++;
                hw_ctr.start();
                predicted = cnn_inference(img,weights);
                if (print_or_not) std::cout<<label<<"	"<<predicted<<std::endl;
                // accuracy test can comment out
                if (predicted==label){
                    correct++;
                    if (print_or_not) std::cout<<"current accuracy is "<<(float)correct/total<<std::endl;
                }
                hw_ctr.stop();
                uint64_t hw_cycles = hw_ctr.avg_cpu_cycles();
                t_img = double(hw_cycles)/arm_freq*1000.0;
                t_total+=t_img;
                sds_free(img);
                sds_free(img_full);
            }
            fclose(f_img);
        }
        std::cout<<"Till batch "<<i<<": the time for each image is "<<t_total/((i+1)*bsize)<<"ms"<<std::endl;
    }

    if (munmap(weights_full, sizeof(float)*(132868842)) == -1) {
        std::cout<<"Error un-mmapping the file"<<std::endl;
    }
    //TODO: comment out PLS, we do not have mapping enabled 
    if (munmap(weights, sizeof(fixed)*132868842) == -1) {
        std::cout<<"Error un-mmapping the file"<<std::endl;
    }
    close(f_we_q);
    //acc = (float) correct/total;
    //std::cout<<"The time for each image is "<<t_total/(num_batch*bsize)<<"ms"<<std::endl;
    return acc;
}


int cnn_inference(fixed *img, fixed *params){
    int predicted;
    int i,j,k;
    int num_classes = 1000;
    /*
    for (k=0;k<num_pixels;k++){
        if (k%300==0 or k == num_pixels)std::cout<<img[k]<<std::endl;
    	//img[i]=img_full[i];
    }
    */
    //std::cout<<"we[0]"<<params[0]<<std::endl;
    // for pytorch, the feature map (4D) is batch_size x channel x row x col
    // for pytorch, the weighs(4D) are ch_out x ch_in x kernel_row x kernel_col
    // for pytorch, the bias(1D) are ch_out
    // after merging 4 params for batch norm into 2, the are 2*ch_out stored in a [w1,b1,w2,b2...] manner




    /*
	fixed* param_l1= (fixed *) sds_alloc(sizeof(fixed)*1920);
    for (i=0;i<1920;i++){
    	param_l1[i]=params[i+0];
    }
	fixed* l1= (fixed *) sds_alloc(sizeof(fixed)*3211264);
    conv(img,param_l1,l1,1,1,64,3,3,224,224,1,1);
    //std::cout<<"l1[0] is "<<l1[0]<<std::endl;
    std::cout<<"l1....... "<<std::endl;
    sds_free(param_l1);
	fixed* param_l2= (fixed *) sds_alloc(sizeof(fixed)*0);
    for (i=0;i<0;i++){
    	param_l2[i]=params[i+1920];
    }
	fixed* l2= (fixed *) sds_alloc(sizeof(fixed)*802816);
    max_pooling(l1,2,l2,64,112,224);
    //std::cout<<"l2[0] is "<<l2[0]<<std::endl;
    std::cout<<"l2....... "<<std::endl;
    sds_free(l1);
    sds_free(param_l2);
	fixed* param_l3= (fixed *) sds_alloc(sizeof(fixed)*74112);
    for (i=0;i<74112;i++){
    	param_l3[i]=params[i+1920];
    }
	fixed* l3= (fixed *) sds_alloc(sizeof(fixed)*1605632);
    conv(l2,param_l3,l3,1,1,128,64,3,112,112,1,1);
    //std::cout<<"l3[0] is "<<l3[0]<<std::endl;
    std::cout<<"l3....... "<<std::endl;
    sds_free(l2);
    sds_free(param_l3);
	fixed* param_l4= (fixed *) sds_alloc(sizeof(fixed)*0);
    for (i=0;i<0;i++){
    	param_l4[i]=params[i+76032];
    }
	fixed* l4= (fixed *) sds_alloc(sizeof(fixed)*401408);
    max_pooling(l3,2,l4,128,56,112);
    //std::cout<<"l4[0] is "<<l4[0]<<std::endl;
    std::cout<<"l4....... "<<std::endl;
    sds_free(l3);
    sds_free(param_l4);
    */

    fixed* l4= (fixed *) sds_alloc(sizeof(fixed)*401408);
    for(i=0; i<401408; i++){
    	l4[i]=params[i];
    }

	fixed* param_l5= (fixed *) sds_alloc(sizeof(fixed)*295680);
    for (i=0;i<295680;i++){
    	param_l5[i]=params[i+76032];
    }
	//#pragma HLS ARRAY_PARTITION variable=param_l5 dim=0 <block|cyclic>
	fixed* l5= (fixed *) sds_alloc(sizeof(fixed)*802816);
    conv(l4,param_l5,l5,1,1,256,128,3,56,56,1,1);
    //std::cout<<"l5[0] is "<<l5[0]<<std::endl;
    std::cout<<"l5....... "<<std::endl;
    sds_free(l4);
    sds_free(param_l5);


    sds_free(l5);
    return 10;
    /*
	fixed* param_l6= (fixed *) sds_alloc(sizeof(fixed)*590592);
    for (i=0;i<590592;i++){
    	param_l6[i]=params[i+371712];
    }
	fixed* l6= (fixed *) sds_alloc(sizeof(fixed)*802816);
    conv(l5,param_l6,l6,1,1,256,256,3,56,56,1,1);
    //std::cout<<"l6[0] is "<<l6[0]<<std::endl;
    std::cout<<"l6....... "<<std::endl;
    sds_free(l5);
    sds_free(param_l6);
	fixed* param_l7= (fixed *) sds_alloc(sizeof(fixed)*0);
    for (i=0;i<0;i++){
    	param_l7[i]=params[i+962304];
    }
	fixed* l7= (fixed *) sds_alloc(sizeof(fixed)*200704);
    max_pooling(l6,2,l7,256,28,56);
    //std::cout<<"l7[0] is "<<l7[0]<<std::endl;
    std::cout<<"l7....... "<<std::endl;
    sds_free(l6);
    sds_free(param_l7);
	fixed* param_l8= (fixed *) sds_alloc(sizeof(fixed)*1181184);
    for (i=0;i<1181184;i++){
    	param_l8[i]=params[i+962304];
    }
	fixed* l8= (fixed *) sds_alloc(sizeof(fixed)*401408);
    conv(l7,param_l8,l8,1,1,512,256,3,28,28,1,1);
    //std::cout<<"l8[0] is "<<l8[0]<<std::endl;
    std::cout<<"l8....... "<<std::endl;
    sds_free(l7);
    sds_free(param_l8);
	fixed* param_l9= (fixed *) sds_alloc(sizeof(fixed)*2360832);
    for (i=0;i<2360832;i++){
    	param_l9[i]=params[i+2143488];
    }
	fixed* l9= (fixed *) sds_alloc(sizeof(fixed)*401408);
    conv(l8,param_l9,l9,1,1,512,512,3,28,28,1,1);
    //std::cout<<"l9[0] is "<<l9[0]<<std::endl;
    std::cout<<"l9....... "<<std::endl;
    sds_free(l8);
    sds_free(param_l9);
	fixed* param_l10= (fixed *) sds_alloc(sizeof(fixed)*0);
    for (i=0;i<0;i++){
    	param_l10[i]=params[i+4504320];
    }
	fixed* l10= (fixed *) sds_alloc(sizeof(fixed)*100352);
    max_pooling(l9,2,l10,512,14,28);
    //std::cout<<"l10[0] is "<<l10[0]<<std::endl;
    std::cout<<"l10....... "<<std::endl;
    sds_free(l9);
    sds_free(param_l10);
	fixed* param_l11= (fixed *) sds_alloc(sizeof(fixed)*2360832);
    for (i=0;i<2360832;i++){
    	param_l11[i]=params[i+4504320];
    }
	fixed* l11= (fixed *) sds_alloc(sizeof(fixed)*100352);
    conv(l10,param_l11,l11,1,1,512,512,3,14,14,1,1);
    //std::cout<<"l11[0] is "<<l11[0]<<std::endl;
    std::cout<<"l11....... "<<std::endl;
    sds_free(l10);
    sds_free(param_l11);
	fixed* param_l12= (fixed *) sds_alloc(sizeof(fixed)*2360832);
    for (i=0;i<2360832;i++){
    	param_l12[i]=params[i+6865152];
    }
	fixed* l12= (fixed *) sds_alloc(sizeof(fixed)*100352);
    conv(l11,param_l12,l12,1,1,512,512,3,14,14,1,1);
    //std::cout<<"l12[0] is "<<l12[0]<<std::endl;
    std::cout<<"l12....... "<<std::endl;
    sds_free(l11);
    sds_free(param_l12);
	fixed* param_l13= (fixed *) sds_alloc(sizeof(fixed)*0);
    for (i=0;i<0;i++){
    	param_l13[i]=params[i+9225984];
    }
	fixed* l13= (fixed *) sds_alloc(sizeof(fixed)*25088);
    max_pooling(l12,2,l13,512,7,14);
    //std::cout<<"l13[0] is "<<l13[0]<<std::endl;
    std::cout<<"l13....... "<<std::endl;
    sds_free(l12);
    sds_free(param_l13);
	fixed* param_l14= (fixed *) sds_alloc(sizeof(fixed)*102764544);
    for (i=0;i<102764544;i++){
    	param_l14[i]=params[i+9225984];
    }
	fixed* l14= (fixed *) sds_alloc(sizeof(fixed)*4096);
    fc(l13,param_l14,l14,0,1,1,4096,25088);
    //std::cout<<"l14[0] is "<<l14[0]<<std::endl;
    std::cout<<"l14....... "<<std::endl;
    sds_free(l13);
    sds_free(param_l14);
	fixed* param_l15= (fixed *) sds_alloc(sizeof(fixed)*16781312);
    for (i=0;i<16781312;i++){
    	param_l15[i]=params[i+111990528];
    }
	fixed* l15= (fixed *) sds_alloc(sizeof(fixed)*4096);
    fc(l14,param_l15,l15,0,1,1,4096,4096);
    //std::cout<<"l15[0] is "<<l15[0]<<std::endl;
    std::cout<<"l15....... "<<std::endl;
    sds_free(l14);
    sds_free(param_l15);
	fixed* param_l16= (fixed *) sds_alloc(sizeof(fixed)*4097000);
    for (i=0;i<4097000;i++){
    	param_l16[i]=params[i+128771840];
    }
	fixed* l16= (fixed *) sds_alloc(sizeof(fixed)*1000);
    fc(l15,param_l16,l16,0,0,1,1000,4096);
    //std::cout<<"l16[0] is "<<l16[0]<<std::endl;
    std::cout<<"l16....... "<<std::endl;
    sds_free(l15);
    sds_free(param_l16);
    //get label
    predicted = get_label(num_classes, l16);
    sds_free(l16);
    return predicted;
    */
}
