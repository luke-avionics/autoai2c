#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>



int main(){
    const int size = 4;
    float *weights_full;
    int f_we=open("test_image.bin", O_RDONLY);
    weights_full = (float *) mmap(0, sizeof(float)*(size), PROT_READ, MAP_SHARED, f_we, 0); 
    if (weights_full == MAP_FAILED) {
        close(f_we);
        perror("Error mmapping the file");
        exit(EXIT_FAILURE);
    }
    int print_freq = 1;
    int i;
    for (i = 0; i <size ;i++){
        if (i%print_freq==0){
            std::cout<<weights_full[i]<<std::endl;
        }
    }
    close(f_we);
}
