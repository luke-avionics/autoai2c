#include "models.h"
#include <csignal>

void signalHandler(int signum){
	std::cout<<"---------------------You use keyboard interrupt!!!!-------------------"<<std::endl;
	std::cout<<"---------------------Reboot the board or you will get wrong output!!!-----------------------"<<std::endl;
	exit(signum);
}

int main(int argc, char** argv){
	signal(SIGINT,signalHandler);
    int num_batch=1;
    int bsize = 1;//you cannot change this one because of the file name
    float acc;
    bool print_or_not = true;
    if (argc==1){;}
    else if(argc==2){
    	num_batch = std::stoi(argv[1]);
    	if (num_batch<1 or num_batch>500){
    		std::cout<<"num_of_batch is within [1,500]"<<std::endl;
    		return -1;
    	}
    }
    else if (argc==3){
    	num_batch = std::stoi(argv[1]);
    	if (num_batch<1 or num_batch>500){
    		std::cout<<"num_of_batch is within [1,500]"<<std::endl;
    		return -1;
    	}
    	int print_or = std::stoi(argv[2]);
    	if (print_or < 0 or print_or>1){
    		std::cout<<"options of print is 0 (dont print) or 1 (print) only"<<std::endl;
    		return -1;
    	}
    	print_or_not = (bool) print_or;
    }
    acc = cnn(num_batch,bsize,print_or_not);
    std::cout<<"For "<<"vgg19_bn"<<" the test accuracy on Imagenet ("<<num_batch<<" batches) is "<< 100*acc<<"%"<<std::endl;
}
