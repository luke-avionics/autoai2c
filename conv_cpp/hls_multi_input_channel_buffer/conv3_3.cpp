#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "conv3x3.h"
using namespace std;



data_type main(){


    //mem allocation
    data_type ****weight=(data_type ****)malloc(256*sizeof(data_type***));
    for(unsigned int i=0; i<256;i++){
        weight[i]=(data_type ***)malloc(128*sizeof(data_type**));
        for(unsigned int j=0; j<128;j++){
            weight[i][j]=(data_type **)malloc(3*sizeof(data_type*));
            for(unsigned int k=0; k<3;k++){
                weight[i][j][k]=(data_type *)malloc(3*sizeof(data_type));
            }
        }
    }


    data_type ***input=(data_type ***)malloc(128*sizeof(data_type**));
    for(unsigned int i=0; i<128; i++){
        input[i]=(data_type **)malloc(58*sizeof(data_type*));
        for(unsigned int j=0;j<58;j++){
            input[i][j]=(data_type *)malloc(58*sizeof(data_type));    
        }
    }


    data_type ***output=(data_type ***)malloc(256*sizeof(data_type**));
    for(unsigned int i=0; i<256; i++){
        output[i]=(data_type **)malloc(56*sizeof(data_type*));
        for(unsigned int j=0;j<56;j++){
            output[i][j]=(data_type *)malloc(56*sizeof(data_type));    
        }
    }



    for(unsigned int i=0; i<128;i++)
        for(unsigned int j=1; j<57;j++)
            for(unsigned int k=1; k<57; k++)
                input[i][j][k]=j-1;

    for(unsigned int i=0; i<256;i++)
        for(unsigned int j=0; j<56;j++)
            for(unsigned int k=0; k<56; k++)
                output[i][j][k]=0;

    //padding zero
    for(unsigned int i=0; i<128;i++)
        for(unsigned int j=0; j<58;j++)
            input[i][j][0]=0;

    for(unsigned int i=0; i<128;i++)
        for(unsigned int j=0; j<58;j++)
            input[i][0][j]=0;


    

    for(unsigned int i=0; i<256;i++)
        for(unsigned int j=0; j<128;j++)
            for(unsigned int k=0; k<3; k++)
                for(unsigned int l=0; l<3; l++ )
                    weight[i][j][k][l]=k;        







    //flatten the matrix
   data_type * flatten_weight=(data_type *)malloc(256*128*3*3*sizeof(data_type));
   data_type * flatten_input=(data_type *)malloc(128*58*58*sizeof(data_type));
   data_type * flatten_output=(data_type *)malloc(256*56*56*sizeof(data_type));
   for(unsigned int i=0; i<256;i++)
       for(unsigned int j=0; j<128;j++)
           for(unsigned int k=0; k<3; k++)
               for(unsigned int l=0; l<3; l++ )
               {
            	   flatten_weight[i*128*3*3+j*3*3+k*3+l]=weight[i][j][k][l];
               }

   for(unsigned int i=0; i<128; i++)
	   for(unsigned int j=0; j<58; j++)
		   for(unsigned int k=0; k<58; k++){
			   flatten_input[i*58*58+j*58+k]=input[i][j][k];
		   }

   for(unsigned int i=0; i<256; i++)
	   for(unsigned int j=0; j<56; j++)
		   for(unsigned int k=0; k<56; k++){
			   flatten_output[i*56*56+j*58+k]=output[i][j][k];
		   }





   //module
   //possible solution, specify regular r/w pattern
    conv3_3(flatten_input,flatten_weight,flatten_output);

    for(unsigned int j=0; j<56;j++){
        for(unsigned int k=0; k<56; k++){
            cout<<flatten_output[j*56+k];
            cout<<", ";
        }
            cout<<"==========\n";
    }

 
    return 0;

}
