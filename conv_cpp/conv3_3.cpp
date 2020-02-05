#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
using namespace std;
typedef int data_type;



data_type main(){


    //mem allocation
    data_type ****weight=(data_type ****)malloc(2*sizeof(data_type***));
    for(unsigned int i=0; i<2;i++){
        weight[i]=(data_type ***)malloc(2*sizeof(data_type**));
        for(unsigned int j=0; j<2;j++){
            weight[i][j]=(data_type **)malloc(3*sizeof(data_type*));
            for(unsigned int k=0; k<3;k++){
                weight[i][j][k]=(data_type *)malloc(3*sizeof(data_type));
            }
        }
    }


    data_type ***input=(data_type ***)malloc(2*sizeof(data_type**));
    for(unsigned int i=0; i<2; i++){
        input[i]=(data_type **)malloc(14*sizeof(data_type*));
        for(unsigned int j=0;j<14;j++){
            input[i][j]=(data_type *)malloc(14*sizeof(data_type));    
        }
    }


    data_type ***output=(data_type ***)malloc(2*sizeof(data_type**));
    for(unsigned int i=0; i<2; i++){
        output[i]=(data_type **)malloc(12*sizeof(data_type*));
        for(unsigned int j=0;j<12;j++){
            output[i][j]=(data_type *)malloc(12*sizeof(data_type));    
        }
    }



    for(unsigned int i=0; i<2;i++)
        for(unsigned int j=1; j<13;j++)
            for(unsigned int k=1; k<13; k++)
                input[i][j][k]=j-1;

    for(unsigned int i=0; i<2;i++)
        for(unsigned int j=0; j<12;j++)
            for(unsigned int k=0; k<12; k++)
                output[i][j][k]=0;

    //padding zero
    for(unsigned int i=0; i<2;i++)
        for(unsigned int j=0; j<14;j++)
            input[i][j][0]=0;

    for(unsigned int i=0; i<2;i++)
        for(unsigned int j=0; j<14;j++)
            input[i][0][j]=0;


    

    for(unsigned int i=0; i<2;i++)
        for(unsigned int j=0; j<2;j++)
            for(unsigned int k=0; k<3; k++)
                for(unsigned int l=0; l<3; l++ )
                    weight[i][j][k][l]=k;        


    unsigned int o_col, o_row, ch_in,ch_out,w_col,w_row;
    for(ch_out=0;ch_out<2; ch_out++){
        for(o_col=0;o_col<12; o_col++){
            for(o_row=0;o_row<12; o_row++){
                for(ch_in=0;ch_in<2;ch_in++){
                    for(w_col=0;w_col<3;w_col++){
                        for(w_row=0;w_row<3;w_row++){
                            output[ch_out][o_row][o_col]+=weight[ch_out][ch_in][w_row][w_col]*input[ch_in][o_row+w_row][o_col+w_col];                    
                        }
                    }                    
                }                   
            }        
        }
    }

    for(unsigned int j=0; j<12;j++){
        for(unsigned int k=0; k<12; k++){
            cout<<output[0][j][k];
            cout<<", ";
        }
            cout<<"\n";
    }
    
 
    return 1; 

}
