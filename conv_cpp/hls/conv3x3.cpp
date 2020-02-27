#include "conv3x3.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
using namespace std;

void input_buffer0(data_type input_buffer[1][58][6], data_type* flatten_input,unsigned int ch_in_1){
	for (unsigned int i=0;i<58;i++){
		for (unsigned int j=0;j<3;j++){
			#pragma HLS PIPELINE
			input_buffer[0][i][j]=flatten_input[ch_in_1*58*58+58*i+j];
		}
	}
}

void weight_buffer0(data_type weight[1][1][3][3],data_type* flatten_weight,
		          unsigned int ch_in_1, unsigned int ch_out_1)
{
	   //for(unsigned int i=0; i<256;i++)
		   //for(unsigned int j=0; j<128;j++)
			   weight_buffer:for(unsigned int k=0; k<3; k++)
				   for(unsigned int l=0; l<3; l++ )
				   {
					   #pragma HLS PIPELINE
					   weight[0][0][k][l]=flatten_weight[ch_out_1*128*3*3+ch_in_1*3*3+k*3+l];
				   }
}

void input_buffer1(data_type input_buffer[0][58][6], data_type* flatten_input,
		          unsigned int ch_in_1, unsigned int o_col, unsigned int o_col_buffer_index){
	for (unsigned int i=0; i<58; i++){
		#pragma HLS PIPELINE
		input_buffer[0][i][o_col_buffer_index]=flatten_input[ch_in_1*58*58+i*58+o_col+3];
		}
}

void output_buffer0(data_type output_buffer[0][56][2], data_type* flatten_output, unsigned int ch_out_1,
		           unsigned int o_col, unsigned int o_col_buffer_output_index){
	data_type tmp[56];
	#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
	for (unsigned int i=0; i<56;i++){
		#pragma HLS UNROLL
		tmp[i]=flatten_output[ch_out_1*56*56+i*56+o_col];

	}
	for (unsigned int i=0; i<56;i++){
		//#pragma HLS PIPELINE
	#pragma HLS UNROLL
		flatten_output[ch_out_1*56*56+i*56+o_col]=tmp[i]+output_buffer[0][i][o_col_buffer_output_index];
		//output_buffer[0][i][o_col_buffer_output_index]=0;

	}
	for (unsigned int i=0; i<56;i++){
	#pragma HLS UNROLL
		output_buffer[0][i][o_col_buffer_output_index]=0;

	}
}
//void output_buffer0(data_type output_buffer[0][56][2], data_type* flatten_output, unsigned int ch_out_1,
//		           unsigned int o_col, unsigned int o_col_buffer_output_index){
//	for (unsigned int i=0; i<56;i++){
//		#pragma HLS PIPELINE
//
//		flatten_output[ch_out_1*56*56+i*56+o_col]+=output_buffer[0][i][o_col_buffer_output_index];
//		output_buffer[0][i][o_col_buffer_output_index]=0;
//
//	}
//}


void column_based_engine1(data_type output_buffer[0][56][2],data_type weight[1][1][3][3],data_type input_buffer[1][58][6],
		                 unsigned int o_row, unsigned int ch_in,unsigned int w_col,unsigned int w_row,
						 unsigned int o_col_buffer_index, unsigned int o_col_buffer_output_index){

	int col0,col1,col2,offset;

	offset=o_col_buffer_index-3;
//	if (offset==6){
//		col0=offset-5;
//		col1=offset-4;
//		col2=offset-3;
//	}
//	else if(offset==5){
//		col0=offset;
//		col1=offset-4;
//		col2=offset-3;
//	}
//	else if(offset==4){
//		col0=offset;
//		col1=offset+1;
//		col2=offset-3;
//	}
//	else if(offset==-1){
//		col0=offset+5;
//		col1=offset+1;
//		col2=offset+2;
//	}
//	else if(offset==-2){
//		col0=offset+5;
//		col1=offset+6;
//		col2=offset+2;
//	}
//	else if(offset==-4){
//		col0=offset+5;
//		col1=offset+6;
//		col2=offset+7;
//	}
//	else{
//		col0=offset;
//		col1=offset+1;
//		col2=offset+2;
//	}

    if(offset==-1){
		col0=offset+6;
		col1=offset+1;
		col2=offset+2;
	}
	else if(offset==-2){
		col0=offset+6;
		col1=offset+7;
		col2=offset+2;
	}
	else if(offset==-3){
		col0=offset+6;
		col1=offset+7;
		col2=offset+8;
	}
	else{
		col0=offset;
		col1=offset+1;
		col2=offset+2;
	}

	for(o_row=0;o_row<20; o_row++){
			//#pragma HLS UNROLL factor=4
	#pragma HLS PIPELINE
	#pragma HLS DEPENDENCE variable=output_buffer intra false
	#pragma HLS DEPENDENCE variable=input_buffer intra false
	#pragma HLS DEPENDENCE variable=weight intra false
			for(ch_in=0;ch_in<1;ch_in++){
				int tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
				int tmp6,tmp7,tmp8,tmp9,tmp10,tmp11;
				int tmp12,tmp13,tmp14,tmp15,tmp16;

				tmp0=weight[0][ch_in][0][0]*input_buffer[ch_in][o_row+0][col0];
				tmp1=weight[0][ch_in][0][1]*input_buffer[ch_in][o_row+0][col1];
				//tmp9=tmp0+tmp1;
				tmp2=weight[0][ch_in][0][2]*input_buffer[ch_in][o_row+0][col2];
				tmp3=weight[0][ch_in][1][0]*input_buffer[ch_in][o_row+1][col0];
				//tmp10=tmp2+tmp3;
				tmp4=weight[0][ch_in][1][1]*input_buffer[ch_in][o_row+1][col1];
				tmp5=weight[0][ch_in][1][2]*input_buffer[ch_in][o_row+1][col2];
				//tmp11=tmp4+tmp5;
				tmp6=weight[0][ch_in][2][0]*input_buffer[ch_in][o_row+2][col0];
				tmp7=weight[0][ch_in][2][1]*input_buffer[ch_in][o_row+2][col1];
				//tmp12=tmp6+tmp7;
				tmp8=weight[0][ch_in][2][2]*input_buffer[ch_in][o_row+2][col2];



				tmp9=tmp0+tmp1;
				tmp10=tmp2+tmp3;
				tmp11=tmp4+tmp5;
				tmp12=tmp6+tmp7;

				tmp13=tmp9+tmp10;
				tmp14=tmp11+tmp12;

				tmp15=tmp13+tmp14;

				output_buffer[0][o_row][o_col_buffer_output_index]=tmp15+tmp8;
			}
		}


}


void column_based_engine_wrapper(data_type output_buffer[0][56][2],data_type weight[1][1][3][3],data_type input_buffer[1][58][6],
        unsigned int o_row, unsigned int ch_in,unsigned int w_col,unsigned int w_row,
		 unsigned int o_col_buffer_index, unsigned int o_col_buffer_output_index){
	//column_based_engine(output_buffer, weight, input_buffer, o_row, ch_in,w_col, w_row,o_col_buffer_index, o_col_buffer_output_index);
	column_based_engine1(output_buffer, weight, input_buffer, o_row, ch_in,w_col, w_row,o_col_buffer_index, o_col_buffer_output_index);
}

void conv3_3(data_type* flatten_input, data_type* flatten_weight, data_type* flatten_output ){
#pragma HLS INTERFACE axis  depth=430592 port=flatten_input
#pragma HLS INTERFACE axis  depth=294912 port=flatten_weight
#pragma HLS INTERFACE m_axi  depth=802816 port=flatten_output
//    data_type input[1][58][58];
	data_type input_buffer[4][58][6];

#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=3
	data_type weight[1][1][3][3];
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3
#pragma HLS ARRAY_PARTITION variable=weight complete dim=4
//	data_type output[1][56][56];
	data_type output_buffer[1][56][2];
#pragma HLS ARRAY_PARTITION variable=output_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=output_buffer complete dim=3



//	   for(unsigned int j=0; j<58; j++){
//		   for(unsigned int k=0; k<58; k++){
//			   cout<<flatten_input[j*58+k];
//			   cout<<",";
//		   }
//		   cout<<"\n";
//	   }
//	   cout<<"=======================\n";
//	   for(unsigned int j=0; j<3; j++){
//		   for(unsigned int k=0; k<3; k++){
//			   cout<<flatten_weight[j*3+k];
//			   cout<<",";
//		   }
//		   cout<<"\n";
//	   }
//	cout<<"\n";
//	cout<<"Testingggggg \n";



	//fill initial ifmap
	for (unsigned int i=0;i<58;i++){
		for (unsigned int j=0;j<6;j++){
            #pragma HLS PIPELINE
			input_buffer[0][i][j]=flatten_input[58*i+j];
		}
	}
	//fill initial ofmap
	for (unsigned int i=0; i<56;i++){
		#pragma HLS PIPELINE
		output_buffer[0][i][0]=0;
		output_buffer[0][i][1]=0;
	}


   //computation
    unsigned int o_col, o_row, ch_in,ch_out,w_col,w_row;
    unsigned int o_col_1, o_row_1, ch_in_1,ch_out_1,w_col_1,w_row_1;








	for(ch_in_1=0; ch_in_1<128;ch_in_1++){

		input_buffer0(input_buffer, flatten_input,ch_in_1);


		for(ch_out_1=0; ch_out_1<256;ch_out_1++){

			weight_buffer0(weight,flatten_weight, ch_in_1,ch_out_1);

			computer_engine:for(ch_out=0;ch_out<1; ch_out++){
				unsigned int o_col_buffer_index=3;
				unsigned int o_col_buffer_output_index=0;
				unsigned int o_col_p1=55;
				for(o_col=0;o_col<56; o_col++){
					#pragma HLS DEPENDENCE variable=output_buffer intra false
					#pragma HLS DEPENDENCE variable=input_buffer intra false


					//column_based_engine(output_buffer, weight, input_buffer, o_row, ch_in,w_col, w_row,o_col_buffer_index, o_col_buffer_output_index);
					//column_based_engine1(output_buffer, weight, input_buffer, o_row, ch_in,w_col, w_row,o_col_buffer_index, o_col_buffer_output_index);
					column_based_engine_wrapper(output_buffer, weight, input_buffer, o_row, ch_in,w_col, w_row,o_col_buffer_index, o_col_buffer_output_index);
					//potential optimization to reduce a couple of more cycles
					input_buffer1(input_buffer, flatten_input,
							      ch_in_1, o_col, o_col_buffer_index);


					//write the col before the current col
				    output_buffer0(output_buffer,flatten_output, ch_out_1, o_col_p1, 1-o_col_buffer_output_index);

					if(o_col_buffer_output_index==0)
						o_col_buffer_output_index++;
					else
						o_col_buffer_output_index=0;


					if(o_col_p1==55)
						o_col_p1=0;
					else
						o_col_p1++;

					o_col_buffer_index++;
					if (o_col_buffer_index>5)
						o_col_buffer_index=0;

				}

			    //output_buffer0(output_buffer,flatten_output, ch_out_1,  o_col-1, 1-o_col_buffer_output_index);
			}

    	}
		//output_buffer0(output_buffer,flatten_output, 255, 55, 1);

    }



}
