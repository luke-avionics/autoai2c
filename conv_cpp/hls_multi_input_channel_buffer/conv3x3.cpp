#include "conv3x3.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
using namespace std;

//TODO: change unroll in buffering to pipeline
void input_buffer0(data_type input_buffer[4][6][58], data_type* flatten_input,unsigned int ch_in_1){
	for (unsigned int k=0; k< 4;k++){
		for (unsigned int i=0;i<3;i++){
			for (unsigned int j=0;j<58;j++){
				#pragma HLS PIPELINE
				input_buffer[k][i][j]=flatten_input[(ch_in_1*4+k)*58*58+58*i+j];
			}
		}
	}
}

void weight_buffer0(data_type weight[1][4][3][3],data_type* flatten_weight,
		          unsigned int ch_in_1, unsigned int ch_out_1)
{
	   //for(unsigned int i=0; i<256;i++)
		   for(unsigned int j=0; j<4;j++)
			   weight_buffer:for(unsigned int k=0; k<3; k++)
				   for(unsigned int l=0; l<3; l++ )
				   {
					   #pragma HLS UNROLL
					   weight[0][j][k][l]=flatten_weight[ch_out_1*128*3*3+(ch_in_1*4+k)*3*3+k*3+l];
				   }
}



void input_buffer1(data_type input_buffer[6][58], data_type* flatten_input,
		          unsigned int ch_in_1, unsigned int o_row, unsigned int o_row_buffer_index,unsigned int ch_in_index){
	#pragma HLS INLINE off
	unsigned int tmp=ch_in_1*4+ch_in_index;
	for (unsigned int i=0; i<58; i++){
		#pragma HLS PIPELINE
		input_buffer[o_row_buffer_index][i]=flatten_input[tmp*58*58+(o_row+3)*58+i];
		}
}
void input_buffer1_2(data_type input_buffer[6][58], data_type* flatten_input,
		          unsigned int ch_in_1, unsigned int o_row, unsigned int o_row_buffer_index,unsigned int ch_in_index){
	#pragma HLS INLINE off
	unsigned int tmp=ch_in_1*4+ch_in_index;
	for (unsigned int i=0; i<58; i++){
		#pragma HLS PIPELINE
		input_buffer[o_row_buffer_index][i]=flatten_input[tmp*58*58+(o_row+3)*58+i];
		}
}
//TODO: inline this function
void input_buffer_wrapper(data_type input_buffer[6][58], data_type input_buffer_1[6][58], data_type* flatten_input,
        unsigned int ch_in_1, unsigned int o_row, unsigned int o_row_buffer_index){
	#pragma HLS INLINE off
	#pragma HLS allocation instances=input_buffer1 limit=1 function
	#pragma HLS allocation instances=input_buffer1_2 limit=1 function
    #pragma HLS DEPENDENCE variable=input_buffer intra false
	input_buffer1(input_buffer, flatten_input, ch_in_1,o_row,o_row_buffer_index,0);
	input_buffer1(input_buffer_1, flatten_input, ch_in_1,o_row,o_row_buffer_index,1);
//	input_buffer1(input_buffer[2], flatten_input, ch_in_1,o_row,o_row_buffer_index,2);
//	input_buffer1(input_buffer[3], flatten_input, ch_in_1,o_row,o_row_buffer_index,3);
}

void output_buffer0(data_type output_buffer[1][2][56], data_type* flatten_output, unsigned int ch_out_1,
		           unsigned int o_row, unsigned int o_row_buffer_output_index){
	data_type tmp[56];
	#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
	for (unsigned int i=0; i<56;i++){
		#pragma HLS PIPELINE
		tmp[i]=flatten_output[ch_out_1*56*56+o_row*56+i];

	}
	for (unsigned int i=0; i<56;i++){
		//#pragma HLS PIPELINE
	#pragma HLS PIPELINE
		flatten_output[ch_out_1*56*56+o_row*56+i]=tmp[i]+output_buffer[0][o_row_buffer_output_index][i];
		//output_buffer[0][o_row_buffer_output_index][i]=0;

	}
	for (unsigned int i=0; i<56;i++){
	#pragma HLS PIPELINE
		output_buffer[0][o_row_buffer_output_index][i]=0;

	}
}


data_type adder_tree(unsigned int ch_in, unsigned int o_col,  int row0, int row1,
		        int row2, unsigned int o_row_buffer_output_index,
				data_type weight[3][3],data_type input_buffer[6][58]){
	data_type tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
	data_type tmp6,tmp7,tmp8,tmp9,tmp10,tmp11;
	data_type tmp12,tmp13,tmp14,tmp15,tmp16;
	for(ch_in=0;ch_in<1;ch_in++){
		tmp0=weight[0][0]*input_buffer[row0][o_col+0];
		tmp1=weight[1][0]*input_buffer[row1][o_col+0];
		//tmp9=tmp0+tmp1;
		tmp2=weight[2][0]*input_buffer[row2][o_col+0];
		tmp3=weight[0][1]*input_buffer[row0][o_col+1];
		//tmp10=tmp2+tmp3;
		tmp4=weight[1][1]*input_buffer[row1][o_col+1];
		tmp5=weight[2][1]*input_buffer[row2][o_col+1];
		//tmp11=tmp4+tmp5;
		tmp6=weight[0][2]*input_buffer[row0][o_col+2];
		tmp7=weight[1][2]*input_buffer[row1][o_col+2];
		//tmp12=tmp6+tmp7;
		tmp8=weight[2][2]*input_buffer[row2][o_col+2];



		tmp9=tmp0+tmp1;
		tmp10=tmp2+tmp3;
		tmp11=tmp4+tmp5;
		tmp12=tmp6+tmp7;

		tmp13=tmp9+tmp10;
		tmp14=tmp11+tmp12;

		tmp15=tmp13+tmp14;

		
	}
	return tmp15+tmp8;

}
data_type adder_tree1(unsigned int ch_in, unsigned int o_col,  int row0, int row1,
		        int row2, unsigned int o_row_buffer_output_index,
				data_type weight[3][3],data_type input_buffer[6][58]){
	data_type tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
	data_type tmp6,tmp7,tmp8,tmp9,tmp10,tmp11;
	data_type tmp12,tmp13,tmp14,tmp15,tmp16;
	for(ch_in=0;ch_in<1;ch_in++){
		tmp0=weight[0][0]*input_buffer[row0][o_col+0];
		tmp1=weight[1][0]*input_buffer[row1][o_col+0];
		//tmp9=tmp0+tmp1;
		tmp2=weight[2][0]*input_buffer[row2][o_col+0];
		tmp3=weight[0][1]*input_buffer[row0][o_col+1];
		//tmp10=tmp2+tmp3;
		tmp4=weight[1][1]*input_buffer[row1][o_col+1];
		tmp5=weight[2][1]*input_buffer[row2][o_col+1];
		//tmp11=tmp4+tmp5;
		tmp6=weight[0][2]*input_buffer[row0][o_col+2];
		tmp7=weight[1][2]*input_buffer[row1][o_col+2];
		//tmp12=tmp6+tmp7;
		tmp8=weight[2][2]*input_buffer[row2][o_col+2];



		tmp9=tmp0+tmp1;
		tmp10=tmp2+tmp3;
		tmp11=tmp4+tmp5;
		tmp12=tmp6+tmp7;

		tmp13=tmp9+tmp10;
		tmp14=tmp11+tmp12;

		tmp15=tmp13+tmp14;

		
	}
	return tmp15+tmp8;

}

data_type adder_tree2(unsigned int ch_in, unsigned int o_col,  int row0, int row1,
		        int row2, unsigned int o_row_buffer_output_index,
				data_type weight[3][3],data_type input_buffer[6][58]){
	data_type tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
	data_type tmp6,tmp7,tmp8,tmp9,tmp10,tmp11;
	data_type tmp12,tmp13,tmp14,tmp15,tmp16;
	for(ch_in=0;ch_in<1;ch_in++){
		tmp0=weight[0][0]*input_buffer[row0][o_col+0];
		tmp1=weight[1][0]*input_buffer[row1][o_col+0];
		//tmp9=tmp0+tmp1;
		tmp2=weight[2][0]*input_buffer[row2][o_col+0];
		tmp3=weight[0][1]*input_buffer[row0][o_col+1];
		//tmp10=tmp2+tmp3;
		tmp4=weight[1][1]*input_buffer[row1][o_col+1];
		tmp5=weight[2][1]*input_buffer[row2][o_col+1];
		//tmp11=tmp4+tmp5;
		tmp6=weight[0][2]*input_buffer[row0][o_col+2];
		tmp7=weight[1][2]*input_buffer[row1][o_col+2];
		//tmp12=tmp6+tmp7;
		tmp8=weight[2][2]*input_buffer[row2][o_col+2];



		tmp9=tmp0+tmp1;
		tmp10=tmp2+tmp3;
		tmp11=tmp4+tmp5;
		tmp12=tmp6+tmp7;

		tmp13=tmp9+tmp10;
		tmp14=tmp11+tmp12;

		tmp15=tmp13+tmp14;

		
	}
	return tmp15+tmp8;

}

data_type adder_tree3(unsigned int ch_in, unsigned int o_col,  int row0, int row1,
		        int row2, unsigned int o_row_buffer_output_index,
				data_type weight[3][3],data_type input_buffer[6][58]){
	data_type tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
	data_type tmp6,tmp7,tmp8,tmp9,tmp10,tmp11;
	data_type tmp12,tmp13,tmp14,tmp15,tmp16;
	for(ch_in=0;ch_in<1;ch_in++){
		tmp0=weight[0][0]*input_buffer[row0][o_col+0];
		tmp1=weight[1][0]*input_buffer[row1][o_col+0];
		//tmp9=tmp0+tmp1;
		tmp2=weight[2][0]*input_buffer[row2][o_col+0];
		tmp3=weight[0][1]*input_buffer[row0][o_col+1];
		//tmp10=tmp2+tmp3;
		tmp4=weight[1][1]*input_buffer[row1][o_col+1];
		tmp5=weight[2][1]*input_buffer[row2][o_col+1];
		//tmp11=tmp4+tmp5;
		tmp6=weight[0][2]*input_buffer[row0][o_col+2];
		tmp7=weight[1][2]*input_buffer[row1][o_col+2];
		//tmp12=tmp6+tmp7;
		tmp8=weight[2][2]*input_buffer[row2][o_col+2];



		tmp9=tmp0+tmp1;
		tmp10=tmp2+tmp3;
		tmp11=tmp4+tmp5;
		tmp12=tmp6+tmp7;

		tmp13=tmp9+tmp10;
		tmp14=tmp11+tmp12;

		tmp15=tmp13+tmp14;

		
	}
	return tmp15+tmp8;

}

//data_type tree_wrapper(data_type weight[4][3][3],data_type input_buffer[4][58][6],
//						unsigned int ch_in, unsigned int o_row, int col0, int col1, int col2, unsigned int o_col_buffer_output_index){
//	int tree_result0, tree_result1;
//	tree_result0=adder_tree(ch_in, o_row, col0,  col1,
//			        col2, o_col_buffer_output_index,
//					 weight[0],input_buffer[0]);
//	tree_result1=adder_tree1(ch_in, o_row, col0,  col1,
//			        col2, o_col_buffer_output_index,
//					 weight[1],input_buffer[1]);
//	return tree_result0+tree_result1;
//
//}

void row_based_engine1(data_type output_buffer[2][56],data_type weight[4][3][3],data_type input_buffer[4][6][58],
		                 unsigned int o_col, unsigned int ch_in,unsigned int w_col,unsigned int w_row,
						 unsigned int o_row_buffer_index, unsigned int o_row_buffer_output_index){
	#pragma HLS INLINE off


	int row0,row1,row2,offset;

	offset=o_row_buffer_index-3;


    if(offset==-1){
		row0=offset+6;
		row1=offset+1;
		row2=offset+2;
	}
	else if(offset==-2){
		row0=offset+6;
		row1=offset+7;
		row2=offset+2;
	}
	else if(offset==-3){
		row0=offset+6;
		row1=offset+7;
		row2=offset+8;
	}
	else{
		row0=offset;
		row1=offset+1;
		row2=offset+2;
	}

	for(o_col=0;o_col<56; o_col++){
		#pragma HLS PIPELINE
		#pragma HLS DEPENDENCE variable=output_buffer intra false
		#pragma HLS DEPENDENCE variable=input_buffer intra false
		#pragma HLS DEPENDENCE variable=weight intra false
		data_type tree_result0,tree_result1,tree_result2,tree_result3;

		//initiate adder tree module and force to use it once

//		#pragma HLS allocation instances=tree_wrapper limit=1 function
		#pragma HLS allocation instances=adder_tree limit=1 function
		#pragma HLS allocation instances=adder_tree1 limit=1 function
//		#pragma HLS allocation instances=adder_tree2 limit=1 function
//		#pragma HLS allocation instances=adder_tree3 limit=1 function
		tree_result0=adder_tree(ch_in, o_col, row0,  row1,
				        row2, o_row_buffer_output_index,
						 weight[0],input_buffer[0]);
		tree_result1=adder_tree1(ch_in, o_col, row0,  row1,
				        row2, o_row_buffer_output_index,
						 weight[1],input_buffer[1]);
//		tree_result0+=adder_tree(ch_in, o_col, row0,  row1,
//				        row2, o_row_buffer_output_index,
//						 weight[2],input_buffer[2]);
//		tree_result1+=adder_tree1(ch_in, o_col, row0,  row1,
//				        row2, o_row_buffer_output_index,
//						 weight[3],input_buffer[3]);

		tree_result2=tree_result0+tree_result1;
		output_buffer[o_row_buffer_output_index][o_col]=tree_result2;

		}





}



void conv3_3(data_type* flatten_input, data_type* flatten_weight, data_type* flatten_output ){
#pragma HLS INTERFACE axis  depth=430592 port=flatten_input
#pragma HLS INTERFACE axis  depth=294912 port=flatten_weight
#pragma HLS INTERFACE m_axi  depth=802816 port=flatten_output
//    data_type input[1][58][58];
	data_type input_buffer[4][6][58];
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=3
	data_type weight[1][4][3][3];
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight complete dim=3
#pragma HLS ARRAY_PARTITION variable=weight complete dim=4
//	data_type output[1][56][56];
	data_type output_buffer[1][2][56];
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
	for (unsigned int i=0;i<6;i++){
		for (unsigned int j=0;j<58;j++){
            #pragma HLS PIPELINE
			input_buffer[0][i][j]=flatten_input[58*i+j];
		}
	}
	//fill initial ofmap
	for (unsigned int i=0; i<56;i++){
		#pragma HLS PIPELINE
		output_buffer[0][0][i]=0;
		output_buffer[0][1][i]=0;
	}


   //computation
    unsigned int o_col, o_row, ch_in,ch_out,w_col,w_row;
    unsigned int ch_in_1,ch_out_1;






	for(ch_in_1=0; ch_in_1<32;ch_in_1++){

		input_buffer0(input_buffer, flatten_input,ch_in_1);


		for(ch_out_1=0; ch_out_1<256;ch_out_1++){

			weight_buffer0(weight,flatten_weight, ch_in_1,ch_out_1);

			computer_engine:for(ch_out=0;ch_out<1; ch_out++){
				unsigned int o_row_buffer_index=3;
				unsigned int o_row_buffer_output_index=0;
				unsigned int o_row_p1=55;
				#pragma HLS allocation instances=input_buffer1 limit=1 function
				for(o_row=0;o_row<56; o_row++){
					#pragma HLS DEPENDENCE variable=output_buffer intra false
					#pragma HLS DEPENDENCE variable=input_buffer intra false


					row_based_engine1(output_buffer[0], weight[0], input_buffer, o_row, ch_in,w_col, w_row,o_row_buffer_index, o_row_buffer_output_index);
					//potential optimization to reduce a couple of more cycles
//					input_buffer_wrapper(input_buffer[0],input_buffer[1],flatten_input,
//						      ch_in_1, o_row, o_row_buffer_index);
					input_buffer1(input_buffer[0],flatten_input,
							      ch_in_1, o_row, o_row_buffer_index,0);
					input_buffer1(input_buffer[1],flatten_input,
								  ch_in_1, o_row, o_row_buffer_index,1);
//					input_buffer1(input_buffer[2],flatten_input,
//							      ch_in_1, o_row, o_row_buffer_index,2);
//					input_buffer1(input_buffer[3],flatten_input,
//								  ch_in_1, o_row, o_row_buffer_index,3);
					//write the row before the current row
				    output_buffer0(output_buffer,flatten_output, ch_out_1, o_row_p1, 1-o_row_buffer_output_index);

					if(o_row_buffer_output_index==0)
						o_row_buffer_output_index++;
					else
						o_row_buffer_output_index=0;


					if(o_row_p1==55)
						o_row_p1=0;
					else
						o_row_p1++;

					o_row_buffer_index++;
					if (o_row_buffer_index>5)
						o_row_buffer_index=0;

				}

			    //output_buffer0(output_buffer,flatten_output, ch_out_1,  o_row-1, 1-o_row_buffer_output_index);
			}

    	}
		//output_buffer0(output_buffer,flatten_output, 255, 55, 1);

    }



}
