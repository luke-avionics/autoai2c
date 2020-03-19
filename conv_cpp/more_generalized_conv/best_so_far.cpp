#include "conv3x3.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;
const int Tr = 4, Tc = 56, Tm = 4, Tn = 16;
//num read ports = num of bram banks


void read_ifmap( data_type feature_temp[Tn][Tr+K-S][Tc+K-S], const data_type feature[N*H*H], int tr, int ti, int tc){
#pragma HLS INLINE off
	int trr, tcc, tii;
	for (tii = 0; tii < Tn; tii++) {
		for (trr = 0; trr < Tr + K - S; trr++) {
			for (tcc = 0; tcc < Tc + K - S; tcc++) {
				#pragma HLS PIPELINE
				feature_temp[tii][trr][tcc] = feature[(tii+ti)*H*H + (tr+trr)*H +tc+ tcc];
			}
		}
	}

}
void read_we(data_type weight_temp[Tm][Tn][K][K],const data_type weight[M*N*K*K], int ti,int to){
#pragma HLS INLINE
	int too, tii ,i, j;
	loop_load_weight:
	for (too = 0; too < Tm; ++too) {
		for (tii = 0; tii < Tn; ++tii) {
			for (i = 0; i < K; ++i) {
				for (j = 0; j < K; ++j) {
					#pragma HLS PIPELINE
					weight_temp[too][tii][i][j] = weight[(too + to)*N*K*K + (tii + ti)*K*K + i*K + j];
				}
			}
		}
	}

}


//data_type adder_tree(data_type weight_temp[K][K], data_type feature_temp[Tr+K-S][Tc+K-S], int trr, int tcc ){
//#pragma HLS INLINE off
//
//
//		data_type tmp0;
//		int i, j;
//		for(i=0;i<K;i++)
//			#pragma HLS UNROLL
//			for(j=0;j<K;j++)
//				#pragma HLS UNROLL
//				tmp0+=weight_temp[i][j]*feature_temp[trr + i][tcc + j];
//
//	return tmp0;
//
//}

data_type adder_tree(data_type weight_temp[K][K], data_type feature_temp[Tr+K-S][Tc+K-S], int trr, int tcc ){
#pragma HLS INLINE off
		data_type tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
		data_type tmp6,tmp7,tmp8,tmp9,tmp10,tmp11;
		data_type tmp12,tmp13,tmp14,tmp15,tmp16;

			tmp0=weight_temp[0][0]*feature_temp[trr + 0][tcc + 0];
			tmp1=weight_temp[1][0]*feature_temp[trr + 1][tcc + 0];
			//tmp9=tmp0+tmp1;
			tmp2=weight_temp[2][0]*feature_temp[trr + 2][tcc + 0];
			tmp3=weight_temp[0][1]*feature_temp[trr + 0][tcc + 1];
			//tmp10=tmp2+tmp3;
			tmp4=weight_temp[1][1]*feature_temp[trr + 1][tcc + 1];
			tmp5=weight_temp[2][1]*feature_temp[trr + 2][tcc + 1];
			//tmp11=tmp4+tmp5;
			tmp6=weight_temp[0][2]*feature_temp[trr + 0][tcc + 2];
			tmp7=weight_temp[1][2]*feature_temp[trr + 1][tcc + 2];
			//tmp12=tmp6+tmp7;
			tmp8=weight_temp[2][2]*feature_temp[trr + 2][tcc + 2];


			tmp9=tmp0+tmp1;
			tmp10=tmp2+tmp3;
			tmp11=tmp4+tmp5;
			tmp12=tmp6+tmp7;

			tmp13=tmp9+tmp10;
			tmp14=tmp11+tmp12;

			tmp15=tmp13+tmp14;

	return tmp15+tmp8;

}




//void comp_engine(int too, int trr, int tcc, int tii,
//				 data_type weight_temp[Tm][Tn][K][K], data_type feature_temp[Tn][Tr+K-S][Tc+K-S],data_type output_core_temp[Tm][Tr][Tc]){
//#pragma HLS INLINE off
//	data_type tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
//	data_type tmp7,tmp8,tmp9,tmp10,tmp11,tmp12,tmp13,tmp14;
//	for (too = 0; too < Tm; ++too) {
//		for (tcc = 0; tcc < Tc; ++tcc) {
//			for (trr = 0; trr < Tr; ++trr) {
//				#pragma HLS UNROLL
//				tmp0=0;
//				for (tii = 0; tii < Tn; ++tii) {
//					#pragma HLS UNROLL
//						//#pragma HLS allocation instances=adder_tree limit=4 function
//						tmp0+=adder_tree(weight_temp[too][tii],feature_temp[tii],trr,tcc);
//				}
//				output_core_temp[too][trr][tcc]+=tmp0;
//				}
//		}
//	}
//
//}

void comp_engine(
				 data_type weight_temp[Tm][Tn][K][K], data_type feature_temp[Tn][Tr+K-S][Tc+K-S],data_type output_core_temp[Tm][Tr][Tc]){
#pragma HLS INLINE off
	int too, tcc, tii, trr;
	data_type tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
	data_type tmp7,tmp8,tmp9,tmp10,tmp11,tmp12,tmp13,tmp14;
	//TODO: balanced unrolling input channel and output channel
		for (tcc = 0; tcc < Tc; ++tcc) {
#pragma HLS PIPELINE
			for (too = 0; too < Tm; ++too) {
			#pragma HLS UNROLL
			for (trr = 0; trr < Tr; ++trr) {
				#pragma HLS UNROLL
				tmp0=0;
				for (tii = 0; tii < Tn; ++tii) {
					#pragma HLS UNROLL
						//#pragma HLS allocation instances=adder_tree limit=4 function
						tmp0+=adder_tree(weight_temp[too][tii],feature_temp[tii],trr,tcc);
				}
				output_core_temp[too][trr][tcc]+=tmp0;
				}
		}
	}

}




void conv3_3(
	const data_type weight[M*N*K*K],
	const data_type feature[N*H*H],
	data_type output_core[M*C*C]) {
#pragma HLS INTERFACE m_axi depth=746496 port=output_core
#pragma HLS INTERFACE m_axi depth=401408 port=feature
#pragma HLS INTERFACE m_axi depth=294912 port=weight

	int i, j, k;
	int tr,tc;
	int row, col, to, ti, trr, tcc, too, tii;
    int tc_r, tr_r, to_r, ti_r;
	int lr_w=0,lr_i=0;
	int lr=0;
	data_type output_core_temp[Tm][Tr][Tc] = { 0 };
	#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=1
	#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=2
	//#pragma HLS ARRAY_PARTITION variable=output_core_temp complete dim=3
	#pragma HLS RESOURCE variable=output_core_temp core=RAM_2P_BRAM

	data_type weight_temp[Tm][Tn][K][K] = { 0}, feature_temp[Tn][Tr + K - S][Tc + K - S] = { 0 };
	#pragma HLS RESOURCE variable=feature_temp core=RAM_2P_BRAM
	//#pragma HLS RESOURCE variable=weight_temp core=RAM_2P_BRAM

	#pragma HLS ARRAY_PARTITION variable=feature_temp complete dim=1
	#pragma HLS ARRAY_PARTITION variable=feature_temp complete dim=2
	//#pragma HLS ARRAY_PARTITION variable=feature_temp complete dim=3
	#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=2
	#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=3
	#pragma HLS ARRAY_PARTITION variable=weight_temp complete dim=4

	data_type feature_temp1[Tn][Tr + K - S][Tc + K - S] = { 0 };
	#pragma HLS RESOURCE variable=feature_temp1 core=RAM_2P_BRAM
	#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=2
	//#pragma HLS ARRAY_PARTITION variable=feature_temp1 complete dim=3



	//TODO: buffer initialization
	read_ifmap(feature_temp, feature,0,0,0);
	//read_we(weight_temp,weight,0,0);
	for (tc=0; tc<C; tc+=Tc){
		for (tr=0; tr <C; tr+=Tr){
			loop_to:
			for (to = 0; to < M; to += Tm) {
				loop_ti:
				for (ti = 0; ti < N; ti += Tn) {
					#pragma HLS allocation instances=comp_engine limit=1 function
					#pragma HLS allocation instances=read_ifmap limit=1 function
					read_we(weight_temp,weight,ti,to);
					if (lr_i==0){


						//ping pong logic for index shifting
						//ti_r=ti;
						to_r=to;
						tc_r=tc;
						tr_r=tr;
						ti_r=ti+Tn;
						if (ti_r==N){
							ti_r=0;
							tr_r=tr+Tr;
							if(tr_r==C){
								tr_r=0;
								tc_r=tc_r+Tc;
								if(tc_r==C){
									tc_r=0;
								}
							}
						}
						//TODO: controlling port to switch
						read_ifmap(feature_temp1, feature,tr_r,ti_r,tc_r);
						comp_engine(weight_temp,feature_temp,output_core_temp);
						lr_i=1-lr_i;
					}
					else{


						//ping pong logic for index shifting
						//ti_r=ti;
						to_r=to;
						tc_r=tc;
						tr_r=tr;
						ti_r=ti+Tn;
						if (ti_r==N){
							ti_r=0;
							tr_r=tr+Tr;
							if(tr_r==C){
								tr_r=0;
								tc_r=tc_r+Tc;
								if(tc_r==C){
									tc_r=0;
								}
							}
						}
						//TODO: controlling port to switch
						read_ifmap(feature_temp, feature,tr_r,ti_r,tc_r);
						comp_engine(weight_temp,feature_temp1,output_core_temp);
						lr_i=1-lr_i;

					}

				}

			loop_store_output_feature_maps:
			for (too = 0; too < Tm; ++too) {
				for (trr = 0; trr < Tr; ++trr) {
					for (tcc = 0; tcc < Tc; ++tcc) {
						#pragma HLS PIPELINE
							output_core[(too + to)*C*C + (tr+trr)*C +tc+ tcc] += output_core_temp[too][trr][tcc];
							output_core_temp[too][trr][tcc] = 0;
						}
					}
				}

			}
		}
	}
};
