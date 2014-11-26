
#include <iostream>
#include <math.h>

const int nrow = 28;
const int ncol = 28;

const int nrow_output = 24;
const int ncol_output = 24;

const int nrow_kernel = 5;
const int ncol_kernel = 5;

const int nfeaturemap=6;
double sums[100];


inline void run1(double ** output_collapsed, double ** kernel_collapsed, double ** img){
	for(int i_epoch=0;i_epoch<1000;i_epoch++){

		double * poutput = &output_collapsed[0][0];
		for(int r=0;r<nrow_output;r++){
			for(int c=0;c<ncol_output;c++){

				for(int i=0;i<nfeaturemap;i++){
					sums[i] = 0.0;
				}

				double * pweight = &kernel_collapsed[0][0];
				for(int ir=r;ir<r+nrow_kernel;ir++){
					for(int ic=c;ic<c+ncol_kernel;ic++){
						for(int i=0;i<nfeaturemap;i++){
							sums[i] += pweight[i] * img[ir][ic];
						}
						pweight += nfeaturemap;
					}
				}

				for(int i=0;i<nfeaturemap;i++){
					poutput[i] = tanh(sums[i]);
				}

				poutput += nfeaturemap;
			}
		}
	}
}
