#include <iostream>
#include <math.h>
#include <cassert>
#include <cstdint>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <avxintrin.h>
#include <smmintrin.h>
#include <algorithm>

#include "timer.h"
#include <assert.h>

#include "Image.h"
#include "Corpus.h"


double STEPSIZE = 0.1;

inline double logadd(double lna, double lnb)
{
    if (lna == 1.0)
        return lnb;
    if (lnb == 1.0)
        return lna;
    
    double diff = lna - lnb;
    if (diff < 500.0)
        return log(exp(diff) + 1.0) + lnb;
    else
        return lna;
}


template<int _X, int _Y, int _Z, int _W>
class Cube{
public:

	float groundtruth[1024];

	const int X;
	const int Y;
	const int Z;
	const int W;

	float data[_W][_X][_Y][_Z] __attribute__ ((aligned (32)));
	
	float grad[_W][_X][_Y][_Z] __attribute__ ((aligned (32)));

	const int npartitions_Z;
	float partitions_Z[_W][_Z/8+(_Z%8!=0)][_X][_Y][8] __attribute__ ((aligned (32)));

	Cube(): 
		X(_X), Y(_Y), Z(_Z), W(_W),
		npartitions_Z(_Z/8+(_Z%8!=0))
	{
		//std::cout << X << " " << Y << " " << Z << " " << W << std::endl;
		/*
		std::cout << X << " " << Y << " " << Z << " " << W << std::endl;
		for(int w=0;w<W;w++){
			for(int x=0;x<X;x++){
				for(int y=0;y<Y;y++){
					for(int z=0;z<Z;z++){
						data[0][0][0][0] = 0.1;
					}
				}
			}
		}
		*/

	}

	void init(){
		std::cout << X << " " << Y << " " << Z << " " << W << std::endl;
		for(int w=0;w<_W;w++){
			for(int x=0;x<_X;x++){
				for(int y=0;y<_Y;y++){
					for(int z=0;z<_Z;z++){
						data[w][x][y][z] = (drand48()/2-1)/10;
					}
				}
			}
		}

	}

	void partition_Z(){
		for(int w=0;w<W;w++){
			for(int ipartition=0;ipartition<npartitions_Z;ipartition++){
				for(int x=0;x<X;x++){
					for(int y=0;y<Y;y++){
						for(int z=0;z<8;z++){
							partitions_Z[w][ipartition][x][y][ipartition*8+z] = 
								ipartition+z>=Z ? 0 : data[w][x][y][ipartition+z];
						}
					}
				}
			}
		}
	}

};


class Cube_xy1536{
public:
	int X;
	int Y;

	float *** data;
	float * _buf;

	void init(){
		_buf = (float *)_mm_malloc(X*Y*1536*sizeof(float), 32);
		data = new float**[X];
		for(int x=0;x<X;x++){
			data[x] = new float*[Y];
			for(int y=0;y<Y;y++){
				data[x][y] = &_buf[(x*Y+y)*1536];
			}
		}
	}
};


class Cube_xy192{
public:
	int X;
	int Y;

	float *** data;
	float * _buf;

	void init(){
		_buf = (float *)_mm_malloc(X*Y*192*sizeof(float), 32);
		data = new float**[X];
		for(int x=0;x<X;x++){
			data[x] = new float*[Y];
			for(int y=0;y<Y;y++){
				data[x][y] = &_buf[(x*Y+y)*192];
			}
		}
	}
};

class Cube_xy8{
public:
	int X;
	int Y;

	float *** data;
	float * _buf;

	void init(){
		_buf = (float *)_mm_malloc(X*Y*8*sizeof(float), 32);
		data = new float**[X];
		for(int x=0;x<X;x++){
			data[x] = new float*[Y];
			for(int y=0;y<Y;y++){
				data[x][y] = &_buf[(x*Y+y)*8];
			}
		}
	}
};

class Cube_xy1{
public:
	int X;
	int Y;

	float ** data;
	float * _buf;

	void init(){
		_buf = (float *)_mm_malloc(X*Y*sizeof(float), 32);
		data = new float*[X];
		for(int x=0;x<X;x++){
			data[x] = &_buf[x*Y];
			for(int y=0;y<Y;y++){
				data[x][y] = 0;
			}
		}
	}
};


inline float stupid_dot8(float * x, float * y){
	float sum = 0;
	for(int i=0;i<8;i++){
		sum += x[i]*y[i];
	}
	return sum;
}

/*
void stupid_conv(Cube_xy8 input, Cube_xy8 kernel, Cube_xy1 output){

	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			float sum = 0.0;
			for(int kx=0;kx<kernel.X;kx++){
				for(int ky=0;ky<kernel.Y;ky++){
					sum += stupid_dot8(input.data[x+kx][y+ky], kernel.data[kx][ky]);
				}
			}
			output.data[x][y] = sum;
		}
	}
}
*/


void stupid_conv(Cube_xy8 input, Cube_xy1536 kernel, Cube_xy192 output, int start, int end){

	int ct;
	float sum;

	for(int x=start;x<end;x++){
		for(int y=start;y<end;y++){
			float * pdata = input.data[x][y];
			for(int kx=0;kx<kernel.X;kx++){
				for(int ky=0;ky<kernel.Y;ky++){
					ct = 0;
					for(int i=0;i<1536;i+=8){
						sum = stupid_dot8(pdata, &kernel.data[kx][ky][i]);
						output.data[x+1-kx][y+1-ky][ct] += sum;
						ct ++;
					}
				}
			}
		}
	}

}

float aaa[10000000];

inline float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

void sse_conv(Cube_xy8 input, Cube_xy1536 kernel, Cube_xy192 output, int start, int end){

	if(kernel.X==3 && kernel.Y==3){

		register __m256 k0 asm ("ymm0");
		register __m256 k1 asm ("ymm1");
		register __m256 k2 asm ("ymm2");
		register __m256 k3 asm ("ymm3");
		register __m256 k4 asm ("ymm4");
		register __m256 k5 asm ("ymm5");
		register __m256 k6 asm ("ymm6");
		register __m256 k7 asm ("ymm7");
		register __m256 k8 asm ("ymm8");

		register __m256 data asm ("ymm9");

		register __m256 rs0 asm ("ymm10");
		register __m256 rs1 asm ("ymm11");
		register __m256 rs2 asm ("ymm12");
		register __m256 rs3 asm ("ymm12");

		register __m256 data2 asm ("ymm13");

		double s = 0.0;

		//k00 = _mm256_load_ps(kernel.data[0][0]);
		//k01 = _mm256_load_ps(kernel.data[0][1]);
		//k02 = _mm256_load_ps(kernel.data[0][2]);
		//k10 = _mm256_load_ps(kernel.data[1][0]);
		//k11 = _mm256_load_ps(kernel.data[1][1]);
		//k12 = _mm256_load_ps(kernel.data[1][2]);
		//k20 = _mm256_load_ps(kernel.data[2][0]);
		//k21 = _mm256_load_ps(kernel.data[2][1]);
		//k22 = _mm256_load_ps(kernel.data[2][2]);

		float * pkernel;
		float * pdata;
		double sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
		double sum;
		float * poutput;

		for(int kx=0;kx<kernel.X;kx++){
			for(int ky=0;ky<kernel.Y;ky++){

				//std::cout << kx << " " << ky << std::endl;

				pkernel = kernel.data[kx][ky];

				// following is a single segment
				for(int kseg=0;kseg<1536;kseg+=64){
					k0 = _mm256_load_ps(pkernel);
					k1 = _mm256_load_ps(pkernel+8);
					k2 = _mm256_load_ps(pkernel+16);
					k3 = _mm256_load_ps(pkernel+24);
					k4 = _mm256_load_ps(pkernel+32);
					k5 = _mm256_load_ps(pkernel+40);
					k6 = _mm256_load_ps(pkernel+48);
					k7 = _mm256_load_ps(pkernel+56);

					pdata = input.data[start][0];
					poutput = output.data[start+1-kx][start+1-ky];
					poutput -= 8;
					for(int x=start;x<end;x++){
						pdata += 8;
						poutput += 8;
						for(int y=start;y<end;y++){
							data = _mm256_load_ps(pdata);

							rs0 = _mm256_mul_ps(data, k0); 
							*(poutput) = sum8(rs0);

							rs1 = _mm256_mul_ps(data, k1); 
							*(poutput+1) = sum8(rs1);

							rs2 = _mm256_mul_ps(data, k2); 
							*(poutput+2) = sum8(rs2);

							rs3 = _mm256_mul_ps(data, k3); 
							*(poutput+3) = sum8(rs3);

							rs0 = _mm256_mul_ps(data, k4); 
							*(poutput+4) = sum8(rs0);

							rs1 = _mm256_mul_ps(data, k5); 
							*(poutput+5) = sum8(rs1);

							rs2 = _mm256_mul_ps(data, k6); 
							*(poutput+6) = sum8(rs2);

							rs3 = _mm256_mul_ps(data, k7); 
							*(poutput+7) = sum8(rs3);

							pdata += 8;
							poutput += 8;
						}
						pdata += 8;
						poutput += 8;
					}
					pkernel += 64;
				}

			}
		}




		/*
		k00 = _mm256_load_ps(kernel.data[0][0]);
		k01 = _mm256_load_ps(kernel.data[0][1]);
		k02 = _mm256_load_ps(kernel.data[0][2]);
		k10 = _mm256_load_ps(kernel.data[1][0]);
		k11 = _mm256_load_ps(kernel.data[1][1]);
		k12 = _mm256_load_ps(kernel.data[1][2]);
		k20 = _mm256_load_ps(kernel.data[2][0]);
		k21 = _mm256_load_ps(kernel.data[2][1]);
		k22 = _mm256_load_ps(kernel.data[2][2]);

		const int NELEMENT = input.X*input.Y;
		float * pdata = &input._buf[0]; 

		double s = 0.0;

		float * poutput = &output._buf[0];

		float * pcoutput = &output._buf[0];

		float sum00 = 0, sum01 = 0, sum02 = 0;
		
		float * pa = &aaa[0];
		data = _mm256_load_ps(pdata);  

		for(int i=0;i<NELEMENT-8;i+=8){

			data2 = _mm256_load_ps(pdata+8); 

			rs00 = _mm256_mul_ps(data, k00); 
			sum00 += sum8(rs00);

			rs01 = _mm256_mul_ps(data, k01); 
			sum01 += sum8(rs01);

			rs02 = _mm256_mul_ps(data, k02); 
			sum02 += sum8(rs02);
			s += sum00 + sum01 + sum02;

			rs00 = _mm256_mul_ps(data, k10); 
			sum00 += sum8(rs00);

			rs01 = _mm256_mul_ps(data, k11); 
			sum01 += sum8(rs01);

			rs02 = _mm256_mul_ps(data, k12); 
			sum02 += sum8(rs02);
			s += sum00 + sum01 + sum02;
			
			rs00 = _mm256_mul_ps(data, k20); 
			sum00 += sum8(rs00);

			rs01 = _mm256_mul_ps(data, k21); 
			sum01 += sum8(rs01);

			rs02 = _mm256_mul_ps(data, k22); 
			sum02 += sum8(rs02);
			s += sum00 + sum01 + sum02;

			data = data2;
			
			*pa = s;
			pa ++;

			pdata += 8;
		}
		*/

		std::cout << "~~~ " << s << std::endl;

	}else{
		assert(false);
	}

}


template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM, int strid, int start>
int back_strid_conv_vanilla_vanilla(Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
															 Cube<KERNELSIZE, KERNELSIZE, INPUTFM, OUTPUTFM> & kernel,
															 const Cube<OUTPUTSIZE, OUTPUTSIZE, OUTPUTFM, 1> & output){

	assert(KERNELSIZE % 2 != 0);
	const int KERNELBOUNDARY = KERNELSIZE/2;
	int outputx, outputy;

	float cvalue, grad;

	for(int outputfm=0;outputfm<OUTPUTFM;outputfm++){

		outputx = 0;
		for(int inputx=start;inputx<start+strid*OUTPUTSIZE;inputx+=strid){

			outputy = 0;
			for(int inputy=start;inputy<start+strid*OUTPUTSIZE;inputy+=strid){
				for(int inputfm=0;inputfm<INPUTFM;inputfm++){
					for(int kx=0;kx<KERNELSIZE;kx++){
						for(int ky=0;ky<KERNELSIZE;ky++){
							
							cvalue = output.data[0][outputx][outputy][outputfm];
							cvalue = 1.0 - (cvalue*cvalue);
							grad = output.grad[0][outputx][outputy][outputfm];

							kernel.data[outputfm][kx][ky][inputfm] = kernel.data[outputfm][kx][ky][inputfm]
								+ STEPSIZE * cvalue * grad * 
								input.data[0][inputx+kx-KERNELBOUNDARY][inputy+ky-KERNELBOUNDARY][inputfm];
						}
					}
				}

				outputy ++;
			}
			outputx ++;
		}
	}

	return OUTPUTFM*(INPUTSIZE/strid)*(INPUTSIZE/strid)*INPUTFM*kernel.X*kernel.Y*2;

}


template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM, int strid, int start>
int strid_conv_vanilla_vanilla(const Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
															 const Cube<KERNELSIZE, KERNELSIZE, INPUTFM, OUTPUTFM> & kernel,
															 Cube<OUTPUTSIZE, OUTPUTSIZE, OUTPUTFM, 1> & output){

	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = 0;
				output.grad[0][x][y][z] = 0;
			}
		}
	}

	assert(KERNELSIZE % 2 != 0);
	const int KERNELBOUNDARY = KERNELSIZE/2;
	int outputx, outputy;

	for(int outputfm=0;outputfm<OUTPUTFM;outputfm++){

		outputx = 0;
		for(int inputx=start;inputx<start+strid*OUTPUTSIZE;inputx+=strid){

			outputy = 0;
			for(int inputy=start;inputy<start+strid*OUTPUTSIZE;inputy+=strid){
				for(int inputfm=0;inputfm<INPUTFM;inputfm++){
					for(int kx=0;kx<KERNELSIZE;kx++){
						for(int ky=0;ky<KERNELSIZE;ky++){
							
							output.data[0][outputx][outputy][outputfm] +=
								kernel.data[outputfm][kx][ky][inputfm] * 
								input.data[0][inputx+kx-KERNELBOUNDARY][inputy+ky-KERNELBOUNDARY][inputfm];
						}
					}
				}

				outputy ++;
			}
			outputx ++;
		}
	}


	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = tanh(output.data[0][x][y][z]);
			}
		}
	}

	return OUTPUTFM*(INPUTSIZE/strid)*(INPUTSIZE/strid)*INPUTFM*kernel.X*kernel.Y*2;

}


template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM, int pad, int input_start>
int back_maxpool2x2_vanilla_vanilla(Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
															 const Cube<OUTPUTSIZE, OUTPUTSIZE, OUTPUTFM, 1> & output){

	assert(INPUTFM == OUTPUTFM);
	assert(KERNELSIZE % 2 != 0);
	const int KERNELBOUNDARY = KERNELSIZE/2;
	int outputx, outputy;
	const int strid = 2;

	for(int outputfm=0;outputfm<OUTPUTFM;outputfm++){

		outputx = pad;
		for(int inputx=input_start;inputx<INPUTSIZE;inputx+=strid){

			outputy = pad;
			for(int inputy=input_start;inputy<INPUTSIZE;inputy+=strid){

				input.data[0][inputx][inputy][outputfm] = 
					output.data[0][outputx][outputy][outputfm] == input.data[0][inputx][inputy][outputfm]?
					output.grad[0][outputx][outputy][outputfm] : 0;

				input.data[0][inputx+1][inputy][outputfm] = 
					output.data[0][outputx][outputy][outputfm] == input.data[0][inputx+1][inputy][outputfm]?
					output.grad[0][outputx][outputy][outputfm] : 0;

				input.data[0][inputx][inputy+1][outputfm] = 
					output.data[0][outputx][outputy][outputfm] == input.data[0][inputx][inputy+1][outputfm]?
					output.grad[0][outputx][outputy][outputfm] : 0;

				input.data[0][inputx][inputy+1][outputfm] = 
					output.data[0][outputx][outputy][outputfm] == input.data[0][inputx][inputy+1][outputfm]?
					output.grad[0][outputx][outputy][outputfm] : 0;

				outputy ++;
			}
			outputx ++;
		}
	}

	return OUTPUTFM*(INPUTSIZE/strid)*(INPUTSIZE/strid)*4;

}



template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM, int pad, int input_start>
int maxpool2x2_vanilla_vanilla(const Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
															 Cube<OUTPUTSIZE, OUTPUTSIZE, OUTPUTFM, 1> & output){

	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = 0;
				output.grad[0][x][y][z] = 0;
			}
		}
	}

	assert(INPUTFM == OUTPUTFM);
	assert(KERNELSIZE % 2 != 0);
	const int KERNELBOUNDARY = KERNELSIZE/2;
	int outputx, outputy;
	const int strid = 2;

	for(int outputfm=0;outputfm<OUTPUTFM;outputfm++){

		outputx = pad;
		for(int inputx=input_start;inputx<INPUTSIZE;inputx+=strid){

			outputy = pad;
			for(int inputy=input_start;inputy<INPUTSIZE;inputy+=strid){

				output.data[0][outputx][outputy][outputfm] =
					std::max(
						input.data[0][inputx][inputy][outputfm],
						std::max(
							input.data[0][inputx+1][inputy][outputfm],
							std::max(
								input.data[0][inputx][inputy+1][outputfm],
							input.data[0][inputx+1][inputy+1][outputfm]
							)
						)
					);

				outputy ++;
			}
			outputx ++;
		}
	}

	return OUTPUTFM*(INPUTSIZE/strid)*(INPUTSIZE/strid)*4;

}




template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM>
int conv_vanilla_vanilla(const Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
													const Cube<KERNELSIZE, KERNELSIZE, INPUTFM, OUTPUTFM> & kernel,
															Cube<OUTPUTSIZE, OUTPUTSIZE, OUTPUTFM, 1> & output){

	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = 0;
			}
		}
	}

	assert(KERNELSIZE % 2 != 0);
	assert(INPUTSIZE == OUTPUTSIZE);
	const int start = 0 + KERNELSIZE/2;
	const int end = OUTPUTSIZE - KERNELSIZE/2;
	const int KERNELBOUNDARY = KERNELSIZE/2;

	for(int outputfm=0;outputfm<OUTPUTFM;outputfm++){

		for(int outputx=start;outputx<end;outputx++){
			for(int outputy=start;outputy<end;outputy++){

				output.data[0][outputx][outputy][outputfm] = 0;
				for(int inputfm=0;inputfm<INPUTFM;inputfm++){
					for(int kx=0;kx<KERNELSIZE;kx++){
						for(int ky=0;ky<KERNELSIZE;ky++){

							assert(outputx >= 0);
							assert(outputx < OUTPUTSIZE);
							assert(outputy >= 0);
							assert(outputy < OUTPUTSIZE);
							assert(outputx+kx-KERNELBOUNDARY >= 0);
							assert(outputy+ky-KERNELBOUNDARY >= 0);
							assert(outputx+kx-KERNELBOUNDARY < INPUTSIZE);
							assert(outputy+ky-KERNELBOUNDARY < INPUTSIZE);

							output.data[0][outputx][outputy][outputfm] +=
								kernel.data[outputfm][kx][ky][inputfm] * input.data[0][outputx+kx-KERNELBOUNDARY][outputy+ky-KERNELBOUNDARY][inputfm];
						
							//if(outputx==1 && outputy == 1 && outputfm==0){
							//	std::cout << "van 0 0 0 0 " <<  
							//			"input=(" << (outputx+kx-1) << "," << (outputy+ky-1) << ")"
							//			<< " kernel=(" << kx << "," << ky << ") -> "
							//		<< (kernel.data[outputfm][kx][ky][inputfm] * input.data[0][outputx+kx-1][outputy+ky-1][inputfm]) << std::endl;
							//}

						}
					}
				}
				//std::cout << 0 << " " << outputx << " " << outputy << " " << outputfm << " -> " 
				//	<< output.data[0][outputx][outputy][outputfm] << std::endl;
			}
		}
	}

	return OUTPUTFM*(end-start)*(end-start)*INPUTFM*kernel.X*kernel.Y*2;

}


template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM>
int back_fullconn_parititionZ_vanilla(Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
															Cube<KERNELSIZE, KERNELSIZE, INPUTFM, OUTPUTFM> & kernel,
															const Cube<1, 1, OUTPUTFM, 1> & output){

	assert(OUTPUTFM % 8 ==0);
	assert(INPUTFM % 8 ==0);
	assert(KERNELSIZE % 2 != 0);
	const int start = (INPUTSIZE - KERNELSIZE)/2;
	const int end = start + KERNELSIZE;

	register __m256 outputfm1 asm ("ymm0");
	register __m256 outputfm2 asm ("ymm1");
	register __m256 outputfm3 asm ("ymm2");
	register __m256 outputfm4 asm ("ymm3");
	register __m256 outputfm5 asm ("ymm4");
	register __m256 outputfm6 asm ("ymm5");
	register __m256 outputfm7 asm ("ymm6");
	register __m256 outputfm8 asm ("ymm7");

	register __m256 data asm ("ymm8");

	register __m256 rs1 asm ("ymm9");
	register __m256 rs2 asm ("ymm10");
	register __m256 rs3 asm ("ymm11");
	register __m256 rs4 asm ("ymm12");

	register __m256 cvalues asm ("ymm13");
	register __m256 cgroundtruth asm ("ymm13");
	register __m256 mygrads;
	register __m256 grads asm ("ymm14");

	register int inputx;
	register int inputy;

	float _ones[8] = {1,1,1,1,1,1,1,1};
	float _stepsizes[8] = {STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE};

	register __m256 ones asm ("ymm15");
	register __m256 stepsizes;

	ones = _mm256_load_ps(_ones);
	stepsizes = _mm256_load_ps(_stepsizes);

	float pgrads[8];

	for(int kx=0;kx<KERNELSIZE;kx++){
		for(int ky=0;ky<KERNELSIZE;ky++){
			for(int inputfm=0;inputfm<INPUTFM;inputfm+=8){

				inputx = kx + start;
				inputy = ky + start;
				data = _mm256_load_ps(&input.data[0][inputx][inputy][inputfm]);
				mygrads = _mm256_setzero_ps();

				for(int outputfm=0;outputfm<OUTPUTFM;outputfm+=8){

					outputfm1 = _mm256_load_ps(&kernel.data[outputfm][kx][ky][inputfm]);
					outputfm2 = _mm256_load_ps(&kernel.data[outputfm+1][kx][ky][inputfm]);
					outputfm3 = _mm256_load_ps(&kernel.data[outputfm+2][kx][ky][inputfm]);
					outputfm4 = _mm256_load_ps(&kernel.data[outputfm+3][kx][ky][inputfm]);
					outputfm5 = _mm256_load_ps(&kernel.data[outputfm+4][kx][ky][inputfm]);
					outputfm6 = _mm256_load_ps(&kernel.data[outputfm+5][kx][ky][inputfm]);
					outputfm7 = _mm256_load_ps(&kernel.data[outputfm+6][kx][ky][inputfm]);
					outputfm8 = _mm256_load_ps(&kernel.data[outputfm+7][kx][ky][inputfm]);

					cvalues = _mm256_load_ps(&output.data[0][0][0][outputfm]);
					grads = _mm256_load_ps(&output.grad[0][0][0][outputfm]);

					// in grad, there is a term 1 - cvalue*cvalue;
					cvalues = _mm256_mul_ps(cvalues, cvalues);
					cvalues = _mm256_sub_ps(ones, cvalues);

					// all grad is (1-cvalues*cvalues) * grads
					grads = _mm256_mul_ps(ones, grads);

					_mm256_store_ps(pgrads, grads);

					mygrads = _mm256_add_ps(mygrads,
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[0]), outputfm1)
						)
					);

					outputfm1 = _mm256_add_ps(outputfm1, 
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[0]), data)
						)
					);


					mygrads = _mm256_add_ps(mygrads,
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[1]), outputfm2)
						)
					);

					outputfm2 = _mm256_add_ps(outputfm2, 
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[1]), data)
						)
					);

					mygrads = _mm256_add_ps(mygrads,
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[2]), outputfm3)
						)
					);

					outputfm3 = _mm256_add_ps(outputfm3, 
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[2]), data)
						)
					);

					mygrads = _mm256_add_ps(mygrads,
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[3]), outputfm4)
						)
					);

					outputfm4 = _mm256_add_ps(outputfm4, 
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[3]), data)
						)
					);

					mygrads = _mm256_add_ps(mygrads,
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[4]), outputfm5)
						)
					);

					outputfm5 = _mm256_add_ps(outputfm5, 
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[4]), data)
						)
					);

					mygrads = _mm256_add_ps(mygrads,
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[5]), outputfm6)
						)
					);

					outputfm6 = _mm256_add_ps(outputfm6, 
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[5]), data)
						)
					);

					mygrads = _mm256_add_ps(mygrads,
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[6]), outputfm7)
						)
					);

					outputfm7 = _mm256_add_ps(outputfm7, 
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[6]), data)
						)
					);			

					mygrads = _mm256_add_ps(mygrads,
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[7]), outputfm8)
						)
					);

					outputfm8 = _mm256_add_ps(outputfm8, 
						_mm256_mul_ps(stepsizes, 
							_mm256_mul_ps(_mm256_set1_ps(pgrads[7]), data)
						)
					);

					_mm256_stream_ps(&kernel.data[outputfm][kx][ky][inputfm], outputfm1);
					_mm256_stream_ps(&kernel.data[outputfm+1][kx][ky][inputfm], outputfm2);
					_mm256_stream_ps(&kernel.data[outputfm+2][kx][ky][inputfm], outputfm3);
					_mm256_stream_ps(&kernel.data[outputfm+3][kx][ky][inputfm], outputfm4);
					_mm256_stream_ps(&kernel.data[outputfm+4][kx][ky][inputfm], outputfm5);
					_mm256_stream_ps(&kernel.data[outputfm+5][kx][ky][inputfm], outputfm6);
					_mm256_stream_ps(&kernel.data[outputfm+6][kx][ky][inputfm], outputfm7);
					_mm256_stream_ps(&kernel.data[outputfm+7][kx][ky][inputfm], outputfm8);

				}

				_mm256_stream_ps(&input.grad[0][inputx][inputy][inputfm], mygrads);

			}
		}
	}

	return OUTPUTFM*INPUTFM*kernel.X*kernel.Y*6;
	//return 0;
}

template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM>
int fullconn_parititionZ_vanilla(const Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
															const Cube<KERNELSIZE, KERNELSIZE, INPUTFM, OUTPUTFM> & kernel,
															Cube<1, 1, OUTPUTFM, 1> & output){

	assert(OUTPUTFM % 8 ==0);
	assert(INPUTFM % 8 ==0);
	assert(KERNELSIZE % 2 != 0);
	const int start = (INPUTSIZE - KERNELSIZE)/2;
	const int end = start + KERNELSIZE;

	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = 0;
				output.grad[0][x][y][z] = 0;
			}
		}
	}

	register __m256 outputfm1 asm ("ymm0");
	register __m256 outputfm2 asm ("ymm1");
	register __m256 outputfm3 asm ("ymm2");
	register __m256 outputfm4 asm ("ymm3");
	register __m256 outputfm5 asm ("ymm4");
	register __m256 outputfm6 asm ("ymm5");
	register __m256 outputfm7 asm ("ymm6");
	register __m256 outputfm8 asm ("ymm7");

	register __m256 data asm ("ymm8");

	register __m256 rs1 asm ("ymm9");
	register __m256 rs2 asm ("ymm10");
	register __m256 rs3 asm ("ymm11");
	register __m256 rs4 asm ("ymm12");

	register int inputx;
	register int inputy;

	for(int kx=0;kx<KERNELSIZE;kx++){
		for(int ky=0;ky<KERNELSIZE;ky++){
			for(int inputfm=0;inputfm<INPUTFM;inputfm+=8){

				inputx = kx + start;
				inputy = ky + start;
				data = _mm256_load_ps(&input.data[0][inputx][inputy][inputfm]);

				for(int outputfm=0;outputfm<OUTPUTFM;outputfm+=8){

					outputfm1 = _mm256_load_ps(&kernel.data[outputfm][kx][ky][inputfm]);
					outputfm2 = _mm256_load_ps(&kernel.data[outputfm+1][kx][ky][inputfm]);
					outputfm3 = _mm256_load_ps(&kernel.data[outputfm+2][kx][ky][inputfm]);
					outputfm4 = _mm256_load_ps(&kernel.data[outputfm+3][kx][ky][inputfm]);
					outputfm5 = _mm256_load_ps(&kernel.data[outputfm+4][kx][ky][inputfm]);
					outputfm6 = _mm256_load_ps(&kernel.data[outputfm+5][kx][ky][inputfm]);
					outputfm7 = _mm256_load_ps(&kernel.data[outputfm+6][kx][ky][inputfm]);
					outputfm8 = _mm256_load_ps(&kernel.data[outputfm+7][kx][ky][inputfm]);

					//for(int inputx=start;inputx<end;inputx++){
					//	for(int inputy=start;inputy<end;inputy++){
					//std::cout << inputx << " " << inputy << " " << inputfm << std::endl;
					//rs1 = _mm256_mul_ps(data, outputfm1); 
					//output.data[0][inputx][inputy][outputfm] += sum8(rs1);

					rs1 = _mm256_mul_ps(data, outputfm1); 
					output.data[0][0][0][outputfm] += sum8(rs1);

					rs2 = _mm256_mul_ps(data, outputfm2); 
					output.data[0][0][0][outputfm+1] += sum8(rs2);

					rs3 = _mm256_mul_ps(data, outputfm3); 
					output.data[0][0][0][outputfm+2] += sum8(rs3);

					rs4 = _mm256_mul_ps(data, outputfm4); 
					output.data[0][0][0][outputfm+3] += sum8(rs4);

					rs1 = _mm256_mul_ps(data, outputfm5); 
					output.data[0][0][0][outputfm+4] += sum8(rs1);

					rs2 = _mm256_mul_ps(data, outputfm6); 
					output.data[0][0][0][outputfm+5] += sum8(rs2);

					rs3 = _mm256_mul_ps(data, outputfm7); 
					output.data[0][0][0][outputfm+6] += sum8(rs3);

					rs4 = _mm256_mul_ps(data, outputfm8);
					output.data[0][0][0][outputfm+7] += sum8(rs4);

					//if(inputx+1-kx==1 && inputy+1-ky == 1 && outputfm==0){
					//	std::cout << "sse 0 0 0 0 " << "input=(" << inputx << "," << inputy << ")"
					//		<< " kernel=(" << kx << "," << ky << ") -> " << sum8(rs1) << std::endl;
					//}

				}
			}
		}
	}

	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = tanh(output.data[0][x][y][z]);
			}
		}
	}


	return OUTPUTFM*INPUTFM*kernel.X*kernel.Y*2;
	//return 0;
}

void print(__m256 m){
	float a[8];
	_mm256_stream_ps(a, m);
	std::cout << "[";
	for(int i=0;i<8;i++){
		std::cout << a[i] << " ";
	}
	std::cout << "]" << std::endl;
}

template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM>
int back_softmax_parititionZ_vanilla(Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
															Cube<KERNELSIZE, KERNELSIZE, INPUTFM, OUTPUTFM> & kernel,
															const Cube<1, 1, OUTPUTFM, 1> & output){

	assert(OUTPUTFM % 8 ==0);
	assert(INPUTFM % 8 ==0);
	assert(KERNELSIZE % 2 != 0);
	const int start = (INPUTSIZE - KERNELSIZE)/2;
	const int end = start + KERNELSIZE;

	register __m256 outputfm1 asm ("ymm0");
	register __m256 outputfm2 asm ("ymm1");
	register __m256 outputfm3 asm ("ymm2");
	register __m256 outputfm4 asm ("ymm3");
	register __m256 outputfm5 asm ("ymm4");
	register __m256 outputfm6 asm ("ymm5");
	register __m256 outputfm7 asm ("ymm6");
	register __m256 outputfm8 asm ("ymm7");

	register __m256 data asm ("ymm8");

	register __m256 rs1 asm ("ymm9");
	register __m256 rs2 asm ("ymm10");
	register __m256 rs3 asm ("ymm11");
	register __m256 rs4 asm ("ymm12");

	register __m256 cvalues asm ("ymm13");
	register __m256 cgroundtruth asm ("ymm13");

	register __m256 mygrads;

	register int inputx;
	register int inputy;

	float _ones[8] = {1,1,1,1,1,1,1,1};
	float _stepsizes[8] = {STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE};

	register __m256 ones asm ("ymm15");
	register __m256 stepsizes;
	register __m256 zeros;
	zeros = _mm256_setzero_ps();

	ones = _mm256_load_ps(_ones);
	stepsizes = _mm256_load_ps(_stepsizes);

	//for(int i=0;i<1000;i++){
	//	std::cout << i << " -> " << output.groundtruth[i] << std::endl;
	//}

	for(int kx=0;kx<KERNELSIZE;kx++){
		for(int ky=0;ky<KERNELSIZE;ky++){
			for(int inputfm=0;inputfm<INPUTFM;inputfm+=8){

				inputx = kx + start;
				inputy = ky + start;
				data = _mm256_load_ps(&input.data[0][inputx][inputy][inputfm]);
				mygrads = _mm256_setzero_ps();

				for(int outputfm=0;outputfm<OUTPUTFM;outputfm+=8){

					//cvalues = _mm256_load_ps(&output.data[0][0][0][outputfm]);
					cgroundtruth = _mm256_load_ps(&output.groundtruth[outputfm]);

					outputfm1 = _mm256_load_ps(&kernel.data[outputfm][kx][ky][inputfm]);
					outputfm2 = _mm256_load_ps(&kernel.data[outputfm+1][kx][ky][inputfm]);
					outputfm3 = _mm256_load_ps(&kernel.data[outputfm+2][kx][ky][inputfm]);
					outputfm4 = _mm256_load_ps(&kernel.data[outputfm+3][kx][ky][inputfm]);
					outputfm5 = _mm256_load_ps(&kernel.data[outputfm+4][kx][ky][inputfm]);
					outputfm6 = _mm256_load_ps(&kernel.data[outputfm+5][kx][ky][inputfm]);
					outputfm7 = _mm256_load_ps(&kernel.data[outputfm+6][kx][ky][inputfm]);
					outputfm8 = _mm256_load_ps(&kernel.data[outputfm+7][kx][ky][inputfm]);

					// 	double grad_x = (label == groundtruth)*w - cvalue*w;
					mygrads = _mm256_add_ps(mygrads,
						_mm256_sub_ps(
							_mm256_mul_ps(output.groundtruth[outputfm]==1?ones:zeros, outputfm1),
							_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm]), outputfm1)
						)
					);

					//std::cout << "~" ; 
					//print(cvalues);
					//print(data);

					// 	double grad_w = (label == groundtruth)*x - cvalue*x;
					outputfm1 = _mm256_add_ps(outputfm1, 
						_mm256_mul_ps(stepsizes,
							_mm256_sub_ps(
								_mm256_mul_ps(output.groundtruth[outputfm]==1?ones:zeros, data),
								_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm]), data)
							)
						)
					);

					// 	double grad_x = (label == groundtruth)*w - cvalue*w;
					mygrads = _mm256_add_ps(mygrads,
						_mm256_sub_ps(
							_mm256_mul_ps(output.groundtruth[outputfm+1]==1?ones:zeros, outputfm2),
							_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+1]), outputfm2)
						)
					);

					// 	double grad_w = (label == groundtruth)*x - cvalue*x;
					outputfm2 = _mm256_add_ps(outputfm2, 
						_mm256_mul_ps(stepsizes,
							_mm256_sub_ps(
								_mm256_mul_ps(output.groundtruth[outputfm+1]==1?ones:zeros, data),
								_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+1]), data)
							)
						)
					);

					// 	double grad_x = (label == groundtruth)*w - cvalue*w;
					mygrads = _mm256_add_ps(mygrads,
						_mm256_sub_ps(
							_mm256_mul_ps(output.groundtruth[outputfm+2]==2?ones:zeros, outputfm3),
							_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+2]), outputfm3)
						)
					);

					// 	double grad_w = (label == groundtruth)*x - cvalue*x;
					outputfm3 = _mm256_add_ps(outputfm3, 
						_mm256_mul_ps(stepsizes,
							_mm256_sub_ps(
								_mm256_mul_ps(output.groundtruth[outputfm+2]==1?ones:zeros, data),
								_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+2]), data)
							)
						)
					);

					// 	double grad_x = (label == groundtruth)*w - cvalue*w;
					mygrads = _mm256_add_ps(mygrads,
						_mm256_sub_ps(
							_mm256_mul_ps(output.groundtruth[outputfm+3]==1?ones:zeros, outputfm4),
							_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+3]), outputfm4)
						)
					);

					// 	double grad_w = (label == groundtruth)*x - cvalue*x;
					outputfm4 = _mm256_add_ps(outputfm4, 
						_mm256_mul_ps(stepsizes,
							_mm256_sub_ps(
								_mm256_mul_ps(output.groundtruth[outputfm+3]==1?ones:zeros, data),
								_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+3]), data)
							)
						)
					);

					// 	double grad_x = (label == groundtruth)*w - cvalue*w;
					mygrads = _mm256_add_ps(mygrads,
						_mm256_sub_ps(
							_mm256_mul_ps(output.groundtruth[outputfm+4]==1?ones:zeros, outputfm5),
							_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+4]), outputfm5)
						)
					);

					// 	double grad_w = (label == groundtruth)*x - cvalue*x;
					outputfm5 = _mm256_add_ps(outputfm5, 
						_mm256_mul_ps(stepsizes,
							_mm256_sub_ps(
								_mm256_mul_ps(output.groundtruth[outputfm+4]==1?ones:zeros, data),
								_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+4]), data)
							)
						)
					);

						// 	double grad_x = (label == groundtruth)*w - cvalue*w;
					mygrads = _mm256_add_ps(mygrads,
						_mm256_sub_ps(
							_mm256_mul_ps(output.groundtruth[outputfm+5]==1?ones:zeros, outputfm6),
							_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+5]), outputfm6)
						)
					);

					// 	double grad_w = (label == groundtruth)*x - cvalue*x;
					outputfm6 = _mm256_add_ps(outputfm6, 
						_mm256_mul_ps(stepsizes,
							_mm256_sub_ps(
								_mm256_mul_ps(output.groundtruth[outputfm+5]==1?ones:zeros, data),
								_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+5]), data)
							)
						)
					);

					// 	double grad_x = (label == groundtruth)*w - cvalue*w;
					mygrads = _mm256_add_ps(mygrads,
						_mm256_sub_ps(
							_mm256_mul_ps(output.groundtruth[outputfm+6]==1?ones:zeros, outputfm7),
							_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+6]), outputfm7)
						)
					);


					// 	double grad_w = (label == groundtruth)*x - cvalue*x;
					outputfm7 = _mm256_add_ps(outputfm7, 
						_mm256_mul_ps(stepsizes,
							_mm256_sub_ps(
								_mm256_mul_ps(output.groundtruth[outputfm+6]==1?ones:zeros, data),
								_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+6]), data)
							)
						)
					);

					//if(outputfm+7==15){
					//	std::cout << "~~~~~" << std::endl;
					//	print(outputfm7);
					//}

					// 	double grad_x = (label == groundtruth)*w - cvalue*w;
					mygrads = _mm256_add_ps(mygrads,
						_mm256_sub_ps(
							_mm256_mul_ps(output.groundtruth[outputfm+7]==1?ones:zeros, outputfm8),
							_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+7]), outputfm8)
						)
					);

					/*
					if(outputfm+7==15){
						std::cout << "-----" << std::endl;
						print(cgroundtruth);
						print(outputfm8);
						print(data);
						print(cvalues);
						std::cout << "change " ;
						print(_mm256_mul_ps(stepsizes,
							_mm256_sub_ps(
								_mm256_mul_ps(cgroundtruth, data),
								_mm256_mul_ps(cvalues, data)
							)
						));
						std::cout << "_mm256_mul_ps(cvalues, data) " ;
						print(_mm256_mul_ps(cvalues, data));
						std::cout << "_mm256_mul_ps(cgroundtruth, data)";
						print(_mm256_mul_ps(cgroundtruth, data));
						std::cout << "_mm256_sub_ps ";
						print(_mm256_sub_ps(
								_mm256_mul_ps(cgroundtruth, data),
								_mm256_mul_ps(cvalues, data)
							));
					}
					*/

					// 	double grad_w = (label == groundtruth)*x - cvalue*x;
					outputfm8 = _mm256_add_ps(outputfm8, 
						_mm256_mul_ps(stepsizes,
							_mm256_sub_ps(
								_mm256_mul_ps(output.groundtruth[outputfm+7]==1?ones:zeros, data),
								_mm256_mul_ps(_mm256_set1_ps(output.data[0][0][0][outputfm+7]), data)
							)
						)
					);

					_mm256_stream_ps(&kernel.data[outputfm][kx][ky][inputfm], outputfm1);
					_mm256_stream_ps(&kernel.data[outputfm+1][kx][ky][inputfm], outputfm2);
					_mm256_stream_ps(&kernel.data[outputfm+2][kx][ky][inputfm], outputfm3);
					_mm256_stream_ps(&kernel.data[outputfm+3][kx][ky][inputfm], outputfm4);
					_mm256_stream_ps(&kernel.data[outputfm+4][kx][ky][inputfm], outputfm5);
					_mm256_stream_ps(&kernel.data[outputfm+5][kx][ky][inputfm], outputfm6);
					_mm256_stream_ps(&kernel.data[outputfm+6][kx][ky][inputfm], outputfm7);
					_mm256_stream_ps(&kernel.data[outputfm+7][kx][ky][inputfm], outputfm8);


				}

				_mm256_stream_ps(&input.grad[0][inputx][inputy][inputfm], mygrads);
			}
		}
	}

	return OUTPUTFM*INPUTFM*kernel.X*kernel.Y*9;
	//return 0;
}


template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM>
int softmax_parititionZ_vanilla(const Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
															const Cube<KERNELSIZE, KERNELSIZE, INPUTFM, OUTPUTFM> & kernel,
															Cube<1, 1, OUTPUTFM, 1> & output){

	assert(OUTPUTFM % 8 ==0);
	assert(INPUTFM % 8 ==0);
	assert(KERNELSIZE % 2 != 0);
	const int start = (INPUTSIZE - KERNELSIZE)/2;
	const int end = start + KERNELSIZE;

	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = 0;
				output.grad[0][x][y][z] = 0;
			}
		}
	}

	register __m256 outputfm1 asm ("ymm0");
	register __m256 outputfm2 asm ("ymm1");
	register __m256 outputfm3 asm ("ymm2");
	register __m256 outputfm4 asm ("ymm3");
	register __m256 outputfm5 asm ("ymm4");
	register __m256 outputfm6 asm ("ymm5");
	register __m256 outputfm7 asm ("ymm6");
	register __m256 outputfm8 asm ("ymm7");

	register __m256 data asm ("ymm8");

	register __m256 rs1 asm ("ymm9");
	register __m256 rs2 asm ("ymm10");
	register __m256 rs3 asm ("ymm11");
	register __m256 rs4 asm ("ymm12");

	register int inputx;
	register int inputy;

	for(int kx=0;kx<KERNELSIZE;kx++){
		for(int ky=0;ky<KERNELSIZE;ky++){
			for(int inputfm=0;inputfm<INPUTFM;inputfm+=8){

				inputx = kx + start;
				inputy = ky + start;
				data = _mm256_load_ps(&input.data[0][inputx][inputy][inputfm]);

				for(int outputfm=0;outputfm<OUTPUTFM;outputfm+=8){

					outputfm1 = _mm256_load_ps(&kernel.data[outputfm][kx][ky][inputfm]);
					outputfm2 = _mm256_load_ps(&kernel.data[outputfm+1][kx][ky][inputfm]);
					outputfm3 = _mm256_load_ps(&kernel.data[outputfm+2][kx][ky][inputfm]);
					outputfm4 = _mm256_load_ps(&kernel.data[outputfm+3][kx][ky][inputfm]);
					outputfm5 = _mm256_load_ps(&kernel.data[outputfm+4][kx][ky][inputfm]);
					outputfm6 = _mm256_load_ps(&kernel.data[outputfm+5][kx][ky][inputfm]);
					outputfm7 = _mm256_load_ps(&kernel.data[outputfm+6][kx][ky][inputfm]);
					outputfm8 = _mm256_load_ps(&kernel.data[outputfm+7][kx][ky][inputfm]);

					//for(int inputx=start;inputx<end;inputx++){
					//	for(int inputy=start;inputy<end;inputy++){
					//std::cout << inputx << " " << inputy << " " << inputfm << std::endl;
					//rs1 = _mm256_mul_ps(data, outputfm1); 
					//output.data[0][inputx][inputy][outputfm] += sum8(rs1);

					rs1 = _mm256_mul_ps(data, outputfm1); 
					output.data[0][0][0][outputfm] += sum8(rs1);

					rs2 = _mm256_mul_ps(data, outputfm2); 
					output.data[0][0][0][outputfm+1] += sum8(rs2);

					rs3 = _mm256_mul_ps(data, outputfm3); 
					output.data[0][0][0][outputfm+2] += sum8(rs3);

					rs4 = _mm256_mul_ps(data, outputfm4); 
					output.data[0][0][0][outputfm+3] += sum8(rs4);

					rs1 = _mm256_mul_ps(data, outputfm5); 
					output.data[0][0][0][outputfm+4] += sum8(rs1);

					rs2 = _mm256_mul_ps(data, outputfm6); 
					output.data[0][0][0][outputfm+5] += sum8(rs2);

					rs3 = _mm256_mul_ps(data, outputfm7); 
					output.data[0][0][0][outputfm+6] += sum8(rs3);

					rs4 = _mm256_mul_ps(data, outputfm8);
					output.data[0][0][0][outputfm+7] += sum8(rs4);

					//if(inputx+1-kx==1 && inputy+1-ky == 1 && outputfm==0){
					//	std::cout << "sse 0 0 0 0 " << "input=(" << inputx << "," << inputy << ")"
					//		<< " kernel=(" << kx << "," << ky << ") -> " << sum8(rs1) << std::endl;
					//}

				}
			}
		}
	}

	float sum = -100000;

	for(int outputfm=0;outputfm<OUTPUTFM;outputfm++){
			sum = logadd(sum, output.data[0][0][0][outputfm]); 
	}

	//std::cout << "     " << 15 << "  " << output.data[0][0][0][15]
	//		<< "   " << exp(output.data[0][0][0][15]-sum) << std::endl;


	for(int outputfm=0;outputfm<OUTPUTFM;outputfm++){
		output.data[0][0][0][outputfm] = 
			exp(output.data[0][0][0][outputfm]-sum);
	}



	return OUTPUTFM*INPUTFM*kernel.X*kernel.Y*2;
	//return 0;
}


template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM>
long long back_conv_parititionZ_vanilla(Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
															Cube<KERNELSIZE, KERNELSIZE, INPUTFM, OUTPUTFM> & kernel,
															const Cube<OUTPUTSIZE, OUTPUTSIZE, OUTPUTFM, 1> & output){


	assert(INPUTSIZE == OUTPUTSIZE);
	assert(OUTPUTFM % 8 ==0);
	assert(INPUTFM % 8 ==0);
	assert(KERNELSIZE % 2 != 0);
	const int start = 0 + KERNELSIZE/2;
	const int end = OUTPUTSIZE - KERNELSIZE/2;
	const int KERNELBOUNDARY = KERNELSIZE/2;

	register __m256 outputfm1 asm ("ymm0");
	register __m256 outputfm2 asm ("ymm1");
	register __m256 outputfm3 asm ("ymm2");
	register __m256 outputfm4 asm ("ymm3");
	register __m256 outputfm5 asm ("ymm4");
	register __m256 outputfm6 asm ("ymm5");
	register __m256 outputfm7 asm ("ymm6");
	register __m256 outputfm8 asm ("ymm7");

	register __m256 data asm ("ymm8");

	register __m256 rs1 asm ("ymm9");
	register __m256 rs2 asm ("ymm10");
	register __m256 rs3 asm ("ymm11");
	register __m256 rs4 asm ("ymm12");

	register __m256 cvalues asm ("ymm13");
	register __m256 grads asm ("ymm14");

	register __m256 mygrads;

	float _ones[8] = {1,1,1,1,1,1,1,1};
	float _stepsizes[8] = {STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE,STEPSIZE};

	register __m256 ones asm ("ymm15");
	register __m256 stepsizes;

	ones = _mm256_load_ps(_ones);
	stepsizes = _mm256_load_ps(_stepsizes);

	float pgrads[8];

	for(int kx=0;kx<KERNELSIZE;kx++){
		for(int ky=0;ky<KERNELSIZE;ky++){
			for(int inputfm=0;inputfm<INPUTFM;inputfm+=8){
				for(int outputfm=0;outputfm<OUTPUTFM;outputfm+=8){

					outputfm1 = _mm256_load_ps(&kernel.data[outputfm][kx][ky][inputfm]);
					outputfm2 = _mm256_load_ps(&kernel.data[outputfm+1][kx][ky][inputfm]);
					outputfm3 = _mm256_load_ps(&kernel.data[outputfm+2][kx][ky][inputfm]);
					outputfm4 = _mm256_load_ps(&kernel.data[outputfm+3][kx][ky][inputfm]);
					outputfm5 = _mm256_load_ps(&kernel.data[outputfm+4][kx][ky][inputfm]);
					outputfm6 = _mm256_load_ps(&kernel.data[outputfm+5][kx][ky][inputfm]);
					outputfm7 = _mm256_load_ps(&kernel.data[outputfm+6][kx][ky][inputfm]);
					outputfm8 = _mm256_load_ps(&kernel.data[outputfm+7][kx][ky][inputfm]);

					for(int inputx=start;inputx<end;inputx++){
						for(int inputy=start;inputy<end;inputy++){

							//std::cout << inputx << " " << inputy << " " << inputfm << std::endl;

							mygrads = _mm256_setzero_ps();
							data = _mm256_load_ps(&input.data[0][inputx][inputy][inputfm]);
							cvalues = _mm256_load_ps(&output.data[0][inputx+KERNELBOUNDARY-kx][inputy+KERNELBOUNDARY-ky][outputfm]);
							grads = _mm256_load_ps(&output.grad[0][inputx+KERNELBOUNDARY-kx][inputy+KERNELBOUNDARY-ky][outputfm]);

							// in grad, there is a term 1 - cvalue*cvalue;
							cvalues = _mm256_mul_ps(cvalues, cvalues);
							cvalues = _mm256_sub_ps(ones, cvalues);
							// all grad is (1-cvalues*cvalues) * grads
							grads = _mm256_mul_ps(ones, grads);

							_mm256_store_ps(pgrads, grads);

							mygrads = _mm256_add_ps(mygrads,
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[0]), outputfm1)
								)
							);

							outputfm1 = _mm256_add_ps(outputfm1, 
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[0]), data)
								)
							);


							mygrads = _mm256_add_ps(mygrads,
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[1]), outputfm2)
								)
							);

							outputfm2 = _mm256_add_ps(outputfm2, 
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[1]), data)
								)
							);

							mygrads = _mm256_add_ps(mygrads,
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[2]), outputfm3)
								)
							);

							outputfm3 = _mm256_add_ps(outputfm3, 
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[2]), data)
								)
							);

							mygrads = _mm256_add_ps(mygrads,
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[3]), outputfm4)
								)
							);

							outputfm4 = _mm256_add_ps(outputfm4, 
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[3]), data)
								)
							);

							mygrads = _mm256_add_ps(mygrads,
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[4]), outputfm5)
								)
							);

							outputfm5 = _mm256_add_ps(outputfm5, 
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[4]), data)
								)
							);

							mygrads = _mm256_add_ps(mygrads,
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[5]), outputfm6)
								)
							);

							outputfm6 = _mm256_add_ps(outputfm6, 
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[5]), data)
								)
							);

							mygrads = _mm256_add_ps(mygrads,
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[6]), outputfm7)
								)
							);

							outputfm7 = _mm256_add_ps(outputfm7, 
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[6]), data)
								)
							);			

							mygrads = _mm256_add_ps(mygrads,
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[7]), outputfm8)
								)
							);

							outputfm8 = _mm256_add_ps(outputfm8, 
								_mm256_mul_ps(stepsizes, 
									_mm256_mul_ps(_mm256_set1_ps(pgrads[7]), data)
								)
							);

							_mm256_stream_ps(&input.grad[0][inputx][inputy][inputfm], mygrads);

						}
					}

					_mm256_stream_ps(&kernel.data[outputfm][kx][ky][inputfm], outputfm1);
					_mm256_stream_ps(&kernel.data[outputfm+1][kx][ky][inputfm], outputfm2);
					_mm256_stream_ps(&kernel.data[outputfm+2][kx][ky][inputfm], outputfm3);
					_mm256_stream_ps(&kernel.data[outputfm+3][kx][ky][inputfm], outputfm4);
					_mm256_stream_ps(&kernel.data[outputfm+4][kx][ky][inputfm], outputfm5);
					_mm256_stream_ps(&kernel.data[outputfm+5][kx][ky][inputfm], outputfm6);
					_mm256_stream_ps(&kernel.data[outputfm+6][kx][ky][inputfm], outputfm7);
					_mm256_stream_ps(&kernel.data[outputfm+7][kx][ky][inputfm], outputfm8);

				}
			}
		}
	}

	return 1L*OUTPUTFM*(end-start)*(end-start)*INPUTFM*kernel.X*kernel.Y*6;

}


template<int INPUTSIZE, int KERNELSIZE, int OUTPUTSIZE, int INPUTFM, int OUTPUTFM>
int conv_parititionZ_vanilla(const Cube<INPUTSIZE, INPUTSIZE, INPUTFM, 1> & input,
															const Cube<KERNELSIZE, KERNELSIZE, INPUTFM, OUTPUTFM> & kernel,
															Cube<OUTPUTSIZE, OUTPUTSIZE, OUTPUTFM, 1> & output){

	assert(INPUTSIZE == OUTPUTSIZE);
	assert(OUTPUTFM % 8 ==0);
	assert(INPUTFM % 8 ==0);
	assert(KERNELSIZE % 2 != 0);
	const int start = 0 + KERNELSIZE/2;
	const int end = OUTPUTSIZE - KERNELSIZE/2;
	const int KERNELBOUNDARY = KERNELSIZE/2;

	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = 0;
				output.grad[0][x][y][z] = 0;
			}
		}
	}

	register __m256 outputfm1 asm ("ymm0");
	register __m256 outputfm2 asm ("ymm1");
	register __m256 outputfm3 asm ("ymm2");
	register __m256 outputfm4 asm ("ymm3");
	register __m256 outputfm5 asm ("ymm4");
	register __m256 outputfm6 asm ("ymm5");
	register __m256 outputfm7 asm ("ymm6");
	register __m256 outputfm8 asm ("ymm7");

	register __m256 data asm ("ymm8");

	register __m256 rs1 asm ("ymm9");
	register __m256 rs2 asm ("ymm10");
	register __m256 rs3 asm ("ymm11");
	register __m256 rs4 asm ("ymm12");

	for(int kx=0;kx<KERNELSIZE;kx++){
		for(int ky=0;ky<KERNELSIZE;ky++){
			for(int inputfm=0;inputfm<INPUTFM;inputfm+=8){
				for(int outputfm=0;outputfm<OUTPUTFM;outputfm+=8){

					outputfm1 = _mm256_load_ps(&kernel.data[outputfm][kx][ky][inputfm]);
					outputfm2 = _mm256_load_ps(&kernel.data[outputfm+1][kx][ky][inputfm]);
					outputfm3 = _mm256_load_ps(&kernel.data[outputfm+2][kx][ky][inputfm]);
					outputfm4 = _mm256_load_ps(&kernel.data[outputfm+3][kx][ky][inputfm]);
					outputfm5 = _mm256_load_ps(&kernel.data[outputfm+4][kx][ky][inputfm]);
					outputfm6 = _mm256_load_ps(&kernel.data[outputfm+5][kx][ky][inputfm]);
					outputfm7 = _mm256_load_ps(&kernel.data[outputfm+6][kx][ky][inputfm]);
					outputfm8 = _mm256_load_ps(&kernel.data[outputfm+7][kx][ky][inputfm]);

					for(int inputx=start;inputx<end;inputx++){
						for(int inputy=start;inputy<end;inputy++){

							//std::cout << inputx << " " << inputy << " " << inputfm << std::endl;

							data = _mm256_load_ps(&input.data[0][inputx][inputy][inputfm]);

							
							//rs1 = _mm256_mul_ps(data, outputfm1); 
							//output.data[0][inputx][inputy][outputfm] += sum8(rs1);

							rs1 = _mm256_mul_ps(data, outputfm1); 
							output.data[0][inputx+KERNELBOUNDARY-kx][inputy+KERNELBOUNDARY-ky][outputfm] += sum8(rs1);

							rs2 = _mm256_mul_ps(data, outputfm2); 
							output.data[0][inputx+KERNELBOUNDARY-kx][inputy+KERNELBOUNDARY-ky][outputfm+1] += sum8(rs2);

							rs3 = _mm256_mul_ps(data, outputfm3); 
							output.data[0][inputx+KERNELBOUNDARY-kx][inputy+KERNELBOUNDARY-ky][outputfm+2] += sum8(rs3);

							rs4 = _mm256_mul_ps(data, outputfm4); 
							output.data[0][inputx+KERNELBOUNDARY-kx][inputy+KERNELBOUNDARY-ky][outputfm+3] += sum8(rs4);

							rs1 = _mm256_mul_ps(data, outputfm5); 
							output.data[0][inputx+KERNELBOUNDARY-kx][inputy+KERNELBOUNDARY-ky][outputfm+4] += sum8(rs1);

							rs2 = _mm256_mul_ps(data, outputfm6); 
							output.data[0][inputx+KERNELBOUNDARY-kx][inputy+KERNELBOUNDARY-ky][outputfm+5] += sum8(rs2);

							rs3 = _mm256_mul_ps(data, outputfm7); 
							output.data[0][inputx+KERNELBOUNDARY-kx][inputy+KERNELBOUNDARY-ky][outputfm+6] += sum8(rs3);

							rs4 = _mm256_mul_ps(data, outputfm8);
							output.data[0][inputx+KERNELBOUNDARY-kx][inputy+KERNELBOUNDARY-ky][outputfm+7] += sum8(rs4);

							//if(inputx+1-kx==1 && inputy+1-ky == 1 && outputfm==0){
							//	std::cout << "sse 0 0 0 0 " << "input=(" << inputx << "," << inputy << ")"
							//		<< " kernel=(" << kx << "," << ky << ") -> " << sum8(rs1) << std::endl;
							//}

						}
					}

				}
			}
		}
	}

	for(int x=0;x<start;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = 0;
			}
		}
	}

	for(int x=end;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = 0;
			}
		}
	}

	
	for(int x=0;x<output.X;x++){
		for(int y=0;y<start;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = 0;
			}
		}
	}

	for(int x=0;x<output.X;x++){
		for(int y=end;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = 0;
			}
		}
	}

	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<output.Z;z++){
				output.data[0][x][y][z] = tanh(output.data[0][x][y][z]);
				//std::cout << output.data[0][x][y][z] << std::endl;
			}
		}
	}

	return OUTPUTFM*(end-start)*(end-start)*INPUTFM*kernel.X*kernel.Y*2;
	//return 0;
}


int main_aaa(int argc, char ** argv){

	Timer t;

	const int ndataword = 43264/8;

	float data[43264] __attribute__ ((aligned (32)));;
	float data1[43264] __attribute__ ((aligned (32)));;
	float data2[43264] __attribute__ ((aligned (32)));;
	float data3[43264] __attribute__ ((aligned (32)));;
	float data4[43264] __attribute__ ((aligned (32)));;

	float models[4096][43264] __attribute__ ((aligned (32)));;

	//float models2[43264][4096] __attribute__ ((aligned (32)));;

	float models2[ndataword][4096*8] __attribute__ ((aligned (32)));;

	// models2 like this.
	// for each [43264/8] data word, chunk model word
	//

	float rs[4096] __attribute__ ((aligned (32)));;
	float sum = 0.0;

	for(int i=0;i<43264;i++){
		data[i] = drand48();
	}
	for(int i=0;i<43264;i++){
		for(int j=0;j<4096;j++){
			models[j][i] = drand48();
		}
	}

	for(int j=0;j<4096;j++){
		rs[j] = 0;
	}


	/*
	t.restart();
	int ct = -1;
	int ccount = 0;
	for(int i=0;i<43264;i++){
		for(int j=0;j<4096;j++){
			models2[ct][ccount] = models[j][i];
			ccount ++;
		}
	}
	std::cout << "elapsed " << t.elapsed() << "    " << models2[0][0] << std::endl;
	*/

	/*
	t.restart();
	for(int i=0;i<43264;i++){
		for(int j=0;j<4096;j++){
			rs[j] += data[i] * models[j][i];
		}
	}
	std::cout << "elapsed " << t.elapsed() << std::endl;
	*/
	register __m256 outputfm1 asm ("ymm0");
	register __m256 outputfm2 asm ("ymm1");
	register __m256 outputfm3 asm ("ymm2");
	register __m256 outputfm4 asm ("ymm3");
	register __m256 outputfm5 asm ("ymm4");
	register __m256 outputfm6 asm ("ymm5");
	register __m256 outputfm7 asm ("ymm6");
	register __m256 outputfm8 asm ("ymm7");
	
	register __m256 d1 asm ("ymm13");
	register __m256 d2 asm ("ymm14");
	register __m256 d3 asm ("ymm15");
	register __m256 d4 asm ("ymm8");

	register __m256 rs1 asm ("ymm9");
	register __m256 rs2 asm ("ymm10");
	register __m256 rs3 asm ("ymm11");
	register __m256 rs4 asm ("ymm12");


	t.restart();
	int iword = 0;
	float * pmodel = &models2[0][0];

	for(int j=0;j<4096;j+=8){

	}

	std::cout << "~~~~~~~" << std::endl;

	for(int j=0;j<4096*43264;j+=64){
		outputfm1 = _mm256_load_ps(pmodel);
		outputfm2 = _mm256_load_ps(pmodel+8);
		outputfm3 = _mm256_load_ps(pmodel+16);
		outputfm4 = _mm256_load_ps(pmodel+24);
		outputfm5 = _mm256_load_ps(pmodel+32);
		outputfm6 = _mm256_load_ps(pmodel+40);
		outputfm7 = _mm256_load_ps(pmodel+48);
		outputfm8 = _mm256_load_ps(pmodel+56);

		for(int i=0;i<43264;i+=8){
			d1 = _mm256_load_ps(&data1[i]);
			//d2 = _mm256_load_ps(&data2[i]);
			//d3 = _mm256_load_ps(&data3[i]);
			//d4 = _mm256_load_ps(&data4[i]);

			rs1 = _mm256_mul_ps(d1, outputfm1); 
			rs[0] += sum8(rs1);

			rs2 = _mm256_mul_ps(d1, outputfm2); 
			rs[0+1] += sum8(rs2);

			rs3 = _mm256_mul_ps(d1, outputfm3); 
			rs[0+2] += sum8(rs3);

			rs4 = _mm256_mul_ps(d1, outputfm4); 
			rs[0+3] += sum8(rs4);

			rs1 = _mm256_mul_ps(d1, outputfm5); 
			rs[0+4] += sum8(rs1);

			rs2 = _mm256_mul_ps(d1, outputfm6); 
			rs[0+5] += sum8(rs2);

			rs3 = _mm256_mul_ps(d1, outputfm7); 
			rs[0+6] += sum8(rs3);

			rs4= _mm256_mul_ps(d1, outputfm8); 
			rs[0+7] += sum8(rs4);

			// 
			/*
			rs1 = _mm256_mul_ps(d2, outputfm1); 
			rs[0] += sum8(rs1);

			rs2 = _mm256_mul_ps(d2, outputfm2); 
			rs[0+1] += sum8(rs2);

			rs3 = _mm256_mul_ps(d2, outputfm3); 
			rs[0+2] += sum8(rs3);

			rs4 = _mm256_mul_ps(d2, outputfm4); 
			rs[0+3] += sum8(rs4);

			rs1 = _mm256_mul_ps(d2, outputfm5); 
			rs[0+4] += sum8(rs1);

			rs2 = _mm256_mul_ps(d2, outputfm6); 
			rs[0+5] += sum8(rs2);

			rs3 = _mm256_mul_ps(d2, outputfm7); 
			rs[0+6] += sum8(rs3);

			rs4 = _mm256_mul_ps(d2, outputfm8); 
			rs[0+7] += sum8(rs4);
			*/
		}

		pmodel += 64;

	}

	std::cout << "elapsed " << t.elapsed() << std::endl;




	/*
	t.restart();
	int iword = 0;
	float * pmodel = &models2[0][0];
	for(int i=0;i<43264;i+=8){
		d = _mm256_load_ps(&data[i]);

		for(int j=0;j<4096*8;j+=64){
			outputfm1 = _mm256_load_ps(pmodel);
			outputfm2 = _mm256_load_ps(pmodel+8);
			outputfm3 = _mm256_load_ps(pmodel+16);
			outputfm4 = _mm256_load_ps(pmodel+24);
			outputfm5 = _mm256_load_ps(pmodel+32);
			outputfm6 = _mm256_load_ps(pmodel+40);
			outputfm7 = _mm256_load_ps(pmodel+48);
			outputfm8 = _mm256_load_ps(pmodel+56);

			rs1 = _mm256_mul_ps(d, outputfm1); 
			rs[j] += sum8(rs1);

			rs2 = _mm256_mul_ps(d, outputfm2); 
			rs[j+1] += sum8(rs2);

			rs3 = _mm256_mul_ps(d, outputfm3); 
			rs[j+2] += sum8(rs3);

			rs4 = _mm256_mul_ps(d, outputfm4); 
			rs[j+3] += sum8(rs4);

			rs1 = _mm256_mul_ps(d, outputfm5); 
			rs[j+4] += sum8(rs1);

			rs2 = _mm256_mul_ps(d, outputfm6); 
			rs[j+5] += sum8(rs2);

			rs3 = _mm256_mul_ps(d, outputfm7); 
			rs[j+6] += sum8(rs3);

			rs4 = _mm256_mul_ps(d, outputfm8); 
			rs[j+7] += sum8(rs4);

			pmodel += 64;

		}

	}
	std::cout << "elapsed " << t.elapsed() << std::endl;
	*/


	/*
	t.restart();
	for(int i=0;i<43264;i+=8){
		d = _mm256_load_ps(&data[i]);

		for(int j=0;j<4096;j+=8){
			outputfm1 = _mm256_load_ps(&models[j][i]);
			outputfm2 = _mm256_load_ps(&models[j+1][i]);
			outputfm3 = _mm256_load_ps(&models[j+2][i]);
			outputfm4 = _mm256_load_ps(&models[j+3][i]);
			outputfm5 = _mm256_load_ps(&models[j+4][i]);
			outputfm6 = _mm256_load_ps(&models[j+5][i]);
			outputfm7 = _mm256_load_ps(&models[j+6][i]);
			outputfm8 = _mm256_load_ps(&models[j+7][i]);

			rs1 = _mm256_mul_ps(d, outputfm1); 
			rs[j] += sum8(rs1);

			rs2 = _mm256_mul_ps(d, outputfm2); 
			rs[j+1] += sum8(rs2);

			rs3 = _mm256_mul_ps(d, outputfm3); 
			rs[j+2] += sum8(rs3);

			rs4 = _mm256_mul_ps(d, outputfm4); 
			rs[j+3] += sum8(rs4);

			rs1 = _mm256_mul_ps(d, outputfm5); 
			rs[j+4] += sum8(rs1);

			rs2 = _mm256_mul_ps(d, outputfm6); 
			rs[j+5] += sum8(rs2);

			rs3 = _mm256_mul_ps(d, outputfm7); 
			rs[j+6] += sum8(rs3);

			rs4 = _mm256_mul_ps(d, outputfm8); 
			rs[j+7] += sum8(rs4);

		}

	}
	std::cout << "elapsed " << t.elapsed() << std::endl;
	*/

	for(int j=0;j<4096;j++){
		sum += rs[j];
	}
	std::cout << sum << std::endl;

	return 0;
}

/*
template<int W, int X, int Y, int Z>
void clear(float * _toclear){
	for(int i=0;i<W*X*Y*Z;i++){
		_toclear[i] = 0;
	}
}
*/


int main_mnist(int argc, char** argv){

	MNISTCorpus corpus("input/train-labels-idx1-ubyte", "input/train-images-idx3-ubyte");
	MNISTCorpus corpus_test("input/t10k-labels-idx1-ubyte", "input/t10k-images-idx3-ubyte");

	Timer t;
	double elapsed;
	long long flop;

	Timer t2;
	
	Cube<32,32,8,1> layer1;			 	 // pad 2
	Cube<32,32,8,1> layer2;			   // pad 2
	Cube<16,16,8,1> layer2_pool;	 // pad 2
	Cube<16,16,32,1> layer3;			 // pad 2
	Cube<7,7,32,1> layer3_pool; // pad 1
	Cube<1,1,256,1> layer4;		   // pad 1
	Cube<1,1,16,1> layer5;      // pad 1

	Cube<5,5,8,8> kernel12;			 	 // pad 1
	Cube<3,3,8,32> kernel23;			   // pad 0
	Cube<7,7,32,256> kernel34;	 // pad 2
	Cube<1,1,256,16> kernel45;			 // pad 2

	//layer1.init();

	kernel12.init();
	kernel23.init();
	kernel34.init();
	kernel45.init();

	float * pdata = &layer1.data[0][0][0][0];
	for(int i=0;i<32*32*8;i++){
		*pdata = 0;
		pdata ++;
	}

	for(int iepoch=0;iepoch<100;iepoch++){

		int ncorr = 0;

		//for(int i_img=0;i_img<corpus.n_image;i_img++){

		for(int i_img=0;i_img<8;i_img++){
			for(int i=0;i<16;i++){
				layer5.groundtruth[i] = 0;
			}
			layer5.groundtruth[corpus.images[i_img]->label] = 1;

			pdata = &layer1.data[0][0][0][0];
			for(int i=0;i<32*32;i++){
				*pdata = corpus.images[i_img]->_buf[i];
				pdata ++;
			}

			t2.restart();
			t.restart();
			flop = conv_parititionZ_vanilla<32, 5, 32, 8, 8>(
				layer1, kernel12, layer2);
			elapsed = t.elapsed();
			//std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

			t.restart();
			flop = maxpool2x2_vanilla_vanilla<32, -1, 16, 8, 8, 1, 1>(
				layer2, layer2_pool);
			elapsed = t.elapsed();
			//std::cout << "[maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

			t.restart();
			flop = conv_parititionZ_vanilla<16, 3, 16, 8, 32>(
				layer2_pool, kernel23, layer3);
			elapsed = t.elapsed();
			//std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

			t.restart();
			flop = maxpool2x2_vanilla_vanilla<16, -1, 7, 32, 32, 0, 1>(
				layer3, layer3_pool);
			elapsed = t.elapsed();
			//std::cout << "[maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

			t.restart();
			flop = fullconn_parititionZ_vanilla<7, 7, 1, 32, 256>(
				layer3_pool, kernel34, layer4);
			elapsed = t.elapsed();
			//std::cout << "[fullconn_parititionZ_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

			t.restart();
			flop = softmax_parititionZ_vanilla<1, 1, 1, 256, 16>(
				layer4, kernel45, layer5);
			elapsed = t.elapsed();
			//std::cout << "[softmax_parititionZ_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

			int gt = corpus.images[i_img]->label;
			int imax;
			double ifloat = -1;
			for(int dig=0;dig<16;dig++){
				double out = layer5.data[0][0][0][dig];
				if(out > ifloat){
					imax = dig;
					ifloat = out;
				}
			}
			ncorr += (gt == imax);

			//std::cout << std::endl;
			std::cout << i_img << "  " << 1.0*ncorr/(i_img+1) << "    " << (gt == imax) << "   gt=" << gt  << "   ans=" << imax << "(" << ifloat << ")" << std::endl;
			//std::cout << std::endl;

			/*
			std::cout << std::endl;
			for(int i=0;i<16;i++){
				std::cout << i << " -> " << layer5.data[0][0][0][i] << std::endl;
			}
			std::cout << std::endl;
			*/

			t.restart();
			flop = back_softmax_parititionZ_vanilla<1, 1, 1, 256, 16>(
				layer4, kernel45, layer5);
			elapsed = t.elapsed();
			//std::cout << "[back_softmax_parititionZ_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

			t.restart();
			flop = back_fullconn_parititionZ_vanilla<7, 7, 1, 32, 256>(
				layer3_pool, kernel34, layer4);
			elapsed = t.elapsed();
			//std::cout << "[back_fullconn_parititionZ_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

			t.restart();
			flop = back_maxpool2x2_vanilla_vanilla<16, -1, 7, 32, 32, 0, 1>(
				layer3, layer3_pool);
			elapsed = t.elapsed();
			//std::cout << "[back_maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


			t.restart();
			flop = back_conv_parititionZ_vanilla<16, 3, 16, 8, 32>(
				layer2_pool, kernel23, layer3);
			elapsed = t.elapsed();
			//std::cout << "[back_conv_parititionZ_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

			t.restart();
			flop = back_maxpool2x2_vanilla_vanilla<32, -1, 16, 8, 8, 1, 1>(
				layer2, layer2_pool);
			elapsed = t.elapsed();
			//std::cout << "[back_maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

			t.restart();
			flop = back_conv_parititionZ_vanilla<32, 5, 32, 8, 8>(
				layer1, kernel12, layer2);
			elapsed = t.elapsed();
			//std::cout << "[back_conv_parititionZ_vanilla();] elapsed " << elapsed
			//						 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;
			
		}


		/*
		t.restart();
		flop = maxpool2x2_vanilla_vanilla<55, -1, 31, 96, 96, 2, 0>(
			layer2, layer2_pool);
		elapsed = t.elapsed();
		std::cout << "[maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = conv_parititionZ_vanilla<31, 5, 31, 96, 256>(
			layer2_pool, kernel23, layer3);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = maxpool2x2_vanilla_vanilla<31, -1, 15, 256, 256, 2, 1>(
			layer3, layer3_pool);
		elapsed = t.elapsed();
		std::cout << "[maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


		t.restart();
		flop = conv_parititionZ_vanilla<15, 3, 15, 256, 384>(
			layer3_pool, kernel34, layer4);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = conv_parititionZ_vanilla<15, 3, 15, 384, 384>(
			layer4, kernel45, layer5);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = conv_parititionZ_vanilla<15, 3, 15, 384, 256>(
			layer5, kernel56, layer6);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = fullconn_parititionZ_vanilla<15, 13, 1, 256, 4096>(
			layer6, kernel67, layer7);
		elapsed = t.elapsed();
		std::cout << "[fullconn_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


		t.restart();
		flop = fullconn_parititionZ_vanilla<1, 1, 1, 4096, 4096>(
			layer7, kernel78, layer8);
		elapsed = t.elapsed();
		std::cout << "[fullconn_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


		t.restart();
		flop = softmax_parititionZ_vanilla<1, 1, 1, 4096, 1000>(
			layer8, kernel89, layer9);
		elapsed = t.elapsed();
		std::cout << "[softmax_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		////// BACK

		t.restart();
		flop = back_softmax_parititionZ_vanilla<1, 1, 1, 4096, 1000>(
			layer8, kernel89, layer9);
		elapsed = t.elapsed();
		std::cout << "[back_softmax_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_fullconn_parititionZ_vanilla<1, 1, 1, 4096, 4096>(
			layer7, kernel78, layer8);
		elapsed = t.elapsed();
		std::cout << "[back_fullconn_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_fullconn_parititionZ_vanilla<15, 13, 1, 256, 4096>(
			layer6, kernel67, layer7);
		elapsed = t.elapsed();
		std::cout << "[back_fullconn_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_conv_parititionZ_vanilla<15, 3, 15, 384, 256>(
			layer5, kernel56, layer6);
		elapsed = t.elapsed();
		std::cout << "[back_conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_conv_parititionZ_vanilla<15, 3, 15, 384, 384>(
			layer4, kernel45, layer5);
		elapsed = t.elapsed();
		std::cout << "[back_conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_conv_parititionZ_vanilla<15, 3, 15, 256, 384>(
			layer3_pool, kernel34, layer4);
		elapsed = t.elapsed();
		std::cout << "[back_conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


		t.restart();
		flop = back_maxpool2x2_vanilla_vanilla<31, -1, 15, 256, 256, 2, 1>(
			layer3, layer3_pool);
		elapsed = t.elapsed();
		std::cout << "[back_maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_conv_parititionZ_vanilla<31, 5, 31, 96, 256>(
			layer2_pool, kernel23, layer3);
		elapsed = t.elapsed();
		std::cout << "[back_conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_maxpool2x2_vanilla_vanilla<55, -1, 31, 96, 96, 2, 0>(
			layer2, layer2_pool);
		elapsed = t.elapsed();
		std::cout << "[back_maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_strid_conv_vanilla_vanilla<227, 11, 55, 3, 96, 4, 5>(
			layer1, kernel12, layer2);
		elapsed = t.elapsed();
		std::cout << "[back_strid_conv_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		std::cout << std::endl;
		std::cout << "TOTAL ELAPSED " << t2.elapsed() << std::endl;
		std::cout << std::endl;

		*/

		/*
		float sumprob = 0.0;
		for(int i=0;i<1000;i++){
			sumprob += layer9.data[0][0][0][i];
		}
		std::cout << "++++ " << sumprob << std::endl;
		std::cout << "|||||||| " << 13 << "\t" << layer9.data[0][0][0][13] << std::endl;
		std::cout << "|||||||| " << 15 << "\t" << layer9.data[0][0][0][15] << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;
		*/
	}

	return 0;

}



int main(int argc, char** argv){

	Timer t;
	double elapsed;
	long long flop;

	Timer t2;
	
	Cube<227,227,3,1> layer1;			 // pad 1
	Cube<55,55,96,1> layer2;			 // pad 0
	Cube<31,31,96,1> layer2_pool;	 // pad 2
	Cube<31,31,256,1> layer3;			 // pad 2
	Cube<15,15,256,1> layer3_pool; // pad 1
	Cube<15,15,384,1> layer4;		   // pad 1
	Cube<15,15,384,1> layer5;      // pad 1
	Cube<15,15,256,1> layer6;      // pad 1

	Cube<1,1,4096,1> layer7;
	Cube<1,1,4096,1> layer8;
	Cube<1,1,1000,1> layer9;

	Cube<11,11,3,96> kernel12;
	Cube<5,5,96,256> kernel23;
	Cube<3,3,256,384> kernel34;
	Cube<3,3,384,384> kernel45;
	Cube<3,3,384,256> kernel56;
	Cube<13,13,256,4096> kernel67;
	Cube<1,1,4096,4096> kernel78;
	Cube<1,1,4096,1000> kernel89;

	layer1.init();

	kernel12.init();
	kernel23.init();
	kernel34.init();
	kernel45.init();
	kernel56.init();

	kernel67.init();
	kernel78.init();
	kernel89.init();

	for(int i=0;i<1000;i++){
		layer9.groundtruth[i] = 0;
	}
	layer9.groundtruth[15] = 1;


		t2.restart();
		t.restart();
		flop = strid_conv_vanilla_vanilla<227, 11, 55, 3, 96, 4, 5>(
			layer1, kernel12, layer2);
		elapsed = t.elapsed();
		std::cout << "[strid_conv_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;
		
		t.restart();
		flop = maxpool2x2_vanilla_vanilla<55, -1, 31, 96, 96, 2, 0>(
			layer2, layer2_pool);
		elapsed = t.elapsed();
		std::cout << "[maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = conv_parititionZ_vanilla<31, 5, 31, 96, 256>(
			layer2_pool, kernel23, layer3);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = maxpool2x2_vanilla_vanilla<31, -1, 15, 256, 256, 2, 1>(
			layer3, layer3_pool);
		elapsed = t.elapsed();
		std::cout << "[maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = conv_parititionZ_vanilla<15, 3, 15, 256, 384>(
			layer3_pool, kernel34, layer4);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = conv_parititionZ_vanilla<15, 3, 15, 384, 384>(
			layer4, kernel45, layer5);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = conv_parititionZ_vanilla<15, 3, 15, 384, 256>(
			layer5, kernel56, layer6);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = fullconn_parititionZ_vanilla<15, 13, 1, 256, 4096>(
			layer6, kernel67, layer7);
		elapsed = t.elapsed();
		std::cout << "[fullconn_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


		t.restart();
		flop = fullconn_parititionZ_vanilla<1, 1, 1, 4096, 4096>(
			layer7, kernel78, layer8);
		elapsed = t.elapsed();
		std::cout << "[fullconn_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


		t.restart();
		flop = softmax_parititionZ_vanilla<1, 1, 1, 4096, 1000>(
			layer8, kernel89, layer9);
		elapsed = t.elapsed();
		std::cout << "[softmax_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		std::cout << "TOTAL ELAPSED " << t2.elapsed() << std::endl;


	for(int iepoch=0;iepoch<100;iepoch++){

		t2.restart();
		t.restart();
		flop = strid_conv_vanilla_vanilla<227, 11, 55, 3, 96, 4, 5>(
			layer1, kernel12, layer2);
		elapsed = t.elapsed();
		std::cout << "[strid_conv_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;
		
		t.restart();
		flop = maxpool2x2_vanilla_vanilla<55, -1, 31, 96, 96, 2, 0>(
			layer2, layer2_pool);
		elapsed = t.elapsed();
		std::cout << "[maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = conv_parititionZ_vanilla<31, 5, 31, 96, 256>(
			layer2_pool, kernel23, layer3);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = maxpool2x2_vanilla_vanilla<31, -1, 15, 256, 256, 2, 1>(
			layer3, layer3_pool);
		elapsed = t.elapsed();
		std::cout << "[maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


		t.restart();
		flop = conv_parititionZ_vanilla<15, 3, 15, 256, 384>(
			layer3_pool, kernel34, layer4);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = conv_parititionZ_vanilla<15, 3, 15, 384, 384>(
			layer4, kernel45, layer5);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = conv_parititionZ_vanilla<15, 3, 15, 384, 256>(
			layer5, kernel56, layer6);
		elapsed = t.elapsed();
		std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = fullconn_parititionZ_vanilla<15, 13, 1, 256, 4096>(
			layer6, kernel67, layer7);
		elapsed = t.elapsed();
		std::cout << "[fullconn_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


		t.restart();
		flop = fullconn_parititionZ_vanilla<1, 1, 1, 4096, 4096>(
			layer7, kernel78, layer8);
		elapsed = t.elapsed();
		std::cout << "[fullconn_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


		t.restart();
		flop = softmax_parititionZ_vanilla<1, 1, 1, 4096, 1000>(
			layer8, kernel89, layer9);
		elapsed = t.elapsed();
		std::cout << "[softmax_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		////// BACK

		t.restart();
		flop = back_softmax_parititionZ_vanilla<1, 1, 1, 4096, 1000>(
			layer8, kernel89, layer9);
		elapsed = t.elapsed();
		std::cout << "[back_softmax_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_fullconn_parititionZ_vanilla<1, 1, 1, 4096, 4096>(
			layer7, kernel78, layer8);
		elapsed = t.elapsed();
		std::cout << "[back_fullconn_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_fullconn_parititionZ_vanilla<15, 13, 1, 256, 4096>(
			layer6, kernel67, layer7);
		elapsed = t.elapsed();
		std::cout << "[back_fullconn_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_conv_parititionZ_vanilla<15, 3, 15, 384, 256>(
			layer5, kernel56, layer6);
		elapsed = t.elapsed();
		std::cout << "[back_conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_conv_parititionZ_vanilla<15, 3, 15, 384, 384>(
			layer4, kernel45, layer5);
		elapsed = t.elapsed();
		std::cout << "[back_conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_conv_parititionZ_vanilla<15, 3, 15, 256, 384>(
			layer3_pool, kernel34, layer4);
		elapsed = t.elapsed();
		std::cout << "[back_conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;


		t.restart();
		flop = back_maxpool2x2_vanilla_vanilla<31, -1, 15, 256, 256, 2, 1>(
			layer3, layer3_pool);
		elapsed = t.elapsed();
		std::cout << "[back_maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_conv_parititionZ_vanilla<31, 5, 31, 96, 256>(
			layer2_pool, kernel23, layer3);
		elapsed = t.elapsed();
		std::cout << "[back_conv_parititionZ_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_maxpool2x2_vanilla_vanilla<55, -1, 31, 96, 96, 2, 0>(
			layer2, layer2_pool);
		elapsed = t.elapsed();
		std::cout << "[back_maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		t.restart();
		flop = back_strid_conv_vanilla_vanilla<227, 11, 55, 3, 96, 4, 5>(
			layer1, kernel12, layer2);
		elapsed = t.elapsed();
		std::cout << "[back_strid_conv_vanilla_vanilla();] elapsed " << elapsed
								 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

		std::cout << std::endl;
		std::cout << "TOTAL ELAPSED " << t2.elapsed() << std::endl;
		std::cout << std::endl;

		/*
		float sumprob = 0.0;
		for(int i=0;i<1000;i++){
			sumprob += layer9.data[0][0][0][i];
		}
		std::cout << "++++ " << sumprob << std::endl;
		std::cout << "|||||||| " << 13 << "\t" << layer9.data[0][0][0][13] << std::endl;
		std::cout << "|||||||| " << 15 << "\t" << layer9.data[0][0][0][15] << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;
		*/
	}

	return 0;

}


int main_old(int argc, char** argv){


	// 55->27 pooling
	const int INPUTSIZE = 55;
	const int OUTPUTSIZE = 29;

	int flop;
	double elapsed;
	float sum;

	Cube<INPUTSIZE,INPUTSIZE,96,1> input;
	for(int x=0;x<input.X;x++){
		for(int y=0;y<input.Y;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = 0;
			}
		}
	}
	for(int x=0;x<INPUTSIZE;x++){
		for(int y=0;y<INPUTSIZE;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = drand48();
			}
		}
	}

	Cube<OUTPUTSIZE, OUTPUTSIZE, 96, 1> output;

	Timer t;
	t.restart();
	input.partition_Z();
	std::cout << "[input.partition_Z();] elapsed " << t.elapsed() << std::endl;

	t.restart();
	flop = maxpool2x2_vanilla_vanilla<INPUTSIZE, -1, OUTPUTSIZE, 96, 96, 1, 0>(
		input, output);
	elapsed = t.elapsed();
	std::cout << "[maxpool2x2_vanilla_vanilla();] elapsed " << elapsed
							 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;
	sum = 0.0;
	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<96;z++){
				sum += output.data[0][x][y][z];
			}
		}
	}
	std::cout << sum << std::endl;
	std::cout << "0 0 0 0 -> " << output.data[0][1][1][0] << std::endl;


	// 3->112 layer
	/*
	const int INPUTSIZE = 225;
	const int KERNELSIZE = 11;
	const int OUTPUTSIZE = 55;

	int flop;
	double elapsed;
	float sum;

	Cube<INPUTSIZE,INPUTSIZE,3,1> input;
	for(int x=0;x<input.X;x++){
		for(int y=0;y<input.Y;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = 0;
			}
		}
	}
	for(int x=KERNELSIZE/2;x<INPUTSIZE-KERNELSIZE/2;x++){
		for(int y=KERNELSIZE/2;y<INPUTSIZE-KERNELSIZE/2;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = drand48();
			}
		}
	}

	Cube<KERNELSIZE, KERNELSIZE, 3, 112> kernel;

	for(int w=0;w<kernel.W;w++){
		for(int x=0;x<kernel.X;x++){
			for(int y=0;y<kernel.Y;y++){
				for(int z=0;z<kernel.Z;z++){
					kernel.data[w][x][y][z] = drand48();
				}
			}
		}
	}

	Cube<OUTPUTSIZE, OUTPUTSIZE, 112, 1> output;

	Timer t;
	t.restart();
	input.partition_Z();
	std::cout << "[input.partition_Z();] elapsed " << t.elapsed() << std::endl;

	t.restart();
	flop = strid_conv_vanilla_vanilla<INPUTSIZE, KERNELSIZE, OUTPUTSIZE, 3, 112, 4, 5>(
		input, kernel, output);
	elapsed = t.elapsed();
	std::cout << "[strid_conv_vanilla_vanilla();] elapsed " << elapsed
							 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;
	sum = 0.0;
	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<112;z++){
				sum += output.data[0][x][y][z];
			}
		}
	}
	std::cout << sum << std::endl;
	std::cout << "0 0 0 0 -> " << output.data[0][1][1][0] << std::endl;
	*/

	// 112->256
	/*
	const int INPUTSIZE = 27;
	const int KERNELSIZE = 5;
	const int OUTPUTSIZE = 27;

	int flop;
	double elapsed;
	float sum;

	Cube<INPUTSIZE,INPUTSIZE,112,1> input;
	for(int x=0;x<input.X;x++){
		for(int y=0;y<input.Y;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = 0;
			}
		}
	}
	for(int x=KERNELSIZE/2;x<INPUTSIZE-KERNELSIZE/2;x++){
		for(int y=KERNELSIZE/2;y<INPUTSIZE-KERNELSIZE/2;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = drand48();
			}
		}
	}

	Cube<KERNELSIZE, KERNELSIZE, 112, 256> kernel;

	for(int w=0;w<kernel.W;w++){
		for(int x=0;x<kernel.X;x++){
			for(int y=0;y<kernel.Y;y++){
				for(int z=0;z<kernel.Z;z++){
					kernel.data[w][x][y][z] = drand48();
				}
			}
		}
	}

	Cube<OUTPUTSIZE, OUTPUTSIZE, 256, 1> output;

	Timer t;
	t.restart();
	input.partition_Z();
	std::cout << "[input.partition_Z();] elapsed " << t.elapsed() << std::endl;

	t.restart();
	flop = conv_vanilla_vanilla<INPUTSIZE, KERNELSIZE, OUTPUTSIZE, 112, 256>(
		input, kernel, output);
	elapsed = t.elapsed();
	std::cout << "[conv_vanilla_vanilla();] elapsed " << elapsed
							 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;
	sum = 0.0;
	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<256;z++){
				sum += output.data[0][x][y][z];
			}
		}
	}
	std::cout << sum << std::endl;
	std::cout << "0 0 0 0 -> " << output.data[0][1][1][0] << std::endl;

	t.restart();
	flop = conv_parititionZ_vanilla<INPUTSIZE, KERNELSIZE, OUTPUTSIZE, 112, 256>(
		input, kernel, output);
	elapsed = t.elapsed();
	std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
							 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

	sum = 0.0;
	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<256;z++){
				sum += output.data[0][x][y][z];
			}
		}
	}
	std::cout << sum << std::endl;
	std::cout << "0 0 0 0 -> " << output.data[0][1][1][0] << std::endl;
	*/

	// 256->200 layer
	/*
	const int INPUTSIZE = 15;
	const int KERNELSIZE = 3;
	const int OUTPUTSIZE = 15;

	int flop;
	double elapsed;
	float sum;

	Cube<INPUTSIZE,INPUTSIZE,256,1> input;
	for(int x=0;x<input.X;x++){
		for(int y=0;y<input.Y;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = 0;
			}
		}
	}
	for(int x=KERNELSIZE/2;x<INPUTSIZE-KERNELSIZE/2;x++){
		for(int y=KERNELSIZE/2;y<INPUTSIZE-KERNELSIZE/2;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = drand48();
			}
		}
	}

	Cube<KERNELSIZE, KERNELSIZE, 256, 200> kernel;

	for(int w=0;w<kernel.W;w++){
		for(int x=0;x<kernel.X;x++){
			for(int y=0;y<kernel.Y;y++){
				for(int z=0;z<kernel.Z;z++){
					kernel.data[w][x][y][z] = drand48();
				}
			}
		}
	}

	Cube<OUTPUTSIZE, OUTPUTSIZE, 200, 1> output;

	Timer t;
	t.restart();
	input.partition_Z();
	std::cout << "[input.partition_Z();] elapsed " << t.elapsed() << std::endl;

	t.restart();
	flop = conv_vanilla_vanilla<INPUTSIZE, KERNELSIZE, OUTPUTSIZE, 256, 200>(
		input, kernel, output);
	elapsed = t.elapsed();
	std::cout << "[conv_vanilla_vanilla();] elapsed " << elapsed
							 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;
	sum = 0.0;
	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<200;z++){
				sum += output.data[0][x][y][z];
			}
		}
	}
	std::cout << sum << std::endl;
	std::cout << "0 0 0 0 -> " << output.data[0][1][1][0] << std::endl;

	t.restart();
	flop = conv_parititionZ_vanilla<INPUTSIZE, KERNELSIZE, OUTPUTSIZE, 256, 200>(
		input, kernel, output);
	elapsed = t.elapsed();
	std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
							 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

	sum = 0.0;
	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<200;z++){
				sum += output.data[0][x][y][z];
			}
		}
	}
	std::cout << sum << std::endl;
	std::cout << "0 0 0 0 -> " << output.data[0][1][1][0] << std::endl;
	*/



	// 192->128 layer

	/*
	const int INPUTSIZE = 15;
	const int KERNELSIZE = 3;
	const int OUTPUTSIZE = 15;

	int flop;
	double elapsed;
	float sum;

	Cube<INPUTSIZE,INPUTSIZE,200,1> input;
	for(int x=0;x<input.X;x++){
		for(int y=0;y<input.Y;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = 0;
			}
		}
	}
	for(int x=KERNELSIZE/2;x<INPUTSIZE-KERNELSIZE/2;x++){
		for(int y=KERNELSIZE/2;y<INPUTSIZE-KERNELSIZE/2;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = drand48();
			}
		}
	}

	Cube<KERNELSIZE, KERNELSIZE, 200, 128> kernel;

	for(int w=0;w<kernel.W;w++){
		for(int x=0;x<kernel.X;x++){
			for(int y=0;y<kernel.Y;y++){
				for(int z=0;z<kernel.Z;z++){
					kernel.data[w][x][y][z] = drand48();
				}
			}
		}
	}

	Cube<OUTPUTSIZE, OUTPUTSIZE, 128, 1> output;

	Timer t;
	t.restart();
	input.partition_Z();
	std::cout << "[input.partition_Z();] elapsed " << t.elapsed() << std::endl;

	t.restart();
	flop = conv_vanilla_vanilla<INPUTSIZE, KERNELSIZE, OUTPUTSIZE, 200, 128>(
		input, kernel, output);
	elapsed = t.elapsed();
	std::cout << "[conv_vanilla_vanilla();] elapsed " << elapsed
							 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;
	sum = 0.0;
	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<200;z++){
				sum += output.data[0][x][y][z];
			}
		}
	}
	std::cout << sum << std::endl;
	std::cout << "0 0 0 0 -> " << output.data[0][1][1][0] << std::endl;

	t.restart();
	flop = conv_parititionZ_vanilla<INPUTSIZE, KERNELSIZE, OUTPUTSIZE, 200, 128>(
		input, kernel, output);
	elapsed = t.elapsed();
	std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
							 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

	sum = 0.0;
	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<200;z++){
				sum += output.data[0][x][y][z];
			}
		}
	}
	std::cout << sum << std::endl;
	std::cout << "0 0 0 0 -> " << output.data[0][1][1][0] << std::endl;
	*/



	// 192->192 layer
	/*
	const int INPUTSIZE = 15;
	const int KERNELSIZE = 3;
	const int OUTPUTSIZE = 15;

	int flop;
	double elapsed;
	float sum;

	Cube<INPUTSIZE,INPUTSIZE,200,1> input;
	for(int x=0;x<input.X;x++){
		for(int y=0;y<input.Y;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = 0;
			}
		}
	}
	for(int x=KERNELSIZE/2;x<INPUTSIZE-KERNELSIZE/2;x++){
		for(int y=KERNELSIZE/2;y<INPUTSIZE-KERNELSIZE/2;y++){
			for(int z=0;z<input.Z;z++){
				input.data[0][x][y][z] = drand48();
			}
		}
	}

	Cube<KERNELSIZE, KERNELSIZE, 200, 200> kernel;

	for(int w=0;w<kernel.W;w++){
		for(int x=0;x<kernel.X;x++){
			for(int y=0;y<kernel.Y;y++){
				for(int z=0;z<kernel.Z;z++){
					kernel.data[w][x][y][z] = drand48();
				}
			}
		}
	}

	Cube<OUTPUTSIZE, OUTPUTSIZE, 200, 1> output;

	Timer t;
	t.restart();
	input.partition_Z();
	std::cout << "[input.partition_Z();] elapsed " << t.elapsed() << std::endl;

	t.restart();
	flop = conv_vanilla_vanilla<INPUTSIZE, KERNELSIZE, OUTPUTSIZE, 200, 200>(
		input, kernel, output);
	elapsed = t.elapsed();
	std::cout << "[conv_vanilla_vanilla();] elapsed " << elapsed
							 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;
	sum = 0.0;
	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<200;z++){
				sum += output.data[0][x][y][z];
			}
		}
	}
	std::cout << sum << std::endl;
	std::cout << "0 0 0 0 -> " << output.data[0][1][1][0] << std::endl;

	t.restart();
	flop = conv_parititionZ_vanilla<INPUTSIZE, KERNELSIZE, OUTPUTSIZE, 200, 200>(
		input, kernel, output);
	elapsed = t.elapsed();
	std::cout << "[conv_parititionZ_vanilla();] elapsed " << elapsed
							 << " flop " << flop << " GFlops " << 1.0*flop/elapsed/1024/1024/1024 << std::endl;

	sum = 0.0;
	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<200;z++){
				sum += output.data[0][x][y][z];
			}
		}
	}
	std::cout << sum << std::endl;
	std::cout << "0 0 0 0 -> " << output.data[0][1][1][0] << std::endl;
	*/



	/*
	Cube_xy8 input;
	input.X = 15;
	input.Y = 15;
	input.init();

	std::cout << "1" << std::endl;

	Cube_xy1536 kernel;
	kernel.X = 3;
	kernel.Y = 3;
	kernel.init();

	std::cout << "2" << std::endl;

	Cube_xy192 output;
	output.X = 15;
	output.Y = 15;
	output.init();

	std::cout << "3" << std::endl;

	for(int x=0;x<input.X;x++){
		for(int y=0;y<input.Y;y++){
			for(int z=0;z<8;z++){
				input.data[x][y][z] = rand();
			}
		}
	}

	std::cout << "++++" << std::endl;

	for(int x=0;x<kernel.X;x++){
		for(int y=0;y<kernel.Y;y++){
			for(int z=0;z<1536;z++){
				kernel.data[x][y][z] = rand();
			}
		}
	}


	std::cout << "~~~~" << std::endl;
	Timer t;

	t.restart();
	sse_conv(input, kernel, output, 1, input.X-1);
	//stupid_conv(input, kernel, output, 1, input.X-1);

	std::cout << "elapsed " << t.elapsed() << std::endl;

	t.restart();
	stupid_conv(input, kernel, output);
	std::cout << "elapsed " << t.elapsed() << std::endl;

	t.restart();
	stupid_conv(input, kernel, output);
	std::cout << "elapsed " << t.elapsed() << std::endl;

	t.restart();
	sse_conv(input, kernel, output);
	std::cout << "elapsed " << t.elapsed() << std::endl;

	t.restart();
	sse_conv(input, kernel, output);
	std::cout << "elapsed " << t.elapsed() << std::endl;

	float sum = 0.0;

	for(int x=0;x<output.X;x++){
		for(int y=0;y<output.Y;y++){
			for(int z=0;z<192;z++){
				sum += output.data[x][y][z];
			}
		}
	}
	std::cout << "    " << sum << std::endl;

	*/

	/*
	double s2 = 0.0;
	for(int i=0;i<input.X*input.X;i++){
		s2 += aaa[i];
	}
	std::cout << "~~~~~" << s2 << std::endl;
	*/

	return 0;

}




