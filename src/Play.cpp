
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

#include "timer.h"
#include <assert.h>

const int nrow = 400;
const int ncol = 400;

const int nrow_output = 391;
const int ncol_output = 391;

const int nrow_kernel = nrow-nrow_output+1;
const int ncol_kernel = ncol-ncol_output+1;

const int nfeaturemap=2;
float sums[1000];
const int size_of_stream = nrow_kernel*nrow_kernel*nrow_output*ncol_output;
float materialize[size_of_stream];

float * const buf_output = new float[nfeaturemap*nrow_output*ncol_output];
float * const buf_kernel = new float[nfeaturemap*nrow_kernel*ncol_kernel];

const int NRUN = 100;

float vsum(const float *a, int _n)
{
    float sum;
    int n = _n - _n%3;
    __m128 vsum = _mm_set1_ps(0.0f);
    assert((n & 3) == 0);
    assert(((uintptr_t)a & 15) == 0);
    for (int i = 0; i < n; i += 4)
    {
        __m128 v = _mm_load_ps(&a[i]);
        vsum = _mm_add_ps(vsum, v);
    }
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    _mm_store_ss(&sum, vsum);

    for(int i=n;i<_n;i++){
    	sum += a[i];
    }

    return sum;
}

/*
// Compute out[i] = a[i] + b[i] with SSE instructions.
void MultiSSE(float *out, const float *a, const float *b, int N)
{
	for(int i = 0; i < N; i += 8)
	{
		// Read 4 floats from a to a_i, and b to b_i.
		__m256 a_i = _mm256_loadu_ps(&a[i]);
		__m256 b_i = _mm256_loadu_ps(&b[i]);
 
		// Compute out_i = a_i + b_i.
		__m256 out_i = _mm256_mul_ps(a_i, b_i);
 
		// Write 4 floats from out_i to out.
		_mm256_store_ps(&out[i], out_i);
	}
}
*/


// Compute out[i] = a[i] + b[i] with SSE instructions.
void MultiSSE(float *out, const float *a, const float *b, int N)
{
	for(int i = 0; i < N; i += 4)
	{
		// Read 4 floats from a to a_i, and b to b_i.
		__m128 a_i = _mm_load_ps(&a[i]);
		__m128 b_i = _mm_load_ps(&b[i]);
 
		// Compute out_i = a_i + b_i.
		__m128 out_i = _mm_mul_ps(a_i, b_i);
 
		// Write 4 floats from out_i to out.
		_mm_store_ps(&out[i], out_i);
	}
}

// Compute out[i] = a[i] + b[i] with SSE instructions.
void AddSSE(float *out, const float *a, const float *b, int N)
{
	for(int i = 0; i < N; i += 4)
	{
		// Read 4 floats from a to a_i, and b to b_i.
		__m128 a_i = _mm_load_ps(&a[i]);
		__m128 b_i = _mm_load_ps(&b[i]);
 
		// Compute out_i = a_i + b_i.
		__m128 out_i = _mm_add_ps(a_i, b_i);
 
		// Write 4 floats from out_i to out.
		_mm_store_ps(&out[i], out_i);
	}
}

void run2(float ** output_collapsed, float ** kernel_collapsed, float ** img){

	int ct = 0;
	for(int r=0;r<nrow_output;r++){
		for(int c=0;c<ncol_output;c++){
			for(int ir=r;ir<r+nrow_kernel;ir++){
				for(int ic=c;ic<c+ncol_kernel;ic++){
					materialize[ct++] = img[ir][ic];
				}
			}
		}
	}

	for(int i_epoch=0;i_epoch<NRUN;i_epoch++){

		float sum[nrow_kernel*ncol_kernel];
		float sum1 = 0.0;
		float * output_flatten = &output_collapsed[0][0];
		for(int m=0;m<nfeaturemap;m++){
			int ct = m;
			const float * const buf_kernel2 = &buf_kernel[nrow_kernel*ncol_kernel*m];
			const float * mat = &materialize[0];
			for(int i=0;i<size_of_stream;i+=nrow_kernel*ncol_kernel){
				MultiSSE(sum, mat, buf_kernel2, nrow_kernel*ncol_kernel);
				output_flatten[ct] = vsum(sum, nrow_kernel*ncol_kernel);
				ct = ct + nfeaturemap;
				mat += nrow_kernel*ncol_kernel;
			}
		}

		/**
			//for(int j=0;j<nrow_kernel*nrow_kernel;j++){
			//	sum[j] = mat[j] * buf_kernel2[j];
			//}
		*/
	}
}



void run1(float ** output_collapsed, float ** kernel_collapsed, float ** img){
	
	for(int i_epoch=0;i_epoch<NRUN;i_epoch++){

		float * poutput = &output_collapsed[0][0];
		for(int r=0;r<nrow_output;r++){
			for(int c=0;c<ncol_output;c++){

				for(int i=0;i<nfeaturemap;i++){
					sums[i] = 0.0;
				}

				float * pweight = &kernel_collapsed[0][0];
				for(int ir=r;ir<r+nrow_kernel;ir++){
					float * pimg = &img[ir][0];

					for(int i=0;i<nfeaturemap;i++){
						sums[i] += pweight[i] * img[ir][c];
					}

					for(int i=0;i<nfeaturemap;i++){
						sums[i] += pweight[i+nfeaturemap] * pimg[c+1];
					}

					for(int i=0;i<nfeaturemap;i++){
						sums[i] += pweight[i+1*nfeaturemap] * pimg[c+2];
					}

					for(int i=0;i<nfeaturemap;i++){
						sums[i] += pweight[i+2*nfeaturemap] * pimg[c+3];
					}

					for(int i=0;i<nfeaturemap;i++){
						sums[i] += pweight[i+3*nfeaturemap] * pimg[c+4];
					}

					/*
					for(int i=0;i<nfeaturemap;i++){
						sums[i] += pweight[i] * img[ir][c];
					}

					for(int i=0;i<nfeaturemap;i++){
						sums[i] += pweight[i+nfeaturemap] * pimg[c+1];
					}

					for(int i=0;i<nfeaturemap;i++){
						sums[i] += pweight[i+1*nfeaturemap] * pimg[c+2];
					}

					for(int i=0;i<nfeaturemap;i++){
						sums[i] += pweight[i+2*nfeaturemap] * pimg[c+3];
					}

					for(int i=0;i<nfeaturemap;i++){
						sums[i] += pweight[i+3*nfeaturemap] * pimg[c+4];
					}
					*/


					pweight += nfeaturemap*ncol_kernel;

				}

				for(int i=0;i<nfeaturemap;i++){
					poutput[i] = sums[i];
				}

				poutput += nfeaturemap;
			}
		}
	}
}


void aaaa(const int c, float * const a3, float * const a4, const float const * fck){
	__m256 a_fck = _mm256_load_ps(fck);

	for(int i=0;i<c;i+=16){
		
		__m256 b_i = _mm256_load_ps(&a4[i]);
		__m256 out_i = _mm256_mul_ps(b_i, a_fck);

		__m256 c_i = _mm256_load_ps(&a4[i+8]);
		__m256 out_i2 = _mm256_mul_ps(c_i, a_fck);

		_mm256_store_ps(&a3[i], out_i);
		_mm256_store_ps(&a3[i+8], out_i2);

	}
}


void trya(){
  Timer tt;
  const int c = 10000000;
  const int c2= 2*c;
  double summm;

  //float * const a1 = new float[c];                                                                     
  //float * const a2 = new float[c];                                                                     
  //float * const a3 = new float[c];                                                                     

  float * const a1 = (float *)_mm_malloc(2*c*sizeof(float), 32);
  float * const a3 = (float *)_mm_malloc(8*c*sizeof(float), 32);

  float * const a4 = (float *)_mm_malloc(c*sizeof(float), 32);
  float * const a5 = (float *)_mm_malloc(c*sizeof(float), 32);;

  //register float always_in = 5;                                                                        
  //register float always_in2 = 6;                                                                       
  //register float always_in3 = 7;                                                                       

  float * fck = (float *)_mm_malloc(8*sizeof(float), 32);
  fck[0] = 1;
  fck[1] = 1;
  fck[2] = 1;
  fck[3] = 1;
  fck[4] = 1;
  fck[5] = 1;
  fck[6] = 1;
  fck[7] = 1;

  float * fck2 = (float *)_mm_malloc(8*sizeof(float), 32);
  float * fck3 = (float *)_mm_malloc(8*sizeof(float), 32);

  float * fck4 = (float *)_mm_malloc(8*sizeof(float), 32);
  float * fck5 = (float *)_mm_malloc(8*sizeof(float), 32);
  float * fck6 = (float *)_mm_malloc(8*sizeof(float), 32);
  float * fck7 = (float *)_mm_malloc(8*sizeof(float), 32);
  float * fck8 = (float *)_mm_malloc(8*sizeof(float), 32);

  float * fck9 = (float *)_mm_malloc(8*sizeof(float), 32);
  float * fck10 = (float *)_mm_malloc(8*sizeof(float), 32);
  float * fck11 = (float *)_mm_malloc(8*sizeof(float), 32);
  float * fck12 = (float *)_mm_malloc(8*sizeof(float), 32);

  fck2[0] = 1;
  fck2[1] = 1;
  fck2[2] = 1;
  fck2[3] = 1;
  fck2[4] = 1;
  fck2[5] = 1;
  fck2[6] = 1;
  fck2[7] = 1;

  fck3[0] = 1; fck3[1] = 1; fck3[2] = 1; fck3[3] = 1;
  fck3[4] = 1; fck3[5] = 1; fck3[6] = 1; fck3[7] = 1;

  fck4[0] = rand(); fck4[1] = rand(); fck4[2] = rand(); fck4[3] = rand();
  fck4[4] = rand(); fck4[5] = rand(); fck4[6] = rand(); fck4[7] = rand();

  fck5[0] = rand(); fck5[1] = rand(); fck5[2] = rand(); fck5[3] = rand();
  fck5[4] = rand(); fck5[5] = rand(); fck5[6] = rand(); fck5[7] = rand();

  fck6[0] = rand(); fck6[1] = rand(); fck6[2] = rand(); fck6[3] = rand();
  fck6[4] = rand(); fck6[5] = rand(); fck6[6] = rand(); fck6[7] = rand();

  fck7[0] = rand(); fck7[1] = rand(); fck7[2] = rand(); fck7[3] = rand();
  fck7[4] = rand(); fck7[5] = rand(); fck7[6] = rand(); fck7[7] = rand();

  fck8[0] = rand(); fck8[1] = rand(); fck8[2] = rand(); fck8[3] = rand();
  fck8[4] = rand(); fck8[5] = rand(); fck8[6] = rand(); fck8[7] = rand();

  fck9[0] = rand(); fck9[1] = rand(); fck9[2] = rand(); fck9[3] = rand();
  fck9[4] = rand(); fck9[5] = rand(); fck9[6] = rand(); fck9[7] = rand();

  fck10[0] = rand(); fck10[1] = rand(); fck10[2] = rand(); fck10[3] = rand();
  fck10[4] = rand(); fck10[5] = rand(); fck10[6] = rand(); fck10[7] = rand();

  fck11[0] = rand(); fck11[1] = rand(); fck11[2] = rand(); fck11[3] = rand();
  fck11[4] = rand(); fck11[5] = rand(); fck11[6] = rand(); fck11[7] = rand();

  fck12[0] = rand(); fck12[1] = rand(); fck12[2] = rand(); fck12[3] = rand();
  fck12[4] = rand(); fck12[5] = rand(); fck12[6] = rand(); fck12[7] = rand();

  //std::cout << "~~~~~~~~~" << std::endl;                                                               

  register __m256 a_fck = _mm256_load_ps(fck);
  register __m256 b_fck = _mm256_load_ps(fck2);
  register __m256 c_fck = _mm256_load_ps(fck3);
  register __m256 d_fck = _mm256_load_ps(fck4);
  register __m256 e_fck = _mm256_load_ps(fck5);
  register __m256 f_fck = _mm256_load_ps(fck6);
  register __m256 g_fck = _mm256_load_ps(fck7);
  register __m256 h_fck = _mm256_load_ps(fck8);

  register __m256 i_fck = _mm256_load_ps(fck9);
  register __m256 j_fck = _mm256_load_ps(fck10);
  register __m256 k_fck = _mm256_load_ps(fck11);
  register __m256 l_fck = _mm256_load_ps(fck12);


  //std::cout << "+++++++++" << std::endl;                                                               

  for(int i=0;i<c2;i++){
    a1[i] = rand();
    //a2[i] = rand();                                                                                    
  }

  for(int i=0;i<c;i++){
    a4[i] = rand();
    a5[i] = rand();
    a3[i] = a1[i];
  }

  //std::cout << "#########" << std::endl;                                                               

  tt.restart();

  __m256 b_i, c_i, d_i, e_i;
  __m256 out_i, out_i2, out_i3, out_i4,
    out_i5, out_i6, out_i7, out_i8,
    out_i9, out_i10, out_i11, out_i12;


  const float * pLoad = a4;
  float * pStore = a3;

  b_i = _mm256_loadu_ps(pLoad);
  const int SKIP = 8;
  for(int i=0;i<c;i+=SKIP){

    out_i = _mm256_mul_ps(b_i, a_fck);

    out_i2 = _mm256_mul_ps(b_i, b_fck);
    out_i3 = _mm256_mul_ps(b_i, c_fck);
    out_i4 = _mm256_mul_ps(b_i, d_fck);
    out_i5 = _mm256_mul_ps(b_i, e_fck);
    out_i6 = _mm256_mul_ps(b_i, f_fck);
    out_i7 = _mm256_mul_ps(b_i, g_fck);
    out_i8 = _mm256_mul_ps(b_i, h_fck);

    out_i9 = _mm256_mul_ps(b_i, i_fck);
    out_i10 = _mm256_mul_ps(b_i, j_fck);
    out_i11 = _mm256_mul_ps(b_i, k_fck);
    out_i12 = _mm256_mul_ps(b_i, l_fck);

    out_i = _mm256_add_ps(out_i, out_i2);
    out_i3 = _mm256_add_ps(out_i3, out_i4);
    out_i5 = _mm256_add_ps(out_i5, out_i6);
    out_i7 = _mm256_add_ps(out_i7, out_i8);
    out_i9 = _mm256_add_ps(out_i9, out_i10);
    out_i11 = _mm256_add_ps(out_i11, out_i12);

    out_i2 = _mm256_add_ps(out_i, out_i3);
    out_i5 = _mm256_add_ps(out_i5, out_i7);
    out_i9 = _mm256_add_ps(out_i9, out_i11);

    out_i = _mm256_add_ps(out_i2, out_i5);
    out_i = _mm256_add_ps(out_i, out_i9);

	//_mm256_storeu_ps(pStore, out_i);
    _mm256_stream_ps(pStore, out_i);
    //_mm256_stream_ps(pStore, out_i2);                                                                  
    //_mm256_stream_ps(pStore, out_i3);                                                                  
    //_mm256_stream_ps(pStore, out_i4);                                                                  

    pStore = pStore + 8;
    pLoad = pLoad + SKIP;

  }

  double ttt = 1.0*tt.elapsed();
  std::cout << ttt << std::endl;

  summm = 0.0;
  for(int i=0;i<c;i++){
    summm += a3[i];
  }
  std::cout << summm << std::endl;

}


int main(int argc, char ** argv){

	trya();

	/*
	float * const a1 = new float[c];
	float * const a2 = new float[c];
	float * const a3 = new float[c];
	for(int i=0;i<c;i++){
		a1[i] = drand48();
		a2[i] = drand48();
		//a1[i] = rand();
		//a2[i] = rand();
	}
	std::cout << "START!" << std::endl;

	for(int i=0;i<c;i++){
		a3[i] += a1[i] * a2[i];
	}

	tt.restart();
	//MultiSSE(a3, a1, a2, c);
	AddSSE(a3, a1, a2, c);
	double ttt = 1.0*tt.elapsed();
	std::cout << ttt << std::endl;

	double summm = 0.0;
	for(int i=0;i<c;i++){
		summm += a3[i]; 
	}
	std::cout << summm << std::endl;

	tt.restart();
	for(int i=0;i<c;i++){
		a3[i] = a1[i] + a2[i];
	}
	ttt = 1.0*tt.elapsed();
	std::cout << ttt << std::endl;

	summm = 0.0;
	for(int i=0;i<c;i++){
		summm += a3[i]; 
	}
	std::cout << summm << std::endl;
	*/


	float ** img = new float*[nrow];
	float * buf_img = new float[nrow*ncol];
	for(int r=0;r<nrow;r++){
		img[r] = &buf_img[r*ncol];
	}
	for(int i=0;i<nrow*ncol;i++){
		buf_img[i] = rand();
	}

	float ** output[1000];
	for(int i=0;i<nfeaturemap;i++){
		output[i] = new float*[nrow_output];
		for(int r=0;r<nrow_output;r++){
			output[i][r] = &buf_output[i*ncol_output*nrow_output + r*ncol_output];
		}
	}

	float ** kernel[1000];
	for(int i=0;i<nfeaturemap;i++){
		kernel[i] = new float*[nrow_kernel];
		for(int r=0;r<nrow_kernel;r++){
			kernel[i][r] = &buf_kernel[i*ncol_kernel*nrow_kernel + r*ncol_kernel];
		}
	}
	for(int i=0;i<nfeaturemap*nrow_kernel*ncol_kernel;i++){
		buf_kernel[i] = rand();
	}

	Timer t;
	for(int i_epoch=0;i_epoch<NRUN;i_epoch++){

		for(int i=0;i<nfeaturemap;i++){
			float ** coutput = output[i];
			float ** weights = kernel[i];
			for(int r=0;r<nrow_output;r++){
				for(int c=0;c<ncol_output;c++){
					float sum = 0.0;
					for(int ir=r;ir<r+nrow_kernel;ir++){
						for(int ic=c;ic<c+ncol_kernel;ic++){
							sum += weights[ir-r][ic-c] * img[ir][ic];
						}
					}
					coutput[r][c] = sum;
				}
			}
		}
	}
	float trainingtime = t.elapsed();
	std::cout << "Training " << trainingtime << " seconds..." << "  " <<
		(trainingtime/NRUN) << " seconds/image." << std::endl;
	float throughput = 1.0*nrow*ncol*sizeof(float)*NRUN/1024/1024/trainingtime;
	std::cout << "     THROUGHPUT = " << throughput << "MB/seconds..." << std::endl;


	// First, collapsed all feature maps

	float ** output_collapsed = new float*[nrow_output];
	float * buf_output_collapsed = new float[nfeaturemap*nrow_output*ncol_output];
	for(int r=0;r<nrow_output;r++){
		output_collapsed[r] = &buf_output_collapsed[r*nfeaturemap*ncol_output];
	}

	float ** kernel_collapsed = new float*[nrow_kernel];
	float * buf_kernel_collapsed = new float[nfeaturemap*nrow_kernel*ncol_kernel];
	for(int r=0;r<nrow_kernel;r++){
		kernel_collapsed[r] = &buf_kernel_collapsed[r*nfeaturemap*ncol_kernel];
	}

	for(int i=0;i<nfeaturemap*nrow_kernel*ncol_kernel;i++){
		buf_kernel_collapsed[i] = rand();
	}

	/*
	t.restart();
	run1(output_collapsed, kernel_collapsed, img);
	trainingtime = t.elapsed();
	std::cout << "Training " << trainingtime << " seconds..." << "  " <<
		(trainingtime/NRUN) << " seconds/image." << std::endl;
	throughput = 1.0*nrow*ncol*sizeof(float)*NRUN/1024/1024/trainingtime;
	std::cout << "     THROUGHPUT = " << throughput << "MB/seconds..." << std::endl;
	*/

	t.restart();
	run2(output_collapsed, kernel_collapsed, img);
	trainingtime = t.elapsed();
	std::cout << "Training " << trainingtime << " seconds..." << "  " <<
		(trainingtime/NRUN) << " seconds/image." << std::endl;
	throughput = 1.0*nrow*ncol*sizeof(float)*NRUN/1024/1024/trainingtime;
	std::cout << "     THROUGHPUT = " << throughput << "MB/seconds..." << std::endl;

	throughput = 1.0*size_of_stream*sizeof(float)*NRUN/1024/1024/trainingtime;
	std::cout << "     THROUGHPUT = " << throughput << "MB/seconds..." << std::endl;

	float sum = 0.0;
	for(int i=0;i<nfeaturemap*nrow_output*ncol_output;i++){
		sum+=buf_output[i];
	}
	std::cout << sum << std::endl;


	float sum2 = 0.0;
	for(int i=0;i<nfeaturemap*nrow_output*ncol_output;i++){
		sum2+=buf_output_collapsed[i];
	}
	std::cout << sum2 << std::endl;

}






	