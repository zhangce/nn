#ifndef _OPERATION_H
#define _OPERATION_H

#include <assert.h>
#include <math.h>

double STEPSIZE = 0.01;
//double INITWEIGHT = 0.1;

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

class Operation{
public:

	int groundtruth;

	int nrow_output;
	int ncol_output;

	int nrow_input;
	int ncol_input;

	int nrow_conv;
	int ncol_conv;

	double** inputs[1024];
	double** grads[1024];
	double** output;
	double* _buf;

	double** grad;
	double* _buf_grad;

	double ** weights[1024];
	double * _buf_weight;

	virtual void forward(){
		assert(false);
	}

	virtual void backward(){
		assert(false);
	}

	virtual void clear_grad(){
		assert(false);
	}

	Operation(int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input){
		nrow_output = _nrow_output;
		ncol_output = _ncol_output;
		nrow_input = _nrow_input;
		ncol_input = _ncol_input;
		
		_buf = new double[nrow_output*ncol_output];
		output = new double*[nrow_output];
		for(int i=0;i<nrow_output;i++){
			output[i] = &_buf[i*ncol_output];
		}

		_buf_grad = new double[nrow_output*ncol_output];
		grad = new double*[nrow_output];
		for(int i=0;i<nrow_output;i++){
			grad[i] = &_buf_grad[i*ncol_output];
		}

		nrow_conv = nrow_input - nrow_output + 1;
		ncol_conv = ncol_input - ncol_output + 1;
		_buf_weight = new double[nrow_conv*ncol_conv*1024];
		for(int j=0;j<1024;j++){
			weights[j] = new double*[nrow_conv];
			for(int i=0;i<nrow_conv;i++){
				weights[j][i] = &_buf_weight[i*ncol_conv + j*nrow_conv*ncol_conv];
			}
		}
	}

	Operation(bool isfull, int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input){
		nrow_output = _nrow_output;
		ncol_output = _ncol_output;
		nrow_input = _nrow_input;
		ncol_input = _ncol_input;
		
		_buf = new double[nrow_output*ncol_output];
		output = new double*[nrow_output];
		for(int i=0;i<nrow_output;i++){
			output[i] = &_buf[i*ncol_output];
		}

		_buf_grad = new double[nrow_output*ncol_output];
		grad = new double*[nrow_output];
		for(int i=0;i<nrow_output;i++){
			grad[i] = &_buf_grad[i*ncol_output];
		}
	}

};

class FullyConnectedOperation : public Operation{
public:

	double bias;

	void clear_grad(){
		if(grads[0] != NULL){
			for(int i_input=0;i_input<n_input;i_input++){
				for(int ir=0;ir<nrow_input;ir++){
					for(int ic=0;ic<ncol_input;ic++){
						grads[i_input][ir][ic] = 0.0;
					}
				}
			}
		}
	}

	int n_input;

	FullyConnectedOperation(int _n_input, int nrow_output, int ncol_output, int nrow_input, int ncol_input):
		Operation(nrow_output, ncol_output, nrow_input, ncol_input){
	
		n_input = _n_input;

		// init weights
		for(int i=0;i<n_input;i++){
			for(int r=0;r<nrow_conv;r++){
				for(int c=0;c<ncol_conv;c++){
					weights[i][r][c] = (drand48()*2-1)/10;
				}
			}
		}

		bias = (drand48()*2-1)/10;

	}

	void backward(){

		for(int r=0;r<nrow_output;r++){
			for(int c=0;c<ncol_output;c++){

				double cvalue = output[r][c];
				double cgrad = grad[r][c];

				for(int i_input=0;i_input<n_input;i_input++){
					for(int ir=r;ir<r+nrow_conv;ir++){
						for(int ic=c;ic<c+ncol_conv;ic++){

							double w = weights[i_input][ir-r][ic-c];
							double x = inputs[i_input][ir][ic];

							double grad_w = (1.0-cvalue*cvalue)*x * cgrad;
							double grad_x = (1.0-cvalue*cvalue)*w * cgrad;

							weights[i_input][ir-r][ic-c] = 
								weights[i_input][ir-r][ic-c] + STEPSIZE * grad_w;

							if(grads[0] != NULL){
								grads[i_input][ir][ic] += grad_x;
							}
						}
					}
				}

				double w = bias;
				double x = 1.0;
				double grad_w = (1.0-cvalue*cvalue)*x * cgrad;
				double grad_x = (1.0-cvalue*cvalue)*w * cgrad;
				bias = bias + STEPSIZE * grad_w;
			}
		}
	}

	void forward(){

		/*
		for(int r=0;r<nrow_output;r++){
			for(int c=0;c<ncol_output;c++){
				double sum = 0.0;
				for(int i_input=0;i_input<n_input;i_input++){
					for(int ir=r;ir<r+nrow_conv;ir++){
						for(int ic=c;ic<c+ncol_conv;ic++){
							sum += weights[i_input][ir-r][ic-c] * inputs[i_input][ir][ic];
						}
					}
				}
				sum += bias;
				output[r][c] = tanh(sum);
			}
		}
		*/

		/*
		for(int r=0;r<nrow_output;r++){
			for(int c=0;c<ncol_output;c++){
				output[r][c] = 0.0;
			}
		}
		*/

		/*
		for(int i_input=0;i_input<n_input;i_input++){
			double ** const pweight = weights[i_input];
			double ** const pinputs = inputs[i_input];
			
			for(int r=0;r<nrow_output;r++){
				for(int c=0;c<ncol_output;c++){
					for(int ir=r;ir<r+nrow_conv;ir++){
						for(int ic=c;ic<c+ncol_conv;ic++){
							output[r][c] += pweight[ir-r][ic-c] * pinputs[ir][ic];
						}
					}
				}
			}
		}
		*/
		
		/*
		for(int r=0;r<nrow_output;r++){
			for(int c=0;c<ncol_output;c++){
				output[r][c] = tanh(output[r][c]);
			}
		}
		*/

		/*
		const int nele = nrow_input * ncol_input;
		const double * const pstart = &inputs[0][0][0];
		double sum = 0.0;
		for(int i=0;i<nele;i++){
			sum += pstart[i];
		}
		output[0][0] = sum;
		*/

		/*
		std::cout << "-------FULL-------" << std::endl;
		for(int r=0;r<nrow_output;r++){
			for(int c=0;c<ncol_output;c++){
				std::cout << output[r][c] << " ";
			}
			std::cout << std::endl;
		}
		*/
	}

};


class MaxPoolingOperation : public Operation{
public:

	void clear_grad(){
		if(grads[0] != NULL){
			for(int i_input=0;i_input<n_input;i_input++){
				for(int ir=0;ir<nrow_input;ir++){
					for(int ic=0;ic<ncol_input;ic++){
						grads[i_input][ir][ic] = 0.0;
					}
				}
			}
		}
	}

	int n_input;

	MaxPoolingOperation(int _n_input, int nrow_output, int ncol_output, int nrow_input, int ncol_input):
		Operation(nrow_output, ncol_output, nrow_input, ncol_input){
	
		n_input = _n_input;

		assert(nrow_input % nrow_output == 0);
		assert(ncol_input % ncol_output == 0);
		assert(n_input == 1);
			// TODO: NEED WORK

	}

	void backward(){

		int row_ratio = nrow_input/nrow_output;
		int col_ratio = ncol_input/ncol_output;

		for(int r=0;r<nrow_output;r++){
			for(int c=0;c<ncol_output;c++){
				double cvalue = output[r][c];
				double cgrad = grad[r][c];
				for(int ir=r*row_ratio;ir<r*row_ratio+row_ratio;ir++){
					for(int ic=c*col_ratio;ic<c*col_ratio+col_ratio;ic++){
						if(inputs[0][ir][ic] == cvalue){
							grads[0][ir][ic] += cgrad;
						}else{
							grads[0][ir][ic] = 0;
						}
					}
				}
			}
		}

	}

	void forward(){

		int row_ratio = nrow_input/nrow_output;
		int col_ratio = ncol_input/ncol_output;

		for(int r=0;r<nrow_output;r++){
			for(int c=0;c<ncol_output;c++){
				double max = -10000;
				for(int ir=r*row_ratio;ir<r*row_ratio+row_ratio;ir++){
					for(int ic=c*col_ratio;ic<c*col_ratio+col_ratio;ic++){
						if(inputs[0][ir][ic] > max){
							max = inputs[0][ir][ic];
						}
					}
				}
				output[r][c] = max;
			}
		}

		/*
		std::cout << "-------FULL-------" << std::endl;
		for(int r=0;r<nrow_output;r++){
			for(int c=0;c<ncol_output;c++){
				std::cout << output[r][c] << " ";
			}
			std::cout << std::endl;
		}
		*/
	}

};


class SoftmaxOperation : public Operation{
public:

	int n_input;

	double** softweights;
	double* biases;
	int n_label;

	void clear_grad(){
		for(int i_input=0;i_input<n_input;i_input++){
			for(int ir=0;ir<nrow_input;ir++){
				for(int ic=0;ic<ncol_input;ic++){
					grads[i_input][ir][ic] = 0.0;
				}
			}
		}
	}

	SoftmaxOperation(int _n_input, int _nlabel, int nrow_input, int ncol_input):
		Operation(true, 1, _nlabel, nrow_input, ncol_input){
	
		n_input = _n_input;
		n_label = _nlabel;

		assert(nrow_input == 1);
		assert(ncol_input == 1);
		assert(nrow_output == 1);

		softweights = new double*[n_label];
		biases = new double[n_label];
		for(int i=0;i<n_label;i++){
			softweights[i] = new double[n_input];
			biases[i] = 0;
			for(int j=0;j<n_input;j++){
				softweights[i][j] = (drand48()*2-1)/10;
			}
		}

	}

	void backward(){

		for(int label=0;label<n_label;label++){
			double cvalue = output[0][label];

			for(int i_input=0;i_input<n_input;i_input++){

				double w = softweights[label][i_input];
				double x = inputs[i_input][0][0];

				double grad_w = (label == groundtruth)*x - cvalue*x;
				double grad_x = (label == groundtruth)*w - cvalue*w;

				softweights[label][i_input] = 
					softweights[label][i_input] + STEPSIZE * grad_w;

				grads[i_input][0][0] += grad_x;

			}

			double w = biases[label];
			double x = 1.0;
			double grad_w = (label == groundtruth)*x - cvalue*x;
			double grad_x = (label == groundtruth)*w - cvalue*w;
			biases[label] = biases[label] + STEPSIZE * grad_w;
		}

	}

	void forward(){
		for(int i=0;i<n_label;i++){
			double sum = 0.0;
			for(int i_input=0;i_input<n_input;i_input++){
				sum += softweights[i][i_input] * inputs[i_input][0][0];
			}
			sum += biases[i];
			output[0][i] = sum;
		}

		double sum = -100000;
		for(int i=0;i<n_label;i++){
			sum = logadd(sum, output[0][i]); 
		}
		for(int i=0;i<n_label;i++){
			output[0][i] = exp(output[0][i]-sum);
		}

	}

};

#endif










