
#include "Image.h"
#include <iostream>

#include "Corpus.h"
#include "Operation.h"
#include "Network.h"

#include "timer.h"

int DIGIT=10;

int main(int argc, char ** argv){

	MNISTCorpus corpus("input/train-labels-idx1-ubyte", "input/train-images-idx3-ubyte");

	// Build Network

	Network network(1);
	network.layers[0] = new Layer(6);
	Layer * layer1 = network.layers[0];
	for(int i=0;i<6;i++){
		layer1->operations[i]
			= (Operation*) new FullyConnectedOperation(1, 24, 24, corpus.n_rows, corpus.n_cols);
		layer1->operations[i]->inputs[0] = corpus.images[0]->pixels;
		layer1->operations[i]->grads[0] = NULL;
	}

	for(int i_epoch=0;i_epoch<100000;i_epoch++){
		int ncorr[10];
		int ncorr_neg[10];
		int npos[10];
		int nneg[10];
		for(int i=0;i<10;i++){
			ncorr[i] = 0;
			ncorr_neg[i] = 0;
			npos[i] = 0;
			nneg[i] = 0;
		}
		double loss = 0.0;
		double loss_test = 0.0;

		Timer t;
		for(int i_img=0;i_img<corpus.n_image;i_img++){
			for(int i=0;i<4;i++){
				layer1->operations[i]->inputs[0] 
					= corpus.images[i_img]->pixels;
			}
			network.forward();
		}
		double trainingtime = t.elapsed();
		std::cout << "Training " << trainingtime << " seconds..." << "  " <<
			(trainingtime/corpus.n_image) << " seconds/image." << std::endl;
		double throughput = 1.0*corpus.n_rows*corpus.n_cols*sizeof(double)*corpus.n_image/1024/1024/trainingtime;
		std::cout << "     THROUGHPUT = " << throughput << "MB/seconds..." << std::endl;
	}

	std::cout << "DONE" << std::endl;

	return 0;
}


void LeNet5(){

	MNISTCorpus corpus("input/train-labels-idx1-ubyte", "input/train-images-idx3-ubyte");
	MNISTCorpus corpus_test("input/t10k-labels-idx1-ubyte", "input/t10k-images-idx3-ubyte");

	// Build Network

	Network network(6);
	network.layers[0] = new Layer(6);
	Layer * layer1 = network.layers[0];
	for(int i=0;i<6;i++){
		layer1->operations[i]
			= (Operation*) new FullyConnectedOperation(1, 24, 24, corpus.n_rows, corpus.n_cols);
		layer1->operations[i]->inputs[0] = corpus.images[0]->pixels;
		layer1->operations[i]->grads[0] = NULL;
	}

	network.layers[1] = new Layer(6);
	Layer * layer2 = network.layers[1];
	for(int i=0;i<6;i++){
		layer2->operations[i]
			= (Operation*) new MaxPoolingOperation(1, 12, 12, 24, 24);
		layer2->operations[i]->inputs[0] = layer1->operations[i]->output;
		layer2->operations[i]->grads[0] = layer1->operations[i]->grad;
	}

	network.layers[2] = new Layer(32);
	Layer * layer3 = network.layers[2];
	for(int i=0;i<32;i++){
		layer3->operations[i]
			= (Operation*) new FullyConnectedOperation(6, 8, 8, 12, 12);
		for(int j=0;j<6;j++){
			layer3->operations[i]->inputs[j] = layer2->operations[j]->output;
			layer3->operations[i]->grads[j] = layer2->operations[j]->grad;
		}
	}

	network.layers[3] = new Layer(32);
	Layer * layer4 = network.layers[3];
	for(int i=0;i<32;i++){
		layer4->operations[i]
			= (Operation*) new MaxPoolingOperation(1, 4, 4, 8, 8);
		layer4->operations[i]->inputs[0] = layer3->operations[i]->output;
		layer4->operations[i]->grads[0] = layer3->operations[i]->grad;
	}

	int NLAYER5=120;
	network.layers[4] = new Layer(NLAYER5);
	Layer * layer5 = network.layers[4];
	for(int i=0;i<NLAYER5;i++){
		layer5->operations[i]
			= (Operation*) new FullyConnectedOperation(32, 1, 1, 4, 4);
		for(int j=0;j<32;j++){
			layer5->operations[i]->inputs[j] = layer4->operations[j]->output;
			layer5->operations[i]->grads[j] = layer4->operations[j]->grad;
		}
	}

	network.layers[5] = new Layer(1);
	Layer * layer6 = network.layers[5];
	layer6->operations[0]
		= (Operation*) new SoftmaxOperation(NLAYER5, DIGIT, 1, 1);
	for(int i=0;i<NLAYER5;i++){
		layer6->operations[0]->inputs[i]
			= layer5->operations[i]->output;
		layer6->operations[0]->grads[i]
			= layer5->operations[i]->grad;
	}

	for(int i_epoch=0;i_epoch<100000;i_epoch++){
		int ncorr[10];
		int ncorr_neg[10];
		int npos[10];
		int nneg[10];
		for(int i=0;i<10;i++){
			ncorr[i] = 0;
			ncorr_neg[i] = 0;
			npos[i] = 0;
			nneg[i] = 0;
		}
		double loss = 0.0;
		double loss_test = 0.0;

		Timer t;
		
		for(int i_img=0;i_img<corpus.n_image;i_img++){

			layer6->operations[0]->groundtruth 
				= (corpus.images[i_img]->label);
			for(int i=0;i<4;i++){
				layer1->operations[i]->inputs[0] 
					= corpus.images[i_img]->pixels;
			}

			network.forward();
			//network.backward();	
		}
		double trainingtime = t.elapsed();
		std::cout << "Training " << trainingtime << " seconds..." << "  " <<
			(trainingtime/corpus.n_image) << " seconds/image." << std::endl;

		t.restart();
		for(int i_img=0;i_img<corpus_test.n_image;i_img++){

			layer6->operations[0]->groundtruth 
				= (corpus_test.images[i_img]->label);
			for(int i=0;i<4;i++){
				layer1->operations[i]->inputs[0] 
					= corpus_test.images[i_img]->pixels;
			}

			network.forward();
			
			int gt = (corpus_test.images[i_img]->label);
			int imax;
			double ifloat = -1;
			for(int dig=0;dig<DIGIT;dig++){
				double out = layer6->operations[0]->output[0][dig];
				if(out > ifloat){
					imax = dig;
					ifloat = out;
				}
			}
			nneg[gt] ++;
			ncorr_neg[gt] += (gt==imax);
			loss_test += (gt==imax);
		}
		double testingtime = t.elapsed();
		std::cout << "Testing " << t.elapsed() << " seconds..." << "  " <<
			(testingtime/corpus_test.n_image) << " seconds/image." << std::endl;
		
		std::cout << "----TEST----" << loss_test/corpus_test.n_image << std::endl;
		for(int dig=0;dig<DIGIT;dig++){
			std::cout << "## DIG=" << dig << " : ";
			std::cout << 1.0*ncorr_neg[dig]/nneg[dig] << " = " << ncorr_neg[dig] << "/" << nneg[dig] << std::endl;
		}

	}

	std::cout << "DONE" << std::endl;

}



	/*
	Network network(3);

	network.layers[0] = new Layer(2);
	for(int i=0;i<2;i++){
		network.layers[0]->operations[i]
			= (Operation*) new ConvOperation(5, 5, corpus.n_rows, corpus.n_cols);
		network.layers[0]->operations[i]->inputs[0]
			= corpus.images[0]->pixels;
		network.layers[0]->operations[i]->grads[0]
			= NULL;	
	}

	network.layers[1] = new Layer(10);
	for(int i=0;i<10;i++){
		network.layers[1]->operations[i]
			= (Operation*) new FullyConnectedOperation(2, 1, 1, 5, 5);
		network.layers[1]->operations[i]->inputs[0]
			= network.layers[0]->operations[0]->output;
		network.layers[1]->operations[i]->inputs[1]
			= network.layers[0]->operations[1]->output;

		network.layers[1]->operations[i]->grads[0]
			= network.layers[0]->operations[0]->grad;
		network.layers[1]->operations[i]->grads[1]
			= network.layers[0]->operations[1]->grad;
	}

	network.layers[2] = new Layer(1);
	network.layers[2]->operations[0]
		= (Operation*) new LogisticOperation(10, 1, 1, 1, 1);
	for(int i=0;i<10;i++){
		network.layers[2]->operations[0]->inputs[i]
			= network.layers[1]->operations[i]->output;
		network.layers[2]->operations[0]->grads[i]
			= network.layers[1]->operations[i]->grad;
	}

	for(int i_epoch=0;i_epoch<100;i_epoch++){
		int ncorr = 0;
		int ncorr_neg = 0;
		int npos = 0;
		int nneg = 0;
		double loss = 0.0;
		for(int i_img=0;i_img<corpus.n_image;i_img++){
			network.layers[2]->operations[0]->groundtruth 
				= corpus.images[i_img]->label == 3;
			network.layers[0]->operations[0]->inputs[0]
				= corpus.images[i_img]->pixels;
			network.layers[0]->operations[1]->inputs[0]
				= corpus.images[i_img]->pixels;
			network.forward();

			int gt = network.layers[2]->operations[0]->groundtruth;
			double out = network.layers[2]->operations[0]->output[0][0];

			
			if(gt == 1){
				npos ++;
				ncorr += ((out>0.5) == gt);
			}else{
				nneg ++;
				ncorr_neg += ((out>0.5) == gt);
			}

			loss += fabs(gt-out);

			network.backward();
		}
		std::cout << "------------" << loss/corpus.n_image << std::endl;
		std::cout << "+ " << 1.0*ncorr/npos << " = " << ncorr << "/" << npos << std::endl;
		std::cout << "- " << 1.0*ncorr_neg/nneg << " = " << ncorr_neg << "/" << nneg << std::endl;
	}
	*/







