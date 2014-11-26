
#include "Image.h"

#ifndef _NETWORK_H
#define _NETWORK_H

class Layer{
public:

	int n_operations;
	Operation ** operations;

	Layer(int _n_operation){
		n_operations = _n_operation;
		operations = new Operation*[_n_operation];
	}

	void forward(){
		for(int i_operation=0;i_operation<n_operations;i_operation++){
			Operation * const operation = operations[i_operation];
			operation->forward();
		}
	}

	void backward(){
		for(int i_operation=0;i_operation<n_operations;i_operation++){
			Operation * operation = operations[i_operation];
			operation->backward();
		}
	}

	void clear_grad(){
		for(int i_operation=0;i_operation<n_operations;i_operation++){
			Operation * operation = operations[i_operation];
			operation->clear_grad();
		}
	}

};


class Network{
public:

	int n_layer;
	Layer ** layers;

	Network(int _n_layer){
		n_layer = _n_layer;
		layers = new Layer*[n_layer];
	}

	void forward(){
		for(int i_layer=0; i_layer<n_layer; i_layer++){
			Layer * const layer = layers[i_layer];
			layer->forward();
		}
	}

	void backward(){

		for(int i_layer=0; i_layer<n_layer; i_layer++){
			Layer * layer = layers[i_layer];
			layer->clear_grad();
		}

		for(int i_layer=n_layer-1;i_layer>=0;i_layer--){
			Layer * layer = layers[i_layer];
			layer->backward();
		}
	}

};


#endif

