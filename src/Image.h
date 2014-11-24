
#include <iostream>

#ifndef _IMAGE_H
#define _IMAGE_H

class Image{
public:
	int img_id;
	int label;
	int nrows;
	int ncols;

	double ** pixels;

	double * _buf;

	Image(int _img_id, int _label, int _nrow, int _ncol){
		img_id = _img_id;
		nrows = _nrow;
		ncols = _ncol;
		label = _label;
		_buf = new double[nrows*ncols];
		pixels = new double*[nrows];
		for(int i=0;i<nrows;i++){
			pixels[i] = &_buf[i*ncols];
		}
	}

	void show(){
		for(int r=0;r<nrows;r++){
			for(int c=0;c<ncols;c++){
				std::cout << pixels[r][c] <<  " " ;
			}
			std::cout << std::endl;
		}
	}

};

#endif