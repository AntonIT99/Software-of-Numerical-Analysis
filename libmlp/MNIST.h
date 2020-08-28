///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef MNIST_PARSER_H
#define MNIST_PARSER_H


#include <iostream>
#include <fstream>

using namespace std;

const string SRC_PATH="D:\\Documents\\Owncloud\\documents\\Insa-GE\\2019\\Clanu\\cpp_qmake";


//! classe de lecture des données MNIST
class MNIST {
public:
//! lecture des données 'image' (les entrées du réseau)
    void ReadInput(string filename, int num_images, float** inputs);
//! lecture des étiquettes (les sorties du réseau, pour l'entrainement)
    void ReadLabel(string filename, int num_labels, float** outputs);
private:
	int BytetoInt(int byte); // convert Byte to Int
};

//! Convert float* of labels to int value (the number value)
int IndexFromByte(float *bb);

//! Print small image in console
void PrintImage(float *im, int R, int C);

#endif
