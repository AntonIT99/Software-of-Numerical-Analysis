///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef MLP_Layer_SGD_LR_H
#define MLP_Layer_SGD_LR_H

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>

#include "MLP_Layer_SGD.h"

using namespace std;

//! Classe d'une couche du réseau
class  MLP_Layer_SGD_LR : public MLP_Layer_SGD
{  
public:
	//! contructeur
    MLP_Layer_SGD_LR(char activation_function='R'):MLP_Layer_SGD(activation_function) { }

    //! passe arrière pour la couche de sortie
    virtual void BackwardPropagateOutputLayer(FLOAT_TYPE* desiredValues);
};

#endif
