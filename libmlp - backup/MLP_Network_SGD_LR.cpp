///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "MLP_Network_SGD_LR.h"

#ifdef _OPENMP
    #include <omp.h>
#endif


void MLP_Network_SGD_LR::Allocate(int nInputUnit,   int nHiddenUnit, int nOutputUnit, int nHiddenLayer,
                           int nTrainingSet)
{
    this->nTrainingSet     = nTrainingSet;
    this->nInputNeurons    = nInputUnit;
    this->nHiddenNeurons   = nHiddenUnit;
    this->nOutputNeurons   = nOutputUnit;
    this->nHiddenLayer     = nHiddenLayer;

    layerNetwork = new MLP_Layer_SGD_LR[nHiddenLayer+1];
    
    layerNetwork[0].Allocate(nInputUnit, nHiddenUnit);
    for (int i = 1; i < nHiddenLayer; i++)
    {
        layerNetwork[i].Allocate(nHiddenUnit, nHiddenUnit);
        layerNetwork[i].SetActivationFunction(Activation_function);
    }
    layerNetwork[nHiddenLayer].Allocate(nHiddenUnit, nOutputUnit);
    layerNetwork[nHiddenLayer].SetActivationFunction('S');
}

void MLP_Network_SGD_LR::Delete()
{
    for (int i = 0; i < nHiddenLayer+1; i++)
    {
        layerNetwork[i].Delete();
    }
}


void MLP_Network_SGD_LR::ForwardPropagateNetwork(float* inputNetwork)
{
    float* outputOfHiddenLayer=NULL;

    outputOfHiddenLayer=layerNetwork[0].ForwardPropagate(inputNetwork);
    for (int i=1; i < nHiddenLayer ; i++)
    {
        outputOfHiddenLayer=layerNetwork[i].ForwardPropagate(outputOfHiddenLayer);                  //hidden forward
    }
    layerNetwork[nHiddenLayer].ForwardPropagate(outputOfHiddenLayer);      // output forward
}

void MLP_Network_SGD_LR::BackwardPropagateNetwork(float* desiredOutput)
{
    layerNetwork[nHiddenLayer].BackwardPropagateOutputLayer(desiredOutput);  // back_propa_output
    for (int i= nHiddenLayer-1; i >= 0  ; i--)
        layerNetwork[i].BackwardPropagateHiddenLayer(&layerNetwork[i+1]);    // back_propa_hidden
}

void MLP_Network_SGD_LR::UpdateWeight(float learningRate)   // update weight according gradient and standart gradient descent
{
    for (int i = 0; i < nHiddenLayer; i++)
        layerNetwork[i].UpdateWeight(learningRate);

    layerNetwork[nHiddenLayer].UpdateWeight(learningRate);
}

float MLP_Network_SGD_LR::CostFunction(float* desiredOutput)
{
    float *outputNetwork = layerNetwork[nHiddenLayer].GetOutput();
    float err = 0.F;
	
// START HERE

// this commented code comes from MLP_Network_SGD::CostFunction
//    for (int j = 0; j < nOutputNeurons; ++j)
//        err += (desiredOutput[j] - outputNetwork[j])*(desiredOutput[j] - outputNetwork[j]);
//    err /= 2;
//    return err;

    for (int j = 0; j < nOutputNeurons; j++)
    {
        err +=  desiredOutput[j] * log(outputNetwork[j] + 0.00001) + (1 - desiredOutput[j]) * log(1- outputNetwork[j] + 0.00001);
    }
    return -err;

// END HERE
  
}


float MLP_Network_SGD_LR::CalculateResult(float* desiredOutput)
{
    int maxIdx = 0;

    if( layerNetwork[nHiddenLayer].GetNumCurrent() > 2 )
        {
        maxIdx = layerNetwork[nHiddenLayer].GetMaxOutputIndex();
        if(desiredOutput[maxIdx] == 1.0f)
            return 1;
        return 0;
        }
    else
        {
        if( layerNetwork[nHiddenLayer].GetBinaryOutput() == desiredOutput[0] )
            return 1;
        else
            return 0;
    }
}


ifstream& operator>>(ifstream& is, MLP_Network_SGD_LR& mlp)
{
    int nInputUnit, nHiddenUnit, nOutputUnit, nHiddenLayer;

    is.read( reinterpret_cast<char *>( &nInputUnit  ), sizeof(int) );
    is.read( reinterpret_cast<char *>( &nHiddenUnit ), sizeof(int) );
    is.read( reinterpret_cast<char *>( &nOutputUnit ), sizeof(int) );
    is.read( reinterpret_cast<char *>( &nHiddenLayer), sizeof(int) );

    mlp.Allocate(nInputUnit, nHiddenUnit, nOutputUnit, nHiddenLayer, 0);

    for (int i = 0; i < mlp.nHiddenLayer + 1; i++)
        is >> mlp.layerNetwork[i];

    return is;
}

ofstream& operator<<(ofstream& os, const MLP_Network_SGD_LR& mlp)
{
    os.write(reinterpret_cast<const char *>( &(mlp.nInputNeurons)   ), sizeof(int) );
    os.write(reinterpret_cast<const char *>( &(mlp.nHiddenNeurons)  ), sizeof(int) );
    os.write(reinterpret_cast<const char *>( &(mlp.nOutputNeurons)  ), sizeof(int) );
    os.write(reinterpret_cast<const char *>( &(mlp.nHiddenLayer) ), sizeof(int) );

    for (int i = 0; i < mlp.nHiddenLayer + 1; i++)
        os << mlp.layerNetwork[i];

    return os;
}



