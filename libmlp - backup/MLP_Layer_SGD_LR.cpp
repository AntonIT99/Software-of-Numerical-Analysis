///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "MLP_Layer_SGD_LR.h"

#ifdef _OPENMP
    #include <omp.h>
#endif


void MLP_Layer_SGD_LR::BackwardPropagateOutputLayer(FLOAT_TYPE *desiredValues)
{
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
// START HERE
// this commented code comes from MLP_Layer_SGD::BackwardPropagateOutputLayer
//    for (int k = 0; k < nCurrentNeurons; k++)
//        {
//        Delta[k] = DerivativeActivation(outputLayer[k]) * (desiredValues[k] - outputLayer[k]);
//        }
  
    for (int i = 0; i < nCurrentNeurons; i++)
    {
        Delta[i] = -(outputLayer[i] - desiredValues[i]);
    }
// END HERE 
  
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int k = 0 ; k < nCurrentNeurons ; k++)
        for (int j = 0 ; j < nPreviousNeurons; j++)
            dW[k*nPreviousNeurons + j] += - (Delta[k] * inputLayer[j]);

#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int k = 0 ; k < nCurrentNeurons   ; k++)
            db[k] += - Delta[k] ;
}
