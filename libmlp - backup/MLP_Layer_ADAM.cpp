///////////////////////////////////////////////////////////
//          YOU HAVE TO MODIFY THIS FILE                 //
///////////////////////////////////////////////////////////

#include "MLP_Layer_ADAM.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

void MLP_Layer_ADAM::Allocate(int previous_num, int current_num)
{
    MLP_Layer::Allocate( previous_num, current_num);

    MW       = new INTERNAL_FLOAT_TYPE[nPreviousNeurons * nCurrentNeurons];
    Mb       = new INTERNAL_FLOAT_TYPE[nCurrentNeurons];
    MW_next   = new INTERNAL_FLOAT_TYPE[nPreviousNeurons * nCurrentNeurons];
    Mb_next   = new INTERNAL_FLOAT_TYPE[nCurrentNeurons];

    SW       = new INTERNAL_FLOAT_TYPE[nPreviousNeurons * nCurrentNeurons];
    Sb       = new INTERNAL_FLOAT_TYPE[nCurrentNeurons];
    SW_next   = new INTERNAL_FLOAT_TYPE[nPreviousNeurons * nCurrentNeurons];
    Sb_next   = new INTERNAL_FLOAT_TYPE[nCurrentNeurons];

    for (int j = 0; j < nCurrentNeurons; j++)
        {
        for (int i = 0; i < nPreviousNeurons; i++)
            {
            MW[j*nPreviousNeurons+i] = 0.0;
            SW[j*nPreviousNeurons+i] = 0.0;
            }
        Mb[j]  = 0.0;
        Sb[j]  = 0.0;
        }
}



void MLP_Layer_ADAM::Delete()
{
 // ADAM
    if( MW          != nullptr) { delete [] MW;            MW = nullptr; }
    if( Mb          != nullptr) { delete [] Mb;            Mb = nullptr; }
    if( MW_next     != nullptr) { delete [] MW_next;  MW_next = nullptr; }
    if( Mb_next     != nullptr) { delete [] Mb_next;  Mb_next = nullptr; }

    if( SW          != nullptr) { delete [] SW;            SW = nullptr; }
    if( Sb          != nullptr) { delete [] Sb;            Sb = nullptr; }
    if( SW_next     != nullptr) { delete [] SW_next;  SW_next = nullptr; }
    if( Sb_next     != nullptr) { delete [] Sb_next;  Sb_next = nullptr; }
}

void MLP_Layer_ADAM::UpdateWeight(FLOAT_TYPE learningRate)
{
   INTERNAL_FLOAT_TYPE Beta1_T =  pow( Beta1, T);
   INTERNAL_FLOAT_TYPE Beta2_T =  pow( Beta2, T);

   INTERNAL_FLOAT_TYPE alpha_T = learningRate * sqrt(1.0 - Beta2_T) / (1.0 - Beta1_T);


////////////////////////////////////////
// START YOUR ADAM IMPLEMENTATION HERE

//////////////// W
    for(int j=0; j<nCurrentNeurons; j++)
    {
        for(int i=0; i<nPreviousNeurons; i++)
        {
            MW_next[j*nPreviousNeurons + i] = Beta1 * MW[j*nPreviousNeurons + i] + (1-Beta1)* dW[j*nPreviousNeurons + i];
            MW_next[j*nPreviousNeurons + i] = MW_next[j*nPreviousNeurons + i] / (1-Beta1_T);

            SW_next[j*nPreviousNeurons + i] = Beta2 * SW[j*nPreviousNeurons + i] + (1-Beta2)* pow(dW[j*nPreviousNeurons + i],2);
            SW_next[j*nPreviousNeurons + i] = SW_next[j*nPreviousNeurons + i] / (1-Beta2_T);

            W[j*nPreviousNeurons + i] +=  -alpha_T * MW_next[j*nPreviousNeurons + i] / (sqrt(SW_next[j*nPreviousNeurons + i]) + Epsilon);
        }
    }

//////////////// b
    for(int j=0; j<nCurrentNeurons; j++)
    {
        Mb_next[j]= Beta1 * Mb[j] + (1-Beta1) * db[j];
        Mb_next[j]= Mb_next[j] / (1-Beta1_T);

        Sb_next[j]= Beta2 * Sb[j] + (1-Beta2) * pow (db[j],2);
        Sb_next[j]= Sb_next[j]/(1-Beta2_T);

        b[j] += -alpha_T * Mb_next[j] / (sqrt(Sb_next[j]) + Epsilon);
    }

// NO MORE MODIFICATIONS AFTER THIS LINE
////////////////////////////////////////

// Update ADAM parameters
    INTERNAL_FLOAT_TYPE *swap_w;  // permutation des pointeurs
    swap_w = MW;
    MW     = MW_next;
    MW_next = swap_w;

    swap_w = SW;
    SW     = SW_next;
    SW_next = swap_w;

    INTERNAL_FLOAT_TYPE *swap_b;
    swap_b = Mb;
    Mb     = Mb_next;
    Mb_next = swap_b;

    swap_b = Sb;
    Sb     = Sb_next;
    Sb_next = swap_b;

 //  T++;   Done outside (in MLP_Network_ADAM::UpdateWeight )

// Clear gradient dW and db
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int j = 0; j < nCurrentNeurons; j++)
        for (int i = 0; i < nPreviousNeurons; i++)
            dW[j*nPreviousNeurons + i] = 0;

#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int j = 0; j < nCurrentNeurons; j++)
        db[j]=0;
}



