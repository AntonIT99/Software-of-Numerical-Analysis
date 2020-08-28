#include <iostream>
#include <string>

#include "MLP_Network_SGD.h"
#include "MLP_Network_ADAM.h"

#include "Echo.h"

#include "timing_functions.h"   //tic() & tac() &  duration() functions

using namespace std;

const string SRC_PATH = "C:\\Users\\alpha\\Documents\\Qt_Projects\\clanu\\";

int main(int argc, char *argv[])
{
    if( argc < 2)
        {
        cerr << " Usage : " << argv[0] << " file.bin" << endl;
        cerr << " where : file.bin is the architecture and weights of the network to be loaded" << endl;
        return -1;
        }

    int nInputUnit      = 64*64;
    int nOutputUnit     = 1;

    int nTrainingSet    = 800;
    int nTestSet        = 200;
       
    //Allocate
    float **inputTest			= new float*[nTestSet];
    float **desiredOutputTest	= new float*[nTestSet];
    
    for(int i = 0;i < nTestSet;i++)
        {
        inputTest[i]			= new float[nInputUnit];
        desiredOutputTest[i]	= new float[nOutputUnit];
        }

#if defined(_OPENMP)
    cout << " OPENMP is activated : great! " << endl;
#else
    cout << " OPENMP is not activated (good for debug)" << endl;
#endif

#ifdef __FAST_MATH__
    cout << " fast-math is activated : great! " << endl;
#else
    cout << " fast-math is strangly not activated " << endl;
#endif

    
    //MNIST Input Array Allocation and Initialization
    ECHO echo;
    cout << " Reading directory : " << SRC_PATH+"data\\ECHO" << "  " <<  echo.ReadPath(SRC_PATH+"data\\ECHO") << endl;

    echo.ReadInput(nTrainingSet, nTestSet, inputTest);
    echo.ReadLabel(nTrainingSet, nTestSet, desiredOutputTest);

    cout << "Reading network models (architecture and weights) from : " << SRC_PATH << "models\\" << argv[1] << endl;
    MLP_Network_ADAM mlp;
    std::ifstream is (SRC_PATH+"models\\"+argv[1], std::ifstream::binary);
    is >> mlp;
    is.close();
   
    tic();
    //TEST ACCURACY and LOSS
    int sums=0;
    float LossTest=0.F;
    float AccuracyTest=0.F;
    for( int i=0; i<nTestSet; i++)
        {
        mlp.ForwardPropagateNetwork(inputTest[i]);
        sums += mlp.CalculateResult(desiredOutputTest[i]);
        LossTest += mlp.CostFunction( desiredOutputTest[i] );
        }
    AccuracyTest = (sums / (float)nTestSet) * 100;
    LossTest /= nTestSet;
    tac();
    cout << "[Test Set]\t Loss : "<< LossTest << "   Accuracy : " << AccuracyTest << " %"<< "   (compute time : "<< duration() << ")" << endl;


/////////////////////////////////////////////////////////////////
//ADD HERE THE CODE THAT PRINT ALL ERRONEOUS PREDICTED IMAGE INDEX

    int i=0;
    int cpt_Tab_Erreur=0;
    int Tab_Erreur[10]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};

    while(i<nTestSet)
    {
        mlp.ForwardPropagateNetwork(inputTest[i]);
        float Predicted = mlp.GetLayerNetwork()[mlp.GetnHiddenLayer()].GetBinaryOutput();
        if(Predicted  != desiredOutputTest[i][0])
        {
            Tab_Erreur[cpt_Tab_Erreur]=i;
            cpt_Tab_Erreur++;
        }
    i++ ;
    }
    i=0;
    cout << "indices des predictions erronees :" << endl;
    while((Tab_Erreur[i]!=-1)&&(i<9))
    {
        cout << Tab_Erreur[i] << endl;
        //ECHO::PrintImage(inputTest[Tab_Erreur[i]],64,64);
        i++;
    }

//STOP HERE THE CODE THAT PRINT ALL ERRONEOUS PREDICTED IMAGE INDEX (no more changes after this line)
/////////////////////////////////////////////////////////////////


// Ask for an image and print it until -1
    int ind_im;
    cout << " which image ? ";
    cin >> ind_im;

while(ind_im != -1)
    {
    ECHO::PrintImage(inputTest[ind_im], 64, 64);
    mlp.ForwardPropagateNetwork(inputTest[ind_im]);
    float Predicted = mlp.GetLayerNetwork()[mlp.GetnHiddenLayer()].GetBinaryOutput();    // récuperation de la valeur prédite par le réseau
    cout << " predicted : " << Predicted << "   true : " << desiredOutputTest[ind_im][0] << endl; //affichage prédiction et vérité

    cout << " which image ? ";
    cin >> ind_im;
    }


    for(int i=0; i<nTestSet; i++)
        {
        delete [] desiredOutputTest[i];
        delete [] inputTest[i];
        }

    delete[] inputTest;
    delete[] desiredOutputTest;
    
    return 0;
}
