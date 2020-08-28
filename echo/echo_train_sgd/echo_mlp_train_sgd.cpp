#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <algorithm>
#include <utility>
#include <thread>
#include <mutex>
#include <csignal>

#include "MLP_Network_SGD.h"
#include "MLP_Layer_SGD.h"
#include "Echo.h"
#include "timing_functions.h"
#include "progress_bar.h"
#include "gnuplot_utilities.h"

using namespace std;


// LINUX / MAC
//    const string SRC_PATH = "/home/tgrenier/Documents/Clanu19/cpp";
//    const string GNUPLOT_PATH="/usr/bin/gnuplot";

// WINDOWS
    const string SRC_PATH = "C:\\Users\\alpha\\Documents\\Qt_Projects\\clanu\\";
    const string GNUPLOT_PATH="C:\\\"Program Files\"\\gnuplot\\bin\\wgnuplot.exe";


std::string *sgd_log_filename_ptr;
int * maxEpoch_ptr;


void signal_callback_handler(int signum);

void ACCURACYandLOSS(int Iteration_boucle,float **Donne_Entree,float **Donne_Desiree, MLP_Network_SGD & mlp_fonction, float *Accuracy, float *Loss)
{
        int sums=0;
        float Accuracy_Tampon=0;
        float Loss_Tampon=0;
        for(int i=0; i<Iteration_boucle; i++)
        {
            mlp_fonction.ForwardPropagateNetwork(Donne_Entree[i]);
            sums += mlp_fonction.CalculateResult(Donne_Desiree[i]);
            Loss_Tampon += mlp_fonction.CostFunction(Donne_Desiree[i]);
        }
        Accuracy_Tampon = (sums / (float)(Iteration_boucle)) * 100;
        Loss_Tampon /= Iteration_boucle;
        (*Accuracy) = Accuracy_Tampon;
        (*Loss) = Loss_Tampon;
}


int main(int argc, char *argv[])
{
    if( argc < 2)
        {
        cerr << " Usage : " << argv[0] << " file.bin" << endl;
        cerr << " where : file.bin is the file where the architecture and weights of the network will be written" << endl;
        return -1;
        }

    signal (SIGINT, signal_callback_handler);
    
    int nInputUnit      = 64*64;
    int nOutputUnit     = 1;
    int nHiddenUnit     = 40;
    int nHiddenLayer    = 2;

    float learningRate  = 0.001;
    int maxEpoch        = 250;
    int nMiniBatch      = 50;

    maxEpoch_ptr = & maxEpoch;   // global variable update

    int nTrainingSet    = 800;
    int nTestSet        = 200;

    float errMinimum    = 0.001;
    
    //Allocate
    float **inputTraining		= new float*[nTrainingSet];
    float **desiredOutputTraining	= new float*[nTrainingSet];
    
    for( int i = 0; i < nTrainingSet; i++ )
        {
        inputTraining[i]		= new float[nInputUnit];
        desiredOutputTraining[i]	= new float[nOutputUnit];
        }

    float **inputTest			= new float*[nTestSet];
    float **desiredOutputTest   	= new float*[nTestSet];
    
    for(int i = 0;i < nTestSet;i++)
        {
        inputTest[i]			= new float[nInputUnit];
        desiredOutputTest[i]    	= new float[nOutputUnit];
        }

    int sums=0;
    float accuracyRate=0.F;

    //ECHO Input Array Allocation and Initialization
    ECHO echo;
    cout << " Reading directory : " << SRC_PATH+"data\\ECHO" << "  "
         <<  echo.ReadPath(SRC_PATH+"data\\ECHO") << endl;

    echo.ReadInput(0, nTrainingSet, inputTraining);
    echo.ReadLabel(0, nTrainingSet, desiredOutputTraining);
    
    echo.ReadInput(nTrainingSet, nTestSet, inputTest);
    echo.ReadLabel(nTrainingSet, nTestSet, desiredOutputTest);
    
    MLP_Network_SGD mlp('L');
    mlp.Allocate(nInputUnit,nHiddenUnit,nOutputUnit,nHiddenLayer,nTrainingSet);

    string fullname = SRC_PATH+"models\\"+argv[1];
    cout << "Will write models and weights to : \n \t" <<fullname << endl;
    size_t lastindex = fullname.find_last_of(".");
    string rawname = fullname.substr(0, lastindex);
    cout << "  Rawname for intermediate saving : \n \t" << rawname << endl;
    string best_filename;

// LOG and GRAPH
    string sgd_log_filename = rawname + ".dat";
    sgd_log_filename_ptr = &sgd_log_filename;  // update global variable
    cout << "  Filename for log : \n \t" << sgd_log_filename << endl;
    fstream sgd_log;
    sgd_log.open(sgd_log_filename.c_str(),fstream::out | fstream::trunc);// ouverture en écriture
    sgd_log << "epoch \t train_{err} \t test_{err} \t lr \t train_{acc} \t  test_{acc} \t duration(s)" << endl;

// THREAD FOR REAL TIME UPDATE
    string gnuplot_graph_acc = SRC_PATH+"models\\graph_acc_sgd.plt";
    GenerateGraphAcc(sgd_log_filename, gnuplot_graph_acc, maxEpoch);
    string system_call_string(GNUPLOT_PATH+" "+gnuplot_graph_acc);
    std::thread gnuplot_thread_acc( system, system_call_string.c_str() );

    string gnuplot_graph_loss = SRC_PATH+"models\\graph_loss_sgd.plt";
    GenerateGraphLoss(sgd_log_filename, gnuplot_graph_loss, maxEpoch);
    system_call_string = GNUPLOT_PATH+" "+gnuplot_graph_loss;
    std::thread gnuplot_thread_loss( system, system_call_string.c_str() );

// Permutation vector
    std::vector<unsigned int> indexes(nTrainingSet);
    for(unsigned int i=0; i<indexes.size(); i++) indexes[i]=i;


    //Start clock
    clock_t start, finish;
    double elapsed_time;
    start = clock();
    float local_time = 0;
    
    float initialLR = learningRate;

    float maxTestAccuracyRate = 0;
    int epoch = 0;
    while (epoch < maxEpoch)
        {
        tic();
        std::srand ( unsigned ( std::time(0) ) );
        std::random_shuffle(indexes.begin(), indexes.end());
        ProgressBar('R');

        float sumError=0;
        int batchCount=0;
        for (int i = 0; i < nTrainingSet; i++)
            {
            if( (i%100) == 0 ) local_time = duration_from_tic();
            if( (i%10) == 0  )
                {
                ProgressBar('P', 1.0*i / nTrainingSet, string(" - " + std::to_string(local_time)).c_str(), 60);
                }

            mlp.ForwardPropagateNetwork(  inputTraining[indexes[i]]         );
            mlp.BackwardPropagateNetwork( desiredOutputTraining[indexes[i]] );
            sumError += mlp.CostFunction( desiredOutputTraining[indexes[i]] );
            
            if( ((batchCount+1) % nMiniBatch) == 0)
                {
                mlp.UpdateWeight(learningRate);
                batchCount=0;
                }

            batchCount++;
            }
        ProgressBar('C');

        sumError /= nTrainingSet;


//TEST ACCURACY and LOSS (TO BE REPLACED BY A FUNCTION)

        float LossTest=0.F;
        float AccuracyTest=0.F;

        ACCURACYandLOSS(nTestSet, inputTest, desiredOutputTest, mlp, &AccuracyTest, &LossTest);

//TRAIN ACCURACY and LOSS (TO BE REPLACED BY A FUNCTION)

        float LossTrain=0.F;
        float AccuracyTrain=0.F;

        ACCURACYandLOSS(nTrainingSet, inputTraining, desiredOutputTraining, mlp, &AccuracyTrain, &LossTrain);

// MARK CHRONO
        tac();
        std::cout << std::fixed << std::setprecision(3);

        if(epoch%30 == 0)
            cout << " epoch | Loss_tn | Loss_tt |     lr     |  ACC_tn  |  ACC_tt  |  duration(s)" << endl;

        cout << std::setw(6) << epoch << " | " << std::setw(7) << LossTrain << " | "<< std::setw(7) << LossTest
             << std::scientific << " | " << std::setw(8) << learningRate
             << std::fixed << " | " << std::setw(8) << AccuracyTrain << " | " << std::setw(8) << AccuracyTest
             << " | " << std::setw(8) << duration();
        sgd_log << epoch<< "\t" <<LossTrain<< "\t" << LossTest << "\t" << learningRate
                 << "\t" << AccuracyTrain << "\t" << AccuracyTest << "\t" << duration() << endl;

// SAVE IF CURRENT IS THE BEST
        if(AccuracyTest > maxTestAccuracyRate)
            {
            maxTestAccuracyRate = AccuracyTest;
            best_filename = rawname + "_" + to_string(epoch) + "_" + to_string(AccuracyTest) + ".bin";
            std::ofstream os(best_filename, std::ofstream::binary);
            os<< mlp;
            os.close();
            cout << " * " << endl;
            }
        else cout << endl;

// EARLY STOP IF NO SIGNIFICANT CHANGE
                if (sumError < errMinimum)
                    break;

        learningRate = initialLR/(1+epoch*learningRate);    // learning rate progressive decay
        ++epoch;
    }

    //Finish clock
    finish = clock();
    elapsed_time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"time: "<<elapsed_time<<" sec"<<endl;
    cout << " Best model : " << best_filename << endl;

    sgd_log.close();

    //saving best model
    MLP_Network_SGD mlp_best;
    std::ifstream is (best_filename, std::ifstream::binary);
    is >> mlp_best;
    is.close();

    std::ofstream os_f(SRC_PATH+"models\\"+argv[1], std::ofstream::binary);
    os_f << mlp_best;
    os_f.close();

    
    //TRAIN ACCURACY and LOSS (TO BE REPLACED BY A FUNCTION)

    // Test Set Result
    cout<<"[Result]"<<endl<<endl;

    accuracyRate=0.F;
    float useless_loss = 0;

    ACCURACYandLOSS(nTrainingSet, inputTraining, desiredOutputTraining, mlp_best, &accuracyRate, &useless_loss);
    
    cout << "[Training Set]\t"<<"Accuracy Rate: " << accuracyRate << " %"<<endl;
    
    //TEST ACCURACY and LOSS (TO BE REPLACED BY A FUNCTION)
    // Test Set Result

    accuracyRate=0.F;
    useless_loss = 0;

    ACCURACYandLOSS(nTestSet, inputTest, desiredOutputTest, mlp_best, &accuracyRate, &useless_loss);
    
    cout << "[Test Set]\t"<<"Accuracy Rate: " << accuracyRate << " %"<<endl;
    


    
    for( int i=0; i<nTrainingSet; i++)
        {
        delete [] desiredOutputTraining[i];
        delete [] inputTraining[i];
        }

    for( int i=0; i<nTestSet; i++)
        {
        delete [] desiredOutputTest[i];
        delete [] inputTest[i];
        }

    delete[] inputTraining;
    delete[] desiredOutputTraining;
    delete[] inputTest;
    delete[] desiredOutputTest;

    signal_callback_handler(0);

    return 0;
}


void signal_callback_handler(int signum)
{
   cout << "Caught signal " << signum << endl;

   //system("killall gnuplot"); // will work only on linux

   string gnuplot_graph_acc = SRC_PATH+"models\\graph_acc_adam.plt";
   GenerateGraphAcc(*sgd_log_filename_ptr, gnuplot_graph_acc, *maxEpoch_ptr, false);
   string system_call_string(GNUPLOT_PATH+" "+gnuplot_graph_acc);
   std::thread gnuplot_thread_acc( system, system_call_string.c_str() );

   string gnuplot_graph_loss = SRC_PATH+"models\\graph_loss_adam.plt";
   GenerateGraphLoss(*sgd_log_filename_ptr, gnuplot_graph_loss, *maxEpoch_ptr, false);
   system_call_string = GNUPLOT_PATH+" "+gnuplot_graph_loss;
   std::thread gnuplot_thread_loss( system, system_call_string.c_str() );

   exit(signum);
}
