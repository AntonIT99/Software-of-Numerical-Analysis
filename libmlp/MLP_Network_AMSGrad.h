///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef MLP_Network_AMSGrad_H
#define MLP_Network_AMSGrad_H
#include "MLP_Layer_AMSGrad.h"


#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

/*! Classe de gestion du réseau Multi Layer Perceptron

*/
class MLP_Network_AMSGrad {
    
private:
	//! Tableau des différentes couches (layer)
    MLP_Layer_AMSGrad *layerNetwork;
    
	//! Nombre de données pour l'entrainement
    int nTrainingSet;
	//! nombre de neurones d'entrée
    int nInputNeurons;
	//! nombre de neurones dans les couches cachées
    int nHiddenNeurons;
	//! nombre de neurones de sortie
    int nOutputNeurons;
	//! nombre de couches cachées
    int nHiddenLayer;

public:
    double Beta1;
    double Beta2;

    char Activation;

	//! constructeur, initialisation des champs
    MLP_Network_AMSGrad(double beta1=0.9, double beta2=0.99, char activation='L'):
        nTrainingSet(-1),nInputNeurons(-1),nHiddenNeurons(-1),nOutputNeurons(-1),nHiddenLayer(-1), Beta1(beta1), Beta2(beta2), Activation(activation)
        {layerNetwork = nullptr;}
    //! destructeur : libération de la mémoire
    ~MLP_Network_AMSGrad(){Delete();}

	//! fonction d'allocation du réseau
    void Allocate(int nInputNeurons,   int nHiddenNeurons, int nOutputNeurons, int nHiddenLayer,
                  int nTrainingSet);
    //! fonction de désallocation du réseau
	void Delete();
    
	//! fonction de calcul de la passe avant à partir de la couche \param inputNetwork
    void ForwardPropagateNetwork(float* inputNetwork);
	//! fonction de calcul de la passe arrière, les labels sont passés à la fonction par \param desiredOutput
    void BackwardPropagateNetwork(float* desiredOutput);
	//! fonction de mise à jour des poids du réseau, coefficienté par le taux d'apprentissage \param learningRate
    void UpdateWeight(float learningRate);
	//! Calcul de la fonction de cout
    float CostFunction(float* desiredOutput);
    //! Calcul de la précision
    float CalculateResult(float* desiredOutput);

	//! fonction pour accéder aux couches du réseau
    const MLP_Layer_AMSGrad * GetLayerNetwork() const {return this->layerNetwork;}

    //! fonction qui retourne la taille du jeu d'entrainement
    int GetnTrainingSet() const { return this->nTrainingSet; }
    //! fonction qui retourne le nombre de neurones de la couche d'entrée (= premiere couche cachée)
    int GetnInputUnit() const { return this->nInputNeurons; }
    //! fonction qui retourne le nombre de neurones dans les couches cachées
    int GetnHiddenUnit() const { return this->nHiddenNeurons; }
    //! fonction qui retourne le nombre de neurones dans la couche de sortie
    int GetnOutputUnit() const { return this->nOutputNeurons; }
    //! fonction qui retourne le nombre de couches cachées
    int GetnHiddenLayer() const { return this->nHiddenLayer; }

	//! fonction de chargement d'un réseau à partir d'un fichier
    friend ifstream& operator>>(ifstream& is, MLP_Network_AMSGrad& mlp);
	//! fonction de sauvegarde du réseau dans un fichier
    friend ofstream& operator<<(ofstream& os, const MLP_Network_AMSGrad& mlp);
};
#endif
