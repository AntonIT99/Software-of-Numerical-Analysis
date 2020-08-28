TEMPLATE = lib
TARGET = libmlp

CONFIG += staticlib c++17 warn_off
CONFIG += create_prl
CONFIG -= qt

SOURCES = MLP_Layer_SGD_LR.cpp MLP_Network_SGD_LR.cpp Echo.cpp timing_functions.cpp MLP_Layer.cpp MLP_Layer_SGD.cpp MLP_Network_SGD.cpp MLP_Layer_ADAM.cpp MLP_Network_ADAM.cpp progress_bar.cpp gnuplot_utilities.cpp
HEADERS = MLP_Layer_SGD_LR.h MLP_Network_SGD_LR.h Echo.h   timing_functions.h   MLP_Layer.h   MLP_Layer_SGD.h   MLP_Network_SGD.h   MLP_Layer_ADAM.h   MLP_Network_ADAM.h   progress_bar.h    stb_image.h gnuplot_utilities.h


LIBS += -lstdc++fs

