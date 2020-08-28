// T. Grenier - Clanu 2017-2018-2019-2020 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef _timing_functions_h
#define _timing_functions_h


#include <ctime>
#include <ratio>
#include <chrono>


static std::chrono::steady_clock::time_point tic_time;
static std::chrono::steady_clock::time_point tac_time;

//! Start the chronometer.
void tic();
//! Stop the chronometer.
void tac();
//! Return time in seconds elapsed between last tic() and tac() calls.
double duration();
//! Return time in seconds elapsed between last tic().
double duration_from_tic();
//! Return time in seconds elapsed between last tac()
double duration_from_tac();

#endif 
