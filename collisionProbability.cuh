/*
collisionProbability.cuh
author: Brian Ichter

This code is a fully hard coded version of MC. 
Knowing the dynamics and the double integator, the Ncomb and Acomb matrices
are the same and then we just randomly pull from a distribution to form
xcomb. Thus, other than an unknown final path and time, we can precompute
our variations from the nominal path.

Ideally I will change this to calculate on the fly, but given memory and speed
requirements, this may actually be ideal.
*/

#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <ctime>
#include <cstdlib>

#include "collisionCheck.cuh"
#include "helper.cuh"
#include "hardCoded.cuh"

// const int T = 43; // number of time steps
// const int numMCParticles = 32; // will be 1024
float collisionProbability(float *d_obstacles, int obstaclesCount, float *d_xcomb, float offsetMult, float *d_path, int T);

__global__ void MCCP(int *isCollision, float *path, float *xcomb, float offsetMult, 
	float *obstacles, int obstaclesCount, int T, float *debugOutput);