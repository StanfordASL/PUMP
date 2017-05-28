/*
PUMP.cuh
author: Brian Ichter

Parallel Uncertainty-aware Multiobjective Planning algorithm
*/
#pragma once

#include <math.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <assert.h>
#include <fstream>
#include <limits>
#include <utility>
#include <algorithm>
#include "cuda.h"
#include "cuda_runtime.h"

#include "helper.cuh"
#include "2pBVP.cuh"
#include "collisionProbability.cuh"
#include "motionPlanningProblem.cuh"

// A macro for checking the error codes of cuda runtime calls
#define CUDA_ERROR_CHECK(expr) \
  {                            \
    cudaError_t err = expr;    \
    if (err != cudaSuccess)    \
    {                          \
      printf("CUDA call failed!\n\t%s\n", cudaGetErrorString(err)); \
      exit(1);                 \
    }                          \
  }

/***********************
CPU functions
***********************/
void PUMP(MotionPlanningProblem mpp);

/***********************
GPU kernels
***********************/
__global__ void calculateCP_HSMC(int numNewPaths, int pathBaseIdx, float *pathCP, bool *validParticles, int numHSMCParticles);
__global__ void dominanceCheck();
__global__ void propagateValidParticles(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, int *countStoredHSs, float *times, float *topts, float dt,
	int maxHSCount, float *offsets, float offsetMult, int numMCParticles, bool *validParticles,
	int numHSMCParticles, float *debugOutput);
__global__ void propagateValidParticlesMany(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, int *countStoredHSs, float *times, float *topts, float dt,
	int maxHSCount, float *offsets, float offsetMult, int numMCParticles, bool *validParticles,
	int numHSMCParticles, int numThreadParticles, float *debugOutput);
__global__ void propagateValidParticlesPathCentric(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, float *times, float *topts, float dt,
	int maxHSCount, float *offsets, int numMCParticles, bool *validParticles,
	int numHSMCParticles, float *debugOutput);
__global__ void setupValidParticles(bool *validParticles, int numHSMCParticles, int maxPathCount,
	float *pathCP, float *pathTime);