/*
CCGMT.cuh
author: Brian Ichter

Chance Constrained Group Marching Tree algorithm
1, 3
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
void CCGMT(float rn, float cpTarget);
void CCGMTcpu(float rn, float cpTarget, int initIdx, int goalIdx, float dt, int numMCSamples, 
	float *h, int *nnGoSizes, int *nnGoEdges, int *nnIdxs, int maxNNSize, bool *isFreeSamples, bool *isFreeEdges,
	int numDisc, int obstaclesCount, float *d_topts, float *topts, float *d_copts, float *copts, 
	float *d_as, float *d_bs, int *d_countStoredHSs, float *d_offsets, float *d_offsetsTime, float offsetMult,
	std::vector<int> P, int *Pcounts, int *pathPrev, int *pathNode, float *pathCost, float *pathTime, float *pathCP, 
	int maxPathCount, std::vector<int> G, int *sizeG, bool *d_pathValidParticles, float *d_pathCP, float *d_pathTime,
	int *wavefrontPathPrev, int *wavefrontNodeNext, int *wavefrontEdge,
	int *d_wavefrontPathPrev, int *d_wavefrontNodeNext, int *d_wavefrontEdge,
	float epsCost, float epsCP, float lambda, int numBuckets, float cpFactor, int numHSMCParticles,
	float *samples, float *obstacles, int Tmax);

/***********************
GPU kernels
***********************/
__global__ void calculateCP_HSMC(int numNewPaths, int pathBaseIdx, float *pathCP, bool *validParticles, int numHSMCParticles);
__global__ void dominanceCheck();
__global__ void propagateValidParticles(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, int *countStoredHSs, float *times, float *topts, float dt,
	int obstaclesCount, float *offsets, float offsetMult, int numMCSamples, bool *validParticles,
	int numHSMCParticles, float *debugOutput);
__global__ void propagateValidParticlesMany(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, int *countStoredHSs, float *times, float *topts, float dt,
	int obstaclesCount, float *offsets, float offsetMult, int numMCSamples, bool *validParticles,
	int numHSMCParticles, int numThreadParticles, float *debugOutput);
__global__ void propagateValidParticlesPathCentric(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, float *times, float *topts, float dt,
	int obstaclesCount, float *offsets, int numMCSamples, bool *validParticles,
	int numHSMCParticles, float *debugOutput);
__global__ void setupValidParticles(bool *validParticles, int numHSMCParticles, int maxPathCount,
	float *pathCP, float *pathTime);