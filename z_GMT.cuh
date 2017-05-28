/*
z_GMT.cuh
author: Brian Ichter

A gutted version of MCMP to do a search for best paths for obstacle inflations
*/
#pragma once

#include <math.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <assert.h>
#include <fstream>
#include <limits>
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
float z_GMT(float rn, int initIdx, int goalIdx, float dt, int numMCParticles, 
	float *h, int *nnGoSizes, int *nnGoEdges, int *nnIdxs, int maxNNSize, bool *isFreeSamples, bool *isFreeEdges,
	int numDisc, int obstaclesCount, float *d_topts, float *topts, float *d_copts, float *copts, 
	float *d_offsets, float offsetMult,
	std::vector<int> P, int *Pcounts, int *pathPrev, int *pathNode, float *pathCost, float *pathTime, 
	int maxPathCount, std::vector<int> G, int *sizeG, float *d_pathTime,
	int *wavefrontPathPrev, int *wavefrontNodeNext, int *wavefrontEdge,
	int *d_wavefrontPathPrev, int *d_wavefrontNodeNext, int *d_wavefrontEdge,
	float lambda, int numBuckets,
	float *samples, float *obstacles, float *d_obstaclesInflated, int Tmax, float *splitPath, int *pathLength, float *costGoal);

/***********************
GPU kernels
***********************/
__global__ 
void isFreePath(float *obstacles, int obstaclesCount, 
	float *path, int pathLength,
	bool *collisionFree);