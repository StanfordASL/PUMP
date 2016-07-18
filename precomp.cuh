/*
precomp.cuh
author: Brian Ichter

Precomputation methods are contained. 
*/
#pragma once

#include <math.h>

// find the maximum number of nearest neighbors to set the array size
__global__ void calculateNumberNN(float r2, float *samples, int *nnSizes, int samplesCount);
__global__ void precompNN(float r2, float *samples, float *distances, int nnSize, int *nnEdges, int samplesCount);
__global__ void setupPrecompNN(float *distances, int nnSize, int *nnEdges, int samplesCount);