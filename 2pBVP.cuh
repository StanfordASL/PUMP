/*
2pBVP.cuh
author: Brian Ichter

This file solves the 3D double integrator two point boundary value problem
*/

#pragma once

#include <math.h> 
#include "helper.cuh"

// find optimal path
float findOptimalPath(float dt, float *splitPath, float *xs, int numWaypoints, int *pathLength);
void discretizePath(float *splitPath, float *x0, float *x1, int numDisc, float topt);
float findDiscretizedPath(float *splitPath, float *x0, float *x1, int numDisc);
// bisection search to find tau for optimal cost
__host__ __device__ float toptBisection(float *x0, float *x1, float tmax);
// cost at tau
__host__ __device__ float cost(float tau, float *x0, float *x1);
// deriviative of the cost at tau
__host__ __device__ float dcost(float tau, float *x0, float *x1);
// put a new point in the path at x
void pathPoint(float t, float tau, float *x0, float *x1, float *x);

__global__ void fillCoptsTopts(float *samples, float *copts, float *topts, float tmax);