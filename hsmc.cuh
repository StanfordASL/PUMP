/*
hsmc.cuh
author: Brian Ichter

Contains functions related to the calculation of half-spaces and CP estimation
for the Half-space Monte Carlo method.
*/

#pragma once

#include "bvls.cuh"

// compute and cache only the half-spaces closer than maxd2
__global__ void cacheHalfspaces(int numEdges, float *discMotions, 
	bool *isFreeEdges,
	int obstaclesCount, float *obstacles, 
	int waypointsCount, float *as, float *bs, int *countStoredHSs, 
	float *sigma, float maxd2);
// compute and cache only the half-spaces closer than maxd2, called on numEdges threads
__global__ void cacheHalfspacesEdge(int numEdges, float *discMotions, 
	bool *isFreeEdges, bool *isFreeSamples,
	int obstaclesCount, float *obstacles, 
	int waypointsCount, float *as, float *bs, int *countStoredHSs, 
	float *sigma, float maxd2);
// compute and cache the half-spaces for a point
__global__ void cacheHalfspacesAll(int halfspaceCount, float *discMotions, 
	int obstaclesCount, float *obstacles, int waypointsCount, 
	float *as, float *bs, float *sigma, float maxd2);
// compute and cache the half-spaces for an edge
__global__ void cacheHalfspacesEdgeCentric(int numEdges, float *discMotions, 
	int obstaclesCount, float *obstacles, int waypointsCount, 
	float *as, float *bs, float *sigma);
__global__ void setupBs(int halfspaceCount, float *bs, float *as);
// find half-spaces
__device__ void calculateHalfspace();
// recalculate and propagate valid particles
__device__ void propagateCP();
// calculate the new CP
__device__ void calculateCP();