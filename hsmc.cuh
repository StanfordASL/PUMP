/*
hsmc.cuh
author: Brian Ichter

Contains functions related to the calculation of half-spaces and CP estimation
for the Half-space Monte Carlo method.
*/

#pragma once

#include "bvls.cuh"

typedef struct HSMC {
	// as and bs
	float *d_as;
	float *d_bs; 
	int *d_countStoredHSs; 
	int maxHSCount;

	// offsets
	float *d_offsets; 
	float *d_offsetsTime; 
	float offsetMult;

	bool* d_pathValidParticles;
} HSMC;

// compute and cache only the half-spaces closer than maxd2
__global__ void cacheHalfspaces(int numEdges, float *discMotions, 
	bool *isFreeEdges,
	int numObstacles, int maxHSCount, float *obstacles, 
	int waypointsCount, float *as, float *bs, int *countStoredHSs, 
	float *sigma, float maxd2);
// compute and cache only the half-spaces closer than maxd2, called on numEdges threads
__global__ void cacheHalfspacesEdge(int numEdges, float *discMotions, 
	bool *isFreeEdges, bool *isFreeSamples,
	int numObstacles, int maxHSCount, float *obstacles, 
	int waypointsCount, float *as, float *bs, int *countStoredHSs, 
	float *sigma, float maxd2);
// compute and cache the half-spaces for an edge
__global__ void setupAsBs(int halfspaceCount, float *bs, float *as);
// find half-spaces
__device__ void calculateHalfspace();
// recalculate and propagate valid particles
__device__ void propagateCP();
// calculate the new CP
__device__ void calculateCP();