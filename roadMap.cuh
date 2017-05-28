/*
roadMap.cuh
Author: Brian Ichter

Struct definition for a motion planning problem.
*/

#pragma once

#include <iostream>
#include <vector>

typedef struct RoadMap {
	// samples
	int numSamples; // number of samples
	std::vector<float> samples; // samples[i*DIM + d] is sample i, dimension d 
	float *d_samples; // size NUM*DIM

	// graph construction
	std::vector<float> h; // heuristic for point to goal, size NUM
	std::vector<float> toptsEdge;
	float *d_toptsEdge;
	std::vector<float> coptsEdge;
	float *d_coptsEdge;

	// nn's
	int nnGoSizes[NUM];
	std::vector<int> nnGoEdges;
	std::vector<int> nnIdxs;
	int *d_nnIdxs;
	int maxNNSize;
	bool isFreeSamples[NUM];
	std::vector<bool> isFreeEdges;
} RoadMap;

// searching, wavefronts, paths, etc
typedef struct PUMPSearch {
	std::vector<int> P; // mpp.pumpSearch.P[i*NUM + j] index of path that exists at node i
	int Pcounts[NUM]; // number of paths
	int *pathPrev;
	int *pathNode;
	float *pathCost;
	float *pathTime;
	float *pathCP;

	int maxPathCount;
	std::vector<int> G;
	int *sizeG;
	bool *d_pathValidParticles;
	float *d_pathCP;
	float *d_pathTime;
	int *wavefrontPathPrev;
	int *wavefrontNodeNext;
	int *wavefrontEdge;
	int *d_wavefrontPathPrev;
	int *d_wavefrontNodeNext;
	int *d_wavefrontEdge;
} PUMPSearch;
