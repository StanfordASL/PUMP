#include "precomp.cuh"

__global__
void calculateNumberNN(float r2, float *samples, int *nnSizes, int samplesCount)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount) 
		return;

	// load up the node's location
	float nodeLoc[DIM];
	for (int d = 0; d < DIM; ++d)
		nodeLoc[d] = samples[node*DIM+d];
	
	int nnSize = 0;
	for (int sample = 0; sample < samplesCount; ++sample) {
		if (sample == node)
			continue;
		float distance2 = 0;
		for (int d = 0; d < DIM; ++d) {
			float difference = nodeLoc[d] - samples[sample*DIM+d];
			distance2 += difference * difference;
		}
		if (distance2 < r2) {
			++nnSize;
		}
	}
	nnSizes[node] = nnSize;
}

__global__ 
void setupPrecompNN(float *distances, int nnSize, int *nnEdges, int samplesCount)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount) 
		return;

	for (int nn = 0; nn < nnSize; ++nn) {
		distances[node*nnSize + nn] = 0;
		nnEdges[node*nnSize + nn] = -1;
	}
}

__global__ 
void precompNN(float r2, float *samples, float *distances, int nnSize, int *nnEdges, int samplesCount)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount) 
		return;

	// load up the node's location
	float nodeLoc[DIM];
	for (int d = 0; d < DIM; ++d) {
		nodeLoc[d] = samples[node*DIM+d];
	}

	// find if part of next wavefront and cost to come
	int nnIdx = 0;
	for (int sample = 0; sample < samplesCount; ++sample) {
		if (sample == node)
			continue;
		float distance2 = 0;
		for (int d = 0; d < DIM; ++d) {
			float difference = nodeLoc[d] - samples[sample*DIM+d];
			distance2 += difference * difference;
		}
		if (distance2 < r2) {
			nnEdges[node*nnSize + nnIdx] = sample;
			distances[node*nnSize + nnIdx] = sqrt(distance2);
			++nnIdx;
		}
	}
}