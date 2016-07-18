#include "GMT.cuh"

void GMT(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	float *d_distancesCome, int *d_nnGoEdges, int *d_nnComeEdges, int nnSize, float *d_samples, int samplesCount,
	float lambda, float r, float *d_costs, int *d_edges, int initial_idx, int goal_idx)
{

}

void GMTwavefront(float *initial, float *goal, float *d_obstacles, int obstaclesCount,
	float *d_distancesCome, int *d_nnGoEdges, int *d_nnComeEdges, int nnSize, float *d_discMotions, int *d_nnIdxs,
	float *d_samples, int samplesCount, bool *d_isFreeSamples, float r, int numDisc,
	float *d_costs, int *d_edges, int initial_idx, int goal_idx)
{
	int maxItrs = 100;
	
	bool *d_wavefrontWas, *d_unvisited, *d_isCollision;
	float *d_costGoal;
	thrust::device_vector<float> d_debugOutput(samplesCount*numDisc);
	thrust::device_vector<bool> d_wavefront(samplesCount);
	thrust::device_vector<bool> d_wavefrontNew(samplesCount);
	thrust::device_vector<int> d_wavefrontScanIdx(samplesCount);
	thrust::device_vector<int> d_wavefrontActiveIdx(samplesCount);

	float *d_debugOutput_ptr = thrust::raw_pointer_cast(d_debugOutput.data());
	bool *d_wavefront_ptr = thrust::raw_pointer_cast(d_wavefront.data());
	bool *d_wavefrontNew_ptr = thrust::raw_pointer_cast(d_wavefrontNew.data());
	int *d_wavefrontScanIdx_ptr = thrust::raw_pointer_cast(d_wavefrontScanIdx.data());
	int *d_wavefrontActiveIdx_ptr = thrust::raw_pointer_cast(d_wavefrontActiveIdx.data());

	cudaMalloc(&d_wavefrontWas, sizeof(bool)*samplesCount);
	cudaMalloc(&d_unvisited, sizeof(bool)*samplesCount);
	cudaMalloc(&d_costGoal, sizeof(float));
	cudaMalloc(&d_isCollision, sizeof(bool)*numDisc*samplesCount);

	if (d_unvisited == NULL) {
		std::cout << "Allocation Failure" << std::endl;
		exit(1);
	}

	// setup array values
	const int blockSize = 128;
	const int gridSize = std::min((samplesCount + blockSize - 1) / blockSize, 65535);
	setupArrays<<<gridSize, blockSize>>>(d_wavefront_ptr, d_wavefrontNew_ptr, 
		d_wavefrontWas, d_unvisited, d_isFreeSamples, d_costGoal, d_costs, d_edges, samplesCount);
	
	float costGoal = 0;
	int itrs = 0;
	int activeSize = 0;
	while (itrs < maxItrs && costGoal == 0)
	{ 
		++itrs;

		// prefix sum to fill wavefrontScanIdx to find points in the current band of the wavefront
		thrust::exclusive_scan(d_wavefront.begin(), d_wavefront.end(), d_wavefrontScanIdx.begin());
		fillWavefront<<<gridSize, blockSize>>>(samplesCount,
			d_wavefrontActiveIdx_ptr, d_wavefrontScanIdx_ptr, d_wavefront_ptr);

		activeSize = d_wavefrontScanIdx[samplesCount-1];
		// increase by one if the last point in array is true because exclusive scan doesnt increment otherwise
		(d_wavefront[d_wavefront.size() - 1]) ? ++activeSize : 0;

		std::cout << "new DIM = " << DIM << " Itr " << itrs << " size " << activeSize << std::endl;

		const int blockSizeActive = 128;
		const int gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 65535);
		
		findWavefront<<<gridSizeActive, blockSizeActive>>>(
			d_unvisited, d_wavefront_ptr, d_wavefrontNew_ptr, d_wavefrontWas, 
			d_nnGoEdges, nnSize, activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);

		// prefix sum to fill wavefrontScanIdx to connect to the next wavefront
		thrust::exclusive_scan(d_wavefrontNew.begin(), d_wavefrontNew.end(), d_wavefrontScanIdx.begin());
		
		// to print thrust vector) thrust::copy(d_wavefrontScanIdx.begin(), d_wavefrontScanIdx.end(), std::ostream_iterator<int>(std::cout, " "));

		fillWavefront<<<gridSize, blockSize>>>(
			samplesCount, d_wavefrontActiveIdx_ptr,
			d_wavefrontScanIdx_ptr, d_wavefrontNew_ptr);

		activeSize = d_wavefrontScanIdx[samplesCount-1]; 
		// increase by one if the last point in array is true because exclusive scan doesnt increment otherwise
		(d_wavefrontNew[d_wavefrontNew.size() - 1]) ?  ++activeSize : 0;
		if (activeSize == 0) // the next wavefront is empty (only valid for GMTwavefront)
			break;

		std::cout << "exp DIM = " << DIM << " Itr " << itrs << " size " << activeSize << std::endl;

		const int blockSizeActiveExp = 128;
		const int gridSizeActiveExp = std::min((activeSize + blockSizeActiveExp - 1) / blockSizeActiveExp, 65535);
		findOptimalConnection<<<gridSizeActiveExp, blockSizeActiveExp>>>(
			d_wavefront_ptr, d_edges, d_costs, d_distancesCome, 
			d_nnComeEdges, nnSize, activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);
		const int blockSizeActiveVerify = 128;
		const int gridSizeActiveVerify = std::min((activeSize*numDisc + blockSizeActiveVerify - 1) / blockSizeActiveVerify, 65535);
		verifyExpansion<<<gridSizeActiveVerify, blockSizeActiveVerify>>>(
			obstaclesCount, d_wavefrontNew_ptr, d_edges,
			d_samples, d_obstacles, d_costs, 
			d_nnIdxs, d_discMotions, numDisc, d_isCollision,
			activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);
		removeInvalidExpansions<<<gridSizeActiveExp, blockSizeActiveExp>>>(
			obstaclesCount, d_wavefrontNew_ptr, d_edges,
			d_samples, d_obstacles, d_costs, 
			d_nnIdxs, d_discMotions, numDisc, d_isCollision,
			activeSize, d_wavefrontActiveIdx_ptr, d_debugOutput_ptr);
		// std::cout << "debug output: " << std::endl;
		// thrust::copy(d_debugOutput.begin(), d_debugOutput.begin()+100, std::ostream_iterator<float>(std::cout, " "));
		updateWavefront<<<gridSize, blockSize>>>(
			samplesCount, goal_idx,
			d_unvisited, d_wavefront_ptr, d_wavefrontNew_ptr, d_wavefrontWas,
			d_costs, d_costGoal, d_debugOutput_ptr);

		// copy over the goal cost (if non zero, then a solution was found)
		cudaMemcpy(&costGoal, d_costGoal, sizeof(float), cudaMemcpyDeviceToHost);
	}
		
	// free arrays
	cudaFree(d_wavefrontWas);
	cudaFree(d_unvisited);
	cudaFree(d_costGoal);
	cudaFree(d_isCollision);
}

__global__ void setupArrays(bool* wavefront, bool* wavefrontNew, bool* wavefrontWas, bool* unvisited, 
	bool *isFreeSamples, float *costGoal, float* costs, int* edges, int samplesCount)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	if (node >= samplesCount) 
		return;

	unvisited[node] = isFreeSamples[node];
	wavefrontNew[node] = false;
	wavefrontWas[node] = !isFreeSamples[node];
	if (node == 0) {		
		wavefront[node] = true;
		*costGoal = 0;
	} else {
		wavefront[node] = false;
	}
	costs[node] = 0;
	edges[node] = -1;

	if (!isFreeSamples[node]) {
		costs[node] = -11;
		edges[node] = -2;
	}
}

__global__
void findWavefront(bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas,
	int *nnGoEdges, int nnSize, int wavefrontSize, int* wavefrontActiveIdx, float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;

	int node = wavefrontActiveIdx[tid];

	if (wavefront[node])
	{
		wavefrontWas[node] = true;
		for (int i = 0; i < nnSize; ++i) {
			int nnIdx = nnGoEdges[node*nnSize + i];
			if (nnIdx == -1)
				return;
			if (unvisited[nnIdx] && !wavefront[nnIdx]) {
				wavefrontNew[nnIdx] = true;
			}
		}
	}
}

__global__
void fillWavefront(int samplesCount, int *wavefrontActiveIdx, int *wavefrontScanIdx, bool* wavefront)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	if (node >= samplesCount)
		return;
	if (!wavefront[node])
		return;

	wavefrontActiveIdx[wavefrontScanIdx[node]] = node;
}

__global__
void findOptimalConnection(bool *wavefront, int* edges, float* costs, float *distancesCome, 
	int *nnComeEdges, int nnSize, int wavefrontSize, int* wavefrontActiveIdx, float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;

	int node = wavefrontActiveIdx[tid];

	for (int i = 0; i < nnSize; ++i) {
		int nnIdx = nnComeEdges[node*nnSize + i];
		if (nnIdx == -1)
			break;
		if (wavefront[nnIdx]) {
			float costTmp = costs[nnIdx] + distancesCome[node*nnSize + i];
			if (costs[node] == 0 || costTmp < costs[node]) {
				costs[node] = costTmp;
				edges[node] = nnIdx;
			}
		}
	}
} 

__global__
void verifyExpansion(int obstaclesCount, bool *wavefrontNew, int* edges,
	float *samples, float* obstacles, float* costs,
	int *nnIdxs, float *discMotions, int numDisc, bool *isCollision,
	int wavefrontSize, int* wavefrontActiveIdx, float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize*numDisc)
		return;
	int nodeIdx = tid/numDisc;
	int discIdx = tid%numDisc;
	int node = wavefrontActiveIdx[nodeIdx];
	
	waypointCollisionCheck(node, edges[node], obstaclesCount, obstacles, 
			nnIdxs, discMotions, discIdx, numDisc, isCollision, tid, debugOutput);
}

__global__
void removeInvalidExpansions(int obstaclesCount, bool *wavefrontNew, int* edges,
	float *samples, float* obstacles, float* costs,
	int *nnIdxs, float *discMotions, int numDisc, bool *isCollision,
	int wavefrontSize, int* wavefrontActiveIdx, float* debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= wavefrontSize)
		return;
	int node = wavefrontActiveIdx[tid];

	bool notValid = isCollision[tid*numDisc];
	for (int i = 1; i < numDisc; ++i) 
		notValid = notValid || isCollision[tid*numDisc + i];
	if (notValid) {
		costs[node] = 0;
		edges[node] = -1;
		wavefrontNew[node] = false;
	}
}

__global__
void updateWavefront(int samplesCount, int goal_idx,
	bool *unvisited, bool *wavefront, bool *wavefrontNew, bool *wavefrontWas,
	float* costs, float* costGoal, float* debugOutput)
{
	int node = blockIdx.x * blockDim.x + threadIdx.x;

	if (node >= samplesCount)
		return;
	if (!unvisited[node])
		return;

	if (wavefrontWas[node]) {
		wavefront[node] = false; // remove from the current wavefront
		unvisited[node] = false; // remove from unvisited
	} else if (wavefrontNew[node]) {
		wavefront[node] = true; // add to wavefront
		wavefrontNew[node] = false;
		if (node == goal_idx)
			*costGoal = costs[node];
	}
}


