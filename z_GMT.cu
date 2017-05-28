#include "z_GMT.cuh"
// CPU version of GMT (see .cuh file)

// TODO: what if MCMP were all on the CPU except for the CP computation?
// now the coding is much simpler, and I can use vectors more easily

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
	float *samples, float *obstacles, float *d_obstaclesInflated, int Tmax, float *splitPath, int *pathLength, float *costGoal)
{ 
	// std::cout << "________________ Beginning Z_GMT ________________" << std::endl;
	double t_ccgmtStart = std::clock();
	float dr = lambda*rn;
	
	int numPaths = 1;

	// setup initial path
	int Gidx = 0;
	bool goalCondition = false;
	bool emptyOpenSet = false;
	P[initIdx*NUM + 0] = 0; // path 0 is at initIdx
	Pcounts[initIdx]++;
	pathPrev[0] = -2; // denote end
	pathNode[0] = initIdx; 
	pathCost[0] = 0;
	pathTime[0] = 0;
	sizeG[Gidx]++;
	G[0] = 0;

	float costThreshold = h[initIdx]; 
	int maxItrs = 20;
	int itrs = 0;

	// *************************** exploration loop ***************************
	while (itrs < maxItrs && !goalCondition && !emptyOpenSet) { // cutoff at solution exists with cp = cpMinSoln or expansion is empty
		++itrs;
		// std::cout << "************** starting iteration " << itrs << " with " << sizeG[Gidx] << " paths" << std::endl;

		int numNewPaths = 0;
		for (int g = 0; g < sizeG[Gidx]; ++g) {
			int pathIdxPrev = G[Gidx*maxPathCount + g]; // path to expand from
			G[Gidx*maxPathCount + g] = -1; // clear out this g
			int nodeIdxPrev = pathNode[pathIdxPrev];
			for (int nn = 0; nn < nnGoSizes[nodeIdxPrev]; ++nn) {
				int nodeIdxNext = nnGoEdges[nodeIdxPrev*maxNNSize + nn]; // node to expand to
				int edgeIdx = nnIdxs[nodeIdxNext*NUM + nodeIdxPrev]; // edge index connecting prev to next
				// check if edge is collision free and the sample is free
				if (!isFreeEdges[edgeIdx] || !isFreeSamples[nodeIdxNext]) 
					continue;

				wavefrontPathPrev[numNewPaths] = pathIdxPrev;
				wavefrontNodeNext[numNewPaths] = nodeIdxNext;
				wavefrontEdge[numNewPaths] = edgeIdx;
				numNewPaths++;
				if (numNewPaths > maxPathCount) {
					return -1;
				}
			}
		}

		if (numPaths + numNewPaths >= maxPathCount) {
			std::cout << "maxPathCount reached, increase max number of paths" << std::endl;
			return -1;
		}
		
		sizeG[Gidx] = 0; // reset G size

		// copy necessary info to GPU
		cudaDeviceSynchronize();
		CUDA_ERROR_CHECK(cudaMemcpy(d_wavefrontPathPrev, wavefrontPathPrev, sizeof(int)*numNewPaths, cudaMemcpyHostToDevice));
		CUDA_ERROR_CHECK(cudaMemcpy(d_wavefrontEdge, wavefrontEdge, sizeof(int)*numNewPaths, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		// calculate CP (do half plane checks per particle, then sum)
		// copy over path times

		for (int i = 0; i < numNewPaths; ++i) {
			pathTime[numPaths + i] = pathTime[wavefrontPathPrev[i]] + topts[wavefrontEdge[i]];
			pathCost[numPaths + i] = pathCost[wavefrontPathPrev[i]] + copts[wavefrontEdge[i]]; // cost = time currently
		}

		// ************************************** dominance check ************************************** 
		// load all new nodes into P
		int PnewCount[NUM];
		for (int i = 0; i < NUM; ++i)
			PnewCount[i] = 0;
		for (int i = 0; i < numNewPaths; ++i) {
			int nodeIdx = wavefrontNodeNext[i];
			int pathIdx = numPaths + i;

			P[nodeIdx*NUM + Pcounts[nodeIdx] + PnewCount[nodeIdx]] = pathIdx;
			PnewCount[nodeIdx]++;
		}

		// check new paths against stored paths
		for (int i = 0; i < numNewPaths; ++i) {
			int nodeIdx = wavefrontNodeNext[i];
			// already eliminated or at the goal idx
			if (wavefrontNodeNext[i] == -1) // || nodeIdx == goalIdx)
				continue;
			int pathIdx = numPaths + i;

			for (int j = 0; j < Pcounts[nodeIdx] + PnewCount[nodeIdx]; ++j) {
				int pathIdxCompare = P[nodeIdx*NUM + j];
				// don't compare to self
				if (pathIdxCompare == pathIdx)
					continue;

				// comparison
				if (pathCost[pathIdxCompare] < pathCost[pathIdx]) {
					
					// check if paths are co-dominant, then keep the one with a lower path number
					if (pathCost[pathIdxCompare] >= pathCost[pathIdx] &&
						pathIdx < pathIdxCompare) {
						continue;
					}

					wavefrontNodeNext[i] = -1; // mark for removal
					break; 
				}
			}
		}

		// ************************************** store good paths ************************************** 
		int numValidNewPaths = 0;
		for (int i = 0; i < numNewPaths; ++i) {
			int nodeIdx = wavefrontNodeNext[i];
			int pathIdx = numPaths + i;

			// TODO: if i break here, decrement path index and path count
			if (wavefrontNodeNext[i] == -1 || wavefrontNodeNext[i] == 0) { // either node is at the init or marked as bad
				wavefrontNodeNext[i] = -1; // clear
				continue;
			}

			int pathIdxStore = numPaths + numValidNewPaths;
			pathTime[pathIdxStore] = pathTime[pathIdx];
			pathCost[pathIdxStore] = pathCost[pathIdx];

			float dCost = (pathCost[pathIdxStore] + h[nodeIdx] - costThreshold);
			dCost = std::max((float) 0, dCost); // if below zero, put into the next bucket
			int bucketIdx = (((int) (dCost / dr)) + Gidx + 1) % numBuckets;
			G[bucketIdx*maxPathCount + sizeG[bucketIdx]] = pathIdxStore;
			sizeG[bucketIdx]++;
			pathPrev[pathIdxStore] = wavefrontPathPrev[i];
			pathNode[pathIdxStore] = wavefrontNodeNext[i];

			P[nodeIdx*NUM + Pcounts[nodeIdx]] = pathIdxStore;
			Pcounts[nodeIdx]++;
			wavefrontNodeNext[i] = -1; // clear, TODO: uncessary, but nice for debugging purposes to have a fresh array

			numValidNewPaths++;
		}
		numPaths += numValidNewPaths;

		// update goal condition
		if (Pcounts[goalIdx] > 0)
			goalCondition = true;
		if (goalCondition) {
			break;
		}

		// update empty open set condition
		emptyOpenSet = true;
		for (int b = 0; b < numBuckets; ++b)
			emptyOpenSet = emptyOpenSet && (sizeG[b] == 0);
		if (emptyOpenSet) {
			std::cout << "emptyOpenSet met" << std::endl;
			break;
		}

		// update G index
		Gidx = (Gidx+1) % numBuckets;
		costThreshold += dr;

		// end and send out warning if maxPathCount is exceeded
		if (numPaths >= maxPathCount) {
			std::cout << "maxPathCount reached, increase max number of paths" << std::endl;
			return -1;
		}
		
	}

	// output all paths
	// find best path with cp < cpTarget, then bisection search
	// how tight is our solution?
	int bestPathIdx = -1;
	float bestPathCost = std::numeric_limits<float>::max();

	for (int i = 0; i < Pcounts[goalIdx]; ++i) {
		int pathIdx = P[goalIdx*NUM + i];

		if (goalCondition && bestPathCost > pathCost[pathIdx]) {
			bestPathCost = pathCost[pathIdx];
			bestPathIdx = pathIdx;
		}

		// output path
		std::cout << "nodes = [" << pathNode[pathIdx];
		while (pathPrev[pathIdx] != -2) {
			pathIdx = pathPrev[pathIdx];
			std::cout << ", " << pathNode[pathIdx];
		}
		std::cout << "]";
	}

	// validate chosen path, or iterate to next path
	std::cout << " with cost = " << bestPathCost << std::endl;
	*costGoal = bestPathCost;
	if (bestPathCost > 10000) {
		std::cout << "FAILED TO FIND A PATH" << std::endl;
		return 0; // return to deflate the obstacles
	}

	// load samples into array
	std::vector<float> xs;
	// for (int d = 0; d < DIM; ++d)
	// 	xs[d] = samples[goalIdx*DIM+d];
	xs.clear();
	int pathIdx = bestPathIdx;
	int nodeIdx = pathNode[pathIdx];
	int pathNumSamples = 0;
	while (pathIdx != -2) {
		++pathNumSamples;
		for (int d = DIM-1; d >= 0; --d)
			xs.insert(xs.begin(), samples[nodeIdx*DIM+d]);
		pathIdx = pathPrev[pathIdx];
		nodeIdx = pathNode[pathIdx];
	}
	// printArray(&(xs[0]),pathNumSamples,DIM,std::cout);
	// std::cout << "path has " << int(xs.size()) << " elements" << std::endl;

	// solve 2pbvp
	float bestPathTopt = findOptimalPath(dt, splitPath, &(xs[0]), pathNumSamples, pathLength);
	// std::cout << "2pbvp soln is " << std::endl;

	float *d_path;
	CUDA_ERROR_CHECK(cudaMalloc(&d_path, sizeof(float)*Tmax*DIM));
	float *d_obstacles;
	CUDA_ERROR_CHECK(cudaMalloc(&d_obstacles, sizeof(float)*2*obstaclesCount*DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_obstacles, obstacles, sizeof(float)*2*obstaclesCount*DIM, cudaMemcpyHostToDevice));

	// ************************************** smoothing ************************************** 

	// load up
	// std::cout << "smoothing" << std::endl;

	float smoothPath[DIM*Tmax]; // empty array for the smoothed path
	int smoothPathLength = 0;
	std::vector<float> xsSmooth(xs.size());

	float lastSmoothBelowPath[DIM*Tmax]; // empty array for the smoothed path that was last seen below the CP constraint
	copyArray(lastSmoothBelowPath, splitPath, DIM*Tmax);
	int lastSmoothBelowPathLength = *pathLength;
	std::vector<float> xsLastSmoothBelow(xs.size());
	std::copy(xs.begin(), xs.end(), xsLastSmoothBelow.begin());
	float lastSmoothBelowCost = 1000000;

	int dtMult = 10;
	float optPath[DIM*Tmax*dtMult]; // empty array for the optimal path from the init to goal (i.e. no obstacles)
	int optPathLength = 0;
	std::vector<float> xsOpt;

	// determine the optimal path from start to goal
	for (int d = 0; d < DIM; ++d)
		xsOpt.push_back(xs[d]);
	for (int d = 0; d < DIM; ++d)
		xsOpt.push_back(xs[(pathNumSamples-1)*DIM+d]);
	float optPathTopt = findOptimalPath(dt/dtMult, optPath, &(xsOpt[0]), 2, &optPathLength);

	// std::cout << " nominal path is: " << std::endl;
	// printArray(splitPath,*pathLength,DIM,std::cout);

	// find the path points that map to the nominal path
	std::vector<float> splitPathTopts(pathNumSamples,0);
	std::vector<int> splitPathIdxs(pathNumSamples,0);
	for (int i = 0; i < pathNumSamples-1; ++i)
		splitPathTopts[i+1] = splitPathTopts[i]+toptBisection(&xs[i*DIM], &xs[(i+1)*DIM], 2);
	for (int i = 0; i < pathNumSamples; ++i)
		splitPathIdxs[i] = (optPathLength-1)*splitPathTopts[i]/splitPathTopts[splitPathTopts.size()-1];

	// std::cout << "found optimal times as: ";
	// for ( int i = 0; i < splitPathTopts.size(); i++) {
 //        std::cout << splitPathTopts[i] << " ";
 //    }
    // std::cout << std::endl;
    // std::cout << "found indexes as: ";
	// for ( int i = 0; i < splitPathIdxs.size(); i++) {
 //        std::cout << splitPathIdxs[i] << " ";
 //    }
 //    std::cout << " of " << optPathLength;

    xsOpt.clear();
    // std::cout << " means we match: " << std::endl;
    for ( int i = 0; i < pathNumSamples; i++) {
    	// printArray(&xs[i*DIM],1,DIM,std::cout);
    	// std::cout << " with "; printArray(&optPath[splitPathIdxs[i]*DIM],1,DIM,std::cout);
    	for (int d = 0; d < DIM; ++d)
    		xsOpt.push_back(optPath[splitPathIdxs[i]*DIM+d]);
    }

    // std::cout << "verify creation of xsOpt: " << std::endl;
    // for ( int i = 0; i < xsOpt.size(); i++) {
    //     std::cout << xsOpt[i] << " ";
    //     if ((i + 1) % 6 == 0)
    //     	std::cout << std::endl;
    // }

	// generate new xsSmooth and enter loop
	int maxSmoothItrs = 15;
	int smoothItrs = 0;
	float alpha = 1.0;
	float alphaU = 1.0;
	float alphaL = 0.0;

	// TODO if exit with maxSmoothItrs, need to default to the last path under the CP constraint
	// save the max close wise
	float smoothPathCost = 0;
	// float solnPathCP = 0;
	while (smoothItrs < maxSmoothItrs) {
		for (int i = 0; i < xs.size(); ++i)
			xsSmooth[i] = (1-alpha)*xs[i] + alpha*xsOpt[i];

        smoothPathLength = 0;
        smoothPathCost = 0;
		findOptimalPath(dt, smoothPath, &(xsSmooth[0]), pathNumSamples, &smoothPathLength); // ignoring return of topt
		for (int i = 0; i < pathNumSamples-1; ++i) {
			float tau = toptBisection(&xsSmooth[i*DIM], &xsSmooth[(i+1)*DIM], 2);
			smoothPathCost += cost(tau, &xsSmooth[i*DIM], &xsSmooth[(i+1)*DIM]);
		}

		CUDA_ERROR_CHECK(cudaMemcpy(d_path, smoothPath, sizeof(float)*Tmax*DIM, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		bool collisionFree;
		bool *d_collisionFree;
		CUDA_ERROR_CHECK(cudaMalloc(&d_collisionFree, sizeof(bool)));

		isFreePath<<<1,1>>>(d_obstaclesInflated, obstaclesCount, d_path, smoothPathLength, d_collisionFree);
		cudaDeviceSynchronize();
		CUDA_ERROR_CHECK(cudaMemcpy(&collisionFree, d_collisionFree, sizeof(bool), cudaMemcpyDeviceToHost));		

		// std::cout << "-------------- iteration " << smoothItrs << ", alpha = " << alpha << 
		// 	", col free = " << collisionFree << ", cost = " << smoothPathCost << std::endl;

		if (!collisionFree)
			alphaU = alpha;
		if (collisionFree)
			alphaL = alpha;
		alpha = (alphaL + alphaU)/2;

		// go for path closest to the cp limit
		if (!collisionFree) {
			// std::cout << " NOT FREE smoothed path is: " << std::endl;
			// printArray(smoothPath,smoothPathLength,DIM,std::cout);
		}

		if (collisionFree) {
			std::cout << "NEW BEST PATH!" << std::endl;	
			
			// std::cout << " FREE smoothed path is: " << std::endl;
			// printArray(smoothPath,smoothPathLength,DIM,std::cout);

			copyArray(lastSmoothBelowPath, smoothPath, DIM*Tmax);
			lastSmoothBelowPathLength = smoothPathLength;
			std::vector<float> xsLastSmoothBelow(xs.size());
			std::copy(xsSmooth.begin(), xsSmooth.end(), xsLastSmoothBelow.begin());
			lastSmoothBelowCost = smoothPathCost;
		}

		++smoothItrs;
	}

	// std::cout << " smoothed path is: " << std::endl;
	// printArray(lastSmoothBelowPath,lastSmoothBelowPathLength,DIM,std::cout);

	copyArray(splitPath, lastSmoothBelowPath, DIM*Tmax);
	std::cout << "cost = " << lastSmoothBelowCost << std::endl;

	*costGoal = lastSmoothBelowCost;

	CUDA_ERROR_CHECK(cudaMemcpy(d_path, splitPath, sizeof(float)*Tmax*DIM, cudaMemcpyHostToDevice));
	
	double t_CPStart = std::clock();
	float cp = collisionProbability(d_obstacles, obstaclesCount, d_offsets, offsetMult, d_path, lastSmoothBelowPathLength);
	double t_CP = (std::clock() - t_CPStart) / (double) CLOCKS_PER_SEC;
	// std::cout << "CP took: " << t_CP << std::endl;
	// std::cout << "Collision Probability = " << cp << std::endl;

	cudaFree(d_path);
	cudaFree(d_obstacles);
	
	return cp;
}

/***********************
GPU kernels
***********************/
// probably just called with one thread, I hate how I am implementing this for the one time use
__global__ 
void isFreePath(float *obstacles, int obstaclesCount, 
	float *path, int pathLength,
	bool *collisionFree) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= 1)
		return;

	float v[DIM], w[DIM];
	float bbMin[DIM], bbMax[DIM];
	bool motionValid = true;
	for (int i = 0; i < pathLength-1; ++i) {
		if (!motionValid)
			break;
		for (int d = 0; d < DIM; ++d) {
			v[d] = path[i*DIM + d];
			w[d] = path[(i+1)*DIM + d];

			if (v[d] > w[d]) {
				bbMin[d] = w[d];
				bbMax[d] = v[d];
			} else {
				bbMin[d] = v[d];
				bbMax[d] = w[d];
			}
		}
		motionValid = motionValid && isMotionValid(v, w, bbMin, bbMax, obstaclesCount, obstacles, NULL);
	}

	collisionFree[0] = motionValid;
}




