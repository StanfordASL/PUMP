#include "CCGMT.cuh"

// TODO: what if CCGMT were all on the CPU except for the CP computation?
// now the coding is much simpler, and I can use vectors more easily

/***********************
CPU functions
***********************/
void CCGMT(float rn, float cpTarget) {
	// TODO: fill in from working CCGMTcpu
}

// a CPU heavy version of CCGMT, the next paths and wavefronts are tracked on the CPU
// easily parallel operations like CP calc and dominance checks are still done on the GPU
void CCGMTcpu(float rn, float cpTarget, int initIdx, int goalIdx, float dt, int numMCSamples, 
	float *h, int *nnGoSizes, int *nnGoEdges, int *nnIdxs, int maxNNSize, bool *isFreeSamples, bool *isFreeEdges,
	int numDisc, int obstaclesCount, float *d_topts, float *topts, float *d_copts, float *copts, 
	float *d_as, float *d_bs, int *d_countStoredHSs, float *d_offsets, float *d_offsetsTime, float offsetMult,
	std::vector<int> P, int *Pcounts, int *pathPrev, int *pathNode, float *pathCost, float *pathTime, float *pathCP, 
	int maxPathCount, std::vector<int> G, int *sizeG, bool *d_pathValidParticles, float *d_pathCP, float *d_pathTime,
	int *wavefrontPathPrev, int *wavefrontNodeNext, int *wavefrontEdge,
	int *d_wavefrontPathPrev, int *d_wavefrontNodeNext, int *d_wavefrontEdge,
	float epsCost, float epsCP, float lambda, int numBuckets, float cpFactor, int numHSMCParticles,
	float *samples, float *obstacles, int Tmax)
{
	std::cout << "*********************** Beginning CCGMT ***********************" << std::endl;
	double t_ccgmtStart = std::clock();
	cudaError_t code;
	float t_propTot = 0;

	float dr = lambda*rn;
	float cpMinSoln = cpTarget/cpFactor; // continue finding new paths until a path below this is found
 	float cpCutoff = cpTarget*cpFactor; // cutoff any paths with CPs higher than this cutoff

 	std::cout << "CP target = " << cpTarget << ", CP min soln = " << cpMinSoln << " CP cutoff = " << cpCutoff << std::endl;

	int numPaths = 1;

	// setup initial path
	int Gidx = 0;
	bool goalCondition = false;
	bool goalConditionNext = false;
	bool emptyOpenSet = false;
	P[initIdx*NUM + 0] = 0; // path 0 is at initIdx
	Pcounts[initIdx]++;
	pathPrev[0] = -2; // denote end
	pathNode[0] = initIdx; 
	pathCost[0] = 0;
	pathCP[0] = 0;
	pathTime[0] = 0;
	sizeG[Gidx]++;
	G[0] = 0;

	float costThreshold = h[initIdx]; 
	int maxItrs = 20;
	int itrs = 0;

	// *************************** exploration loop ***************************
	int numConsideredPaths = 0;
	while (itrs < maxItrs && !goalCondition && !emptyOpenSet) { // cutoff at solution exists with cp = cpMinSoln or expansion is empty
		++itrs;
		std::cout << "************** starting iteration " << itrs << " with " << sizeG[Gidx] << " paths" << std::endl;

		if (goalConditionNext)
			goalCondition = true;

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
					return;
				}
			}
		}

		if (numPaths + numNewPaths >= maxPathCount) {
			std::cout << "maxPathCount reached, increase max number of paths" << std::endl;
			return;
		}
		
		sizeG[Gidx] = 0; // reset G size

		// copy necessary info to GPU
		cudaDeviceSynchronize();
		CUDA_ERROR_CHECK(cudaMemcpy(d_wavefrontPathPrev, wavefrontPathPrev, sizeof(int)*numNewPaths, cudaMemcpyHostToDevice));
		CUDA_ERROR_CHECK(cudaMemcpy(d_wavefrontEdge, wavefrontEdge, sizeof(int)*numNewPaths, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		// calculate CP (do half plane checks per particle, then sum)
		// copy over path times
		float *d_debugOutput;
		CUDA_ERROR_CHECK(cudaMalloc(&d_debugOutput, sizeof(float)*numNewPaths*numHSMCParticles));

		double t_propStart = std::clock();
		numConsideredPaths += numNewPaths;

		const int blockPropCP = 512;
		// call correct propogate for number of paths
		if ((numNewPaths*numHSMCParticles + blockPropCP - 1) / blockPropCP < 2147483647) {
			const int gridPropCP = std::min(
				(numNewPaths*numHSMCParticles + blockPropCP - 1) / blockPropCP, 2147483647);
			// if path centric ^ is numNewPaths, otherwisae it is numNewPaths*numHSMCParticles
			std::cout << "launching propagateValidParticles (" << blockPropCP << ", " << gridPropCP << ") new paths = " << numNewPaths;
			propagateValidParticles<<<gridPropCP, blockPropCP>>>(
				numNewPaths, numDisc+1, d_wavefrontEdge, d_wavefrontPathPrev, numPaths,	 
				d_as, d_bs, d_countStoredHSs, d_pathTime, d_topts, dt, obstaclesCount, d_offsetsTime, offsetMult,
				numMCSamples, d_pathValidParticles,
				numHSMCParticles, d_debugOutput);
		} else {
			int numThreadParticles = 2;
			while ((numNewPaths*numHSMCParticles/numThreadParticles + blockPropCP - 1) / blockPropCP > 2147483647)
				numThreadParticles *= 2;

			const int gridPropCP = std::min(
				(numNewPaths*numHSMCParticles/numThreadParticles + blockPropCP - 1) / blockPropCP, 2147483647);
			std::cout << "launching propagateValidParticlesMany (" << blockPropCP << ", " << gridPropCP << ")" << " and numThreadParticles = " << numThreadParticles;
			propagateValidParticlesMany<<<gridPropCP, blockPropCP>>>(
				numNewPaths, numDisc+1, d_wavefrontEdge, d_wavefrontPathPrev, numPaths,	 
				d_as, d_bs, d_countStoredHSs, d_pathTime, d_topts, dt, obstaclesCount, d_offsetsTime, offsetMult, 
				numMCSamples, d_pathValidParticles,
				numHSMCParticles, numThreadParticles, d_debugOutput);
		}

		// float debugOutput[numNewPaths*numHSMCParticles];
		// CUDA_ERROR_CHECK(cudaMemcpy(debugOutput, d_debugOutput, sizeof(float)*numNewPaths*numHSMCParticles, cudaMemcpyDeviceToHost));
		// std::cout << "debugOutput = "; printArray(debugOutput,1,200,std::cout);

		cudaDeviceSynchronize();
		float t_prop = (std::clock() - t_propStart) / (double) CLOCKS_PER_SEC;
		t_propTot += t_prop;
		std::cout << ", prop time: " << t_prop << " s" << std::endl;

		code = cudaPeekAtLastError();
		if (cudaSuccess != code) { std::cout << "ERROR on propagateValidParticles: " << cudaGetErrorString(code) << std::endl; }

		double t_calcCPStart = std::clock();
		const int blockCP_HSMC = 128;
		const int gridCP_HSMC = std::min(
			(numNewPaths + blockCP_HSMC - 1) / blockCP_HSMC, 2147483647);
		std::cout << "launching calculateCP_HSMC (" << blockCP_HSMC << ", " << gridCP_HSMC << ")";
		calculateCP_HSMC<<<gridCP_HSMC, blockCP_HSMC>>>(
			numNewPaths, numPaths, d_pathCP, d_pathValidParticles, numHSMCParticles);

		cudaDeviceSynchronize();
		float t_calcCP = (std::clock() - t_calcCPStart) / (double) CLOCKS_PER_SEC;
		std::cout << ", calc time: " << t_calcCP << " s" << std::endl;

		code = cudaPeekAtLastError();
		if (cudaSuccess != code) { std::cout << "ERROR on calculateCP_HSMC: " << cudaGetErrorString(code) << std::endl; }

		// float debugOutput[numNewPaths*numHSMCParticles];
		// CUDA_ERROR_CHECK(cudaMemcpy(debugOutput, d_debugOutput, sizeof(float)*numNewPaths*numHSMCParticles, cudaMemcpyDeviceToHost));
		// std::cout << "debugOutput = "; printArray(debugOutput,1,200,std::cout);

		// int npp = 20;
		// bool validParticles[numHSMCParticles*npp];
		// cudaMemcpy(validParticles, d_pathValidParticles, sizeof(bool)*numHSMCParticles*npp, cudaMemcpyDeviceToHost);
		// std::cout << "valid particles"; printArray(validParticles, npp, numHSMCParticles, std::cout);

		// copy back CP, fill in CP, time, cost
		CUDA_ERROR_CHECK(cudaMemcpy(pathCP + numPaths, d_pathCP + numPaths, sizeof(float)*numNewPaths, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		// std::cout << "path CP = "; printArray(pathCP,1,20,std::cout);

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

			if (pathCP[pathIdx] > cpCutoff) { // CP cutoff reached
				wavefrontNodeNext[i] = -1;
				continue;
			}

			P[nodeIdx*NUM + Pcounts[nodeIdx] + PnewCount[nodeIdx]] = pathIdx;
			PnewCount[nodeIdx]++;
		}

		// check new paths against stored paths
		for (int i = 0; i < numNewPaths; ++i) {
			int nodeIdx = wavefrontNodeNext[i];
			// already eliminated or at the goal idx
			if (wavefrontNodeNext[i] == -1 || nodeIdx == goalIdx) // don't check goal
				continue;
			int pathIdx = numPaths + i;

			for (int j = 0; j < Pcounts[nodeIdx] + PnewCount[nodeIdx]; ++j) {
				int pathIdxCompare = P[nodeIdx*NUM + j];
				// don't compare to self
				if (pathIdxCompare == pathIdx)
					continue;

				// eps comparison
				if (pathCP[pathIdxCompare] < epsCP + pathCP[pathIdx] && 
					pathCost[pathIdxCompare] < epsCost + pathCost[pathIdx]) {
					
					// check if paths are co-dominant, then keep the one with a lower path number
					if (pathCP[pathIdxCompare] + epsCP >= pathCP[pathIdx] && 
						pathCost[pathIdxCompare] + epsCost >= pathCost[pathIdx] &&
						pathIdx < pathIdxCompare) {
						// std::cout << "co dom: (" << pathCP[pathIdxCompare] << ", " << pathCost[pathIdxCompare] << ") vs ( " 
						// 	<< pathCP[pathIdx] << ", " << pathCost[pathIdx] << ")" << std::endl;
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
			pathCP[pathIdxStore] = pathCP[pathIdx];
			pathTime[pathIdxStore] = pathTime[pathIdx];
			pathCost[pathIdxStore] = pathCost[pathIdx];
			CUDA_ERROR_CHECK(cudaMemcpy(d_pathValidParticles + numHSMCParticles*pathIdxStore,
				d_pathValidParticles + numHSMCParticles*pathIdx, sizeof(bool)*numHSMCParticles, cudaMemcpyDeviceToDevice));

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
		CUDA_ERROR_CHECK(cudaMemcpy(d_pathCP + numPaths, pathCP + numPaths, sizeof(float)*numValidNewPaths, cudaMemcpyHostToDevice));
		CUDA_ERROR_CHECK(cudaMemcpy(d_pathTime + numPaths, pathTime + numPaths, sizeof(float)*numValidNewPaths, cudaMemcpyHostToDevice));

		numPaths += numValidNewPaths;

		// update goal condition
		for (int i = 0; i < Pcounts[goalIdx]; ++i) {
			int pathIdx = P[goalIdx*NUM + i];
			if (pathCP[pathIdx] <= cpMinSoln) {
				goalConditionNext = true;
				goalCondition = true;
			}
		}
		if (goalCondition) {
			std::cout << "goalCondition met" << std::endl;
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
			return;
		}
		
	}
	// std::cout << "path CP = "; printArray(pathCP,1,numPaths,std::cout);
	float t_ccgmt = (std::clock() - t_ccgmtStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Explore took: " << t_ccgmt << " s" << std::endl;
	std::cout << "number of goal paths found = " << Pcounts[goalIdx] << " with a total number of paths = " << numPaths << " of " << numConsideredPaths << std::endl;

	// ************************************** post exploration loop ************************************** 
	// load samples into array
	std::vector<float> xs;

	int pathIdx;
	int nodeIdx;
	int pathNumSamples = 0;

	float splitPath[DIM*Tmax]; // empty array for the optimal path
	int pathLength = 0;

	float *d_path;
	CUDA_ERROR_CHECK(cudaMalloc(&d_path, sizeof(float)*Tmax*DIM));
	float *d_obstacles;
	CUDA_ERROR_CHECK(cudaMalloc(&d_obstacles, sizeof(float)*2*obstaclesCount*DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_obstacles, obstacles, sizeof(float)*2*obstaclesCount*DIM, cudaMemcpyHostToDevice));

	std::cout << "*********************** output paths and check cp heursitic ********************" << std::endl;
	for (int i = 0; i < Pcounts[goalIdx]; ++i) {
		xs.clear();
		int solnPathIdx = P[goalIdx*NUM + i];
		pathIdx = solnPathIdx;
		nodeIdx = pathNode[pathIdx];
		pathNumSamples = 0;
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
		pathLength = 0;
		findOptimalPath(dt, splitPath, &(xs[0]), pathNumSamples, &pathLength); // ignoring return of topt

		CUDA_ERROR_CHECK(cudaMemcpy(d_path, splitPath, sizeof(float)*Tmax*DIM, cudaMemcpyHostToDevice));
		
		float actualCP = collisionProbability(d_obstacles, obstaclesCount, d_offsets, offsetMult, d_path, pathLength);
		std::cout << "Path " << solnPathIdx << ": cost = " << pathCost[solnPathIdx] << 
			", time = " << pathTime[solnPathIdx] << 
			", cp = " << actualCP << " vs " << pathCP[solnPathIdx] << " est, ";
		pathIdx = solnPathIdx;
		std::cout << "nodes = [" << pathNode[pathIdx];
		while (pathPrev[pathIdx] != -2) {
			pathIdx = pathPrev[pathIdx];
			std::cout << ", " << pathNode[pathIdx];
		}
		std::cout << "]"<< std::endl;
	}

	// ************************************** search for best path ************************************** 
	// not currently implemented as bisection, 
	// but rather choose best path's by cost and check if cp condition is met
	// put paths in a proirity queue by cost

	float t_searchStart = std::clock();
	int goalPathsCount = Pcounts[goalIdx];
	std::vector< std::pair<float, int> > goalPaths;
	for (int i = 0; i < goalPathsCount; ++i) {
		int pathIdx = P[goalIdx*NUM + i];
		std::pair<float, int> pathPair(pathCost[pathIdx], pathIdx);
		goalPaths.push_back(pathPair);
	}
	std::sort(goalPaths.begin(), goalPaths.end());

	int solnPathIdx = -1;
	float solnPathCP = 0;
	float solnPathCost = 0;
	bool success = false;

	int searchItrs = 0;
	std::cout << "****************** Beginning search ******************" << std::endl;
	for (std::vector< std::pair<float, int> >::iterator solnPath = goalPaths.begin(); 
		solnPath != goalPaths.end(); 
		++solnPath) {
		xs.clear();

		solnPathIdx = solnPath->second;
		pathIdx = solnPathIdx;
		nodeIdx = pathNode[pathIdx];
		pathNumSamples = 0;
		while (pathIdx != -2) {
			++pathNumSamples;
			for (int d = DIM-1; d >= 0; --d)
				xs.insert(xs.begin(), samples[nodeIdx*DIM+d]);
			pathIdx = pathPrev[pathIdx];
			nodeIdx = pathNode[pathIdx];
		}

		pathLength = 0;
		findOptimalPath(dt, splitPath, &(xs[0]), pathNumSamples, &pathLength); // ignoring return of topt

		CUDA_ERROR_CHECK(cudaMemcpy(d_path, splitPath, sizeof(float)*Tmax*DIM, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		solnPathCP = collisionProbability(d_obstacles, obstaclesCount, d_offsets, offsetMult, d_path, pathLength);
		solnPathCost = pathCost[solnPathIdx];
		cudaDeviceSynchronize();
		searchItrs++;

		std::cout << "tried path " << solnPathIdx << ", cost = " << pathCost[solnPathIdx] << ", cp = " << solnPathCP << std::endl;
		if (solnPathCP < cpTarget) {
			std::cout << "SUCCESS" << std::endl;
			success = true;
			break;
		}
	}
	std::cout << "****************** Beginning smoothing ******************" << std::endl;
	// start smoothing procedure
	// the nominal solution path is in splitPath, xs, and pathLength
	float smoothPath[DIM*Tmax]; // empty array for the smoothed path
	int smoothPathLength = 0;
	std::vector<float> xsSmooth(xs.size());

	float lastSmoothBelowPath[DIM*Tmax]; // empty array for the smoothed path that was last seen below the CP constraint
	copyArray(lastSmoothBelowPath, splitPath, DIM*Tmax);
	int lastSmoothBelowPathLength = pathLength;
	std::vector<float> xsLastSmoothBelow(xs.size());
	std::copy(xs.begin(), xs.end(), xsLastSmoothBelow.begin());
	float lastSmoothBelowCost = solnPathCost;
	float lastSmoothBelowCP = solnPathCP;

	int dtMult = 10;
	float optPath[DIM*Tmax*dtMult]; // empty array for the optimal path from the init to goal (i.e. no obstacles)
	int optPathLength = 0;
	std::vector<float> xsOpt;

	if (success) {
		// determine the optimal path from start to goal
		for (int d = 0; d < DIM; ++d)
			xsOpt.push_back(xs[d]);
		for (int d = 0; d < DIM; ++d)
			xsOpt.push_back(xs[(pathNumSamples-1)*DIM+d]);
		float optPathTopt = findOptimalPath(dt/dtMult, optPath, &(xsOpt[0]), 2, &optPathLength);

		// std::cout << " optimal path is: " << std::endl;
		// printArray(optPath,optPathLength,DIM,std::cout);

		std::cout << " from the original nodes: " << std::endl;
		for ( int i = 0; i < xs.size(); i++) {
            std::cout << xs[i] << " ";
            if ((i + 1) % 6 == 0)
            	std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << "which had a path time = " << pathTime[solnPathIdx] << std::endl;

		// find the path points that map to the nominal path
		std::vector<float> splitPathTopts(pathNumSamples,0);
		std::vector<int> splitPathIdxs(pathNumSamples,0);
		for (int i = 0; i < pathNumSamples-1; ++i)
			splitPathTopts[i+1] = splitPathTopts[i]+toptBisection(&xs[i*DIM], &xs[(i+1)*DIM], 2);
		for (int i = 0; i < pathNumSamples; ++i)
			splitPathIdxs[i] = (optPathLength-1)*splitPathTopts[i]/splitPathTopts[splitPathTopts.size()-1];

		std::cout << "found optimal times as: ";
		for ( int i = 0; i < splitPathTopts.size(); i++) {
            std::cout << splitPathTopts[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "found indexes as: ";
		for ( int i = 0; i < splitPathIdxs.size(); i++) {
            std::cout << splitPathIdxs[i] << " ";
        }
        std::cout << " of " << optPathLength;

        xsOpt.clear();
        std::cout << " means we match: " << std::endl;
        for ( int i = 0; i < pathNumSamples; i++) {
        	printArray(&xs[i*DIM],1,DIM,std::cout);
        	std::cout << " with "; printArray(&optPath[splitPathIdxs[i]*DIM],1,DIM,std::cout);
        	for (int d = 0; d < DIM; ++d)
        		xsOpt.push_back(optPath[splitPathIdxs[i]*DIM+d]);
        }

        std::cout << "verify creation of xsOpt: " << std::endl;
        for ( int i = 0; i < xsOpt.size(); i++) {
            std::cout << xsOpt[i] << " ";
            if ((i + 1) % 6 == 0)
            	std::cout << std::endl;
        }

		// generate new xsSmooth and enter loop
		int maxSmoothItrs = 30;
		int smoothItrs = 0;
		float epsSmoothCP = 0.0002;
		float alpha = 1.0;
		float alphaU = 1.0;
		float alphaL = 0.0;

		// TODO if exit with maxSmoothItrs, need to default to the last path under the CP constraint
		// save the max close wise
		float smoothPathCost = 0;
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

			solnPathCP = collisionProbability(d_obstacles, obstaclesCount, d_offsets, offsetMult, d_path, smoothPathLength);
			cudaDeviceSynchronize();

			std::cout << "-------------- iteration " << smoothItrs << ", alpha = " << alpha << ", cp = " << solnPathCP << ", cost = " << smoothPathCost << std::endl;
			// for ( int i = 0; i < xsSmooth.size(); i++) {
			//           std::cout << xsSmooth[i] << " ";
			//           if ((i + 1) % 6 == 0)
			//           	std::cout << std::endl;
			//       }

			if (solnPathCP < cpTarget)
				alphaL = alpha;
			if (solnPathCP >= cpTarget)
				alphaU = alpha;
			alpha = (alphaL + alphaU)/2;

			// go for path closest to the cp limit
			if (solnPathCP < cpTarget && solnPathCP > lastSmoothBelowCP) {
				std::cout << "NEW BEST PATH!" << std::endl;	

				copyArray(lastSmoothBelowPath, smoothPath, DIM*Tmax);
				lastSmoothBelowPathLength = smoothPathLength;
				std::vector<float> xsLastSmoothBelow(xs.size());
				std::copy(xsSmooth.begin(), xsSmooth.end(), xsLastSmoothBelow.begin());
				lastSmoothBelowCost = smoothPathCost;
				lastSmoothBelowCP = solnPathCP;
			}

			if (std::abs(cpTarget - solnPathCP) < epsSmoothCP &&
				cpTarget - solnPathCP > 0) {
				std::cout << "breaking" << std::endl;
				break;
			}

			++smoothItrs;
		}

		std::cout << "smoothing gave a cost savings of " << 100-100*lastSmoothBelowCost/solnPathCost << "%" << std::endl;
		solnPathCost = lastSmoothBelowCost;
	}

	float t_search = (std::clock() - t_searchStart) / (double) CLOCKS_PER_SEC;
	std::cout << "****************** Final output ******************" << std::endl;
	if (!success) {
		std::cout << "FAILURE, need to restart search" << std::endl;
	} else {
		std::cout << "Search took: " << t_search << " s" << std::endl;
		std::cout << "Solution path = " << solnPathIdx << 
			" with cp = " << lastSmoothBelowCP <<
			" and cost = " << lastSmoothBelowCost << std::endl;

		// output soln path to path.txt
		std::ofstream ofs ("path.txt", std::ofstream::out);
		printArray(lastSmoothBelowPath,lastSmoothBelowPathLength,DIM,ofs);
	  	ofs.close();
	}

	cudaFree(d_path);
	cudaFree(d_obstacles);

	std::cout << "****************** To Excel results: ******************" << std::endl;
	std::cout << "cp: \t\t\t" << lastSmoothBelowCP << std::endl;
	std::cout << "cost: \t\t\t" << lastSmoothBelowCost << std::endl;
	std::cout << "Explore time: \t\t" << t_ccgmt*1000 << " ms (" << t_propTot*1000 << " ms in propagate cp)" << std::endl;
	std::cout << "SolnSearch time: \t" << t_search*1000 << " ms" << std::endl;
	
	return;
}

/***********************
GPU kernels
***********************/
// propogate valid particles for new halfspaces along the edge
__global__ void propagateValidParticles(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, int *countStoredHSs, float *times, float *topts, float dt,
	int obstaclesCount, float *offsets,  float offsetMult, int numMCSamples, bool *validParticles,
	int numHSMCParticles, float *debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numNewPaths*numHSMCParticles)
		return;

	int wavefrontIdx = tid/numHSMCParticles;
	int pathIdx = pathBaseIdx + tid/numHSMCParticles;
	int edgeIdx = wavefrontEdge[wavefrontIdx];
	int pathIdxPrev = wavefrontPathPrev[wavefrontIdx];
	int particleIdx = tid % numHSMCParticles;

	// as and bs into shared memory
	bool validParticle = validParticles[pathIdxPrev*numHSMCParticles + particleIdx];

	float dtLocal = topts[edgeIdx]/(waypointsCount-1);
	// loop through waypoints
	// debugOutput[tid] = countStoredHSs[edgeIdx*waypointsCount + waypointsCount-1];
	int hsHeadIdx = edgeIdx*waypointsCount*obstaclesCount;
	for (int i = 0; i < waypointsCount; ++i) {
		float time = times[pathIdxPrev] + i*dtLocal;
		int dtIdx = time/dt; // TODO: make this round, not floor

		// countStoredHSs[edgeIdx*waypointsCount+i];
		for (int o = 0; o < obstaclesCount; ++o) { // loop through obstacles
			if (bs[hsHeadIdx + i*obstaclesCount + o] >= 1000)
				continue; 

			float LHS = 0; // offset*a
			for (int d = 0; d < DIM/2; ++d) {
				LHS += offsets[dtIdx*numMCSamples*DIM/2 + particleIdx*DIM/2 + d]*   //particleIdx*Tmax*DIM/2 + dtIdx*DIM/2 +d]*
					as[hsHeadIdx*DIM/2 + i*obstaclesCount*DIM/2 + o*DIM/2 + d];
			}
			validParticle &=
				(LHS*offsetMult < bs[hsHeadIdx + i*obstaclesCount + o]);
		}
	}
	validParticles[pathIdx*numHSMCParticles + particleIdx] = validParticle;
}

// propogate valid particles for new halfspaces along the edge
__global__ void propagateValidParticlesMany(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, int *countStoredHSs, float *times, float *topts, float dt,
	int obstaclesCount, float *offsets, float offsetMult, int numMCSamples, bool *validParticles,
	int numHSMCParticles, int numThreadParticles, float *debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numNewPaths*numHSMCParticles/numThreadParticles)
		return;
	int baseTid = numThreadParticles*tid;

	// loop through particles assined to this thread, could add checking to make sure 
	// we don't go over the actual max number of particles, but with since everything is multiples
	// of 2, 4, 8, 16, 32 it is unlikely we'd need
	for (int p = baseTid; p < baseTid+numThreadParticles; ++p) {
		int wavefrontIdx = p/numHSMCParticles;
		int pathIdx = pathBaseIdx + p/numHSMCParticles;
		int edgeIdx = wavefrontEdge[wavefrontIdx];
		int pathIdxPrev = wavefrontPathPrev[wavefrontIdx];
		int particleIdx = p % numHSMCParticles;
		int hsHeadIdx = edgeIdx*waypointsCount*obstaclesCount;

		// validParticles[pathIdx*numHSMCParticles + particleIdx] = validParticles[pathIdxPrev*numHSMCParticles + particleIdx];
		bool validParticle = validParticles[pathIdxPrev*numHSMCParticles + particleIdx];

		float dtLocal = topts[edgeIdx]/(waypointsCount-1);
		// loop through waypoints
		for (int i = 0; i < waypointsCount; ++i) {
			float time = times[pathIdxPrev] + i*dtLocal;
			int dtIdx = time/dt; // TODO: make this round, not floor

			// countStoredHSs[edgeIdx*waypointsCount + i]
			for (int o = 0; o < obstaclesCount; ++o) { // loop through obstacles
				if (bs[hsHeadIdx + i*obstaclesCount + o] >= 1000)
					continue; // TODO: or continue if using make all hs's

				float LHS = 0; // offset*a
				for (int d = 0; d < DIM/2; ++d) {
					LHS += offsets[dtIdx*numMCSamples*DIM/2 + particleIdx*DIM/2 + d]*   //particleIdx*Tmax*DIM/2 + dtIdx*DIM/2 +d]*
						as[hsHeadIdx*DIM/2 + i*obstaclesCount*DIM/2 + o*DIM/2 + d];
				}
				validParticle &=
					(LHS*offsetMult < bs[hsHeadIdx + i*obstaclesCount + o]);
			}
		}
		validParticles[pathIdx*numHSMCParticles + particleIdx] = validParticle;
	}
}

__global__ void propagateValidParticlesPathCentric(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, float *times, float *topts, float dt,
	int obstaclesCount, float *offsets, int numMCSamples, bool *validParticles,
	int numHSMCParticles, float *debugOutput)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numNewPaths)
		return;

	int wavefrontIdx = tid;
	int pathIdx = pathBaseIdx + tid;
	int edgeIdx = wavefrontEdge[wavefrontIdx];
	int pathIdxPrev = wavefrontPathPrev[wavefrontIdx];

	for (int p = 0; p < numHSMCParticles; ++p) {
		validParticles[pathIdx*numHSMCParticles + p] = validParticles[pathIdxPrev*numHSMCParticles + p];

		float dtLocal = topts[edgeIdx]/(waypointsCount);
		// loop through waypoints
		for (int i = 0; i < waypointsCount; ++i) {
			float time = times[pathIdxPrev] + i*dtLocal;
			int dtIdx = time/dt; // TODO: make this round, not floor

			for (int o = 0; o < obstaclesCount; ++o) { // loop through obstacles
				float LHS = 0; // offset*a
				for (int d = 0; d < DIM/2; ++d) {
					LHS += offsets[p*DIM/2*Tmax + dtIdx*DIM/2 + d]* //  [dtIdx*numMCSamples*DIM + p*DIM + d]*
						as[edgeIdx*(waypointsCount)*obstaclesCount*DIM/2 + i*obstaclesCount*DIM/2 + o*DIM/2 + d];
				}
				validParticles[pathIdx*numHSMCParticles + p] &=
					(LHS < bs[edgeIdx*waypointsCount*obstaclesCount + i*obstaclesCount + o]);
			}
		}
	}
}

// sum valid particles to find the new CP estimate
__global__ void calculateCP_HSMC(int numNewPaths, int pathBaseIdx, float *pathCP, bool *validParticles, int numHSMCParticles) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numNewPaths)
		return;

	int numCollisions = 0;
	for (int i = 0; i < numHSMCParticles; ++i)
		if (!validParticles[(pathBaseIdx + tid)*numHSMCParticles + i])
			numCollisions++;

	pathCP[pathBaseIdx + tid] = ((float) numCollisions)/numHSMCParticles;
	// pathCP[pathBaseIdx + tid] = 0.02;
}

// perform dominance check to mark dominated paths as dead
__global__ void dominanceCheck() 
{

}

__global__ void setupValidParticles(bool *validParticles, int numHSMCParticles, int maxPathCount,
	float *pathCP, float *pathTime)
{
	int pathIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pathIdx >= maxPathCount)
		return;

	for (int i = 0; i < numHSMCParticles; ++i)
		validParticles[pathIdx*numHSMCParticles + i] = true;
	pathCP[pathIdx] = 1;
	pathTime[pathIdx] = 0;
}




