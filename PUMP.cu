#include "PUMP.cuh"

void PUMP(MotionPlanningProblem mpp)
{
	std::cout << "*********************** Beginning PUMP ***********************" << std::endl;
	double t_ccgmtStart = std::clock();
	cudaError_t code;
	float t_propTot = 0;

	float dr = mpp.lambda*mpp.rn;
	float cpMinSoln = mpp.cpTarget/mpp.cpFactor; // continue finding new paths until a path below this is found
 	float cpCutoff = mpp.cpTarget*mpp.cpFactor; // cutoff any paths with CPs higher than this cutoff

 	std::cout << "CP target = " << mpp.cpTarget << ", CP min soln = " << cpMinSoln << " CP cutoff = " << cpCutoff << std::endl;

	int numPaths = 1;

	// setup initial path
	int Gidx = 0;
	bool goalCondition = false;
	bool goalConditionNext = false;
	bool emptyOpenSet = false;
	mpp.pumpSearch.P[mpp.initIdx*NUM + 0] = 0; // path 0 is at mpp.initIdx
	mpp.pumpSearch.Pcounts[mpp.initIdx]++;
	mpp.pumpSearch.pathPrev[0] = -2; // denote end
	mpp.pumpSearch.pathNode[0] = mpp.initIdx; 
	mpp.pumpSearch.pathCost[0] = 0;
	mpp.pumpSearch.pathCP[0] = 0;
	mpp.pumpSearch.pathTime[0] = 0;
	mpp.pumpSearch.sizeG[Gidx]++;
	mpp.pumpSearch.G[0] = 0;

	float costThreshold = mpp.roadmap.h[mpp.initIdx]; 
	int maxItrs = 20;
	int itrs = 0;

	// *************************** exploration loop ***************************
	int numConsideredPaths = 0;
	while (itrs < maxItrs && !goalCondition && !emptyOpenSet) { // cutoff at solution exists with cp = cpMinSoln or expansion is empty
		++itrs;
		std::cout << "************** starting iteration " << itrs << " with " << mpp.pumpSearch.sizeG[Gidx] << " paths" << std::endl;

		if (goalConditionNext)
			goalCondition = true;

		int numNewPaths = 0;
		for (int g = 0; g < mpp.pumpSearch.sizeG[Gidx]; ++g) {
			int pathIdxPrev = mpp.pumpSearch.G[Gidx*mpp.pumpSearch.maxPathCount + g]; // path to expand from
			mpp.pumpSearch.G[Gidx*mpp.pumpSearch.maxPathCount + g] = -1; // clear out this g
			int nodeIdxPrev = mpp.pumpSearch.pathNode[pathIdxPrev];
			for (int nn = 0; nn < mpp.roadmap.nnGoSizes[nodeIdxPrev]; ++nn) {
				int nodeIdxNext = mpp.roadmap.nnGoEdges[nodeIdxPrev*mpp.roadmap.maxNNSize + nn]; // node to expand to
				int edgeIdx = mpp.roadmap.nnIdxs[nodeIdxNext*NUM + nodeIdxPrev]; // edge index connecting prev to next
				// check if edge is collision free and the sample is free
				if (!mpp.roadmap.isFreeEdges[edgeIdx] || !mpp.roadmap.isFreeSamples[nodeIdxNext]) 
					continue;

				mpp.pumpSearch.wavefrontPathPrev[numNewPaths] = pathIdxPrev;
				mpp.pumpSearch.wavefrontNodeNext[numNewPaths] = nodeIdxNext;
				mpp.pumpSearch.wavefrontEdge[numNewPaths] = edgeIdx;
				numNewPaths++;
				if (numNewPaths > mpp.pumpSearch.maxPathCount) {
					return;
				}
			}
		}

		if (numNewPaths == 0) {
			std::cout << "no new valid paths" << std::endl;
			break;
		}

		if (numPaths + numNewPaths >= mpp.pumpSearch.maxPathCount) {
			std::cout << "mpp.pumpSearch.maxPathCount reached, increase max number of paths" << std::endl;
			return;
		}
		
		mpp.pumpSearch.sizeG[Gidx] = 0; // reset G size

		// copy necessary info to GPU
		cudaDeviceSynchronize();
		CUDA_ERROR_CHECK(cudaMemcpy(mpp.pumpSearch.d_wavefrontPathPrev, mpp.pumpSearch.wavefrontPathPrev, sizeof(int)*numNewPaths, cudaMemcpyHostToDevice));
		CUDA_ERROR_CHECK(cudaMemcpy(mpp.pumpSearch.d_wavefrontEdge, mpp.pumpSearch.wavefrontEdge, sizeof(int)*numNewPaths, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		// calculate CP (do half plane checks per particle, then sum)
		// copy over path times
		float *d_debugOutput;
		CUDA_ERROR_CHECK(cudaMalloc(&d_debugOutput, sizeof(float)*numNewPaths*mpp.numHSMCParticles));

		double t_propStart = std::clock();
		numConsideredPaths += numNewPaths;

		const int blockPropCP = 512;
		// call correct propogate for number of paths
		if ((numNewPaths*mpp.numHSMCParticles + blockPropCP - 1) / blockPropCP < 2147483647) {
			const int gridPropCP = std::min(
				(numNewPaths*mpp.numHSMCParticles + blockPropCP - 1) / blockPropCP, 2147483647);
			// if path centric ^ is numNewPaths, otherwisae it is numNewPaths*mpp.numHSMCParticles
			std::cout << "launching propagateValidParticles (" << blockPropCP << ", " << gridPropCP << ") new paths = " << numNewPaths;
			propagateValidParticles<<<gridPropCP, blockPropCP>>>(
				numNewPaths, mpp.numDisc+1, mpp.pumpSearch.d_wavefrontEdge, mpp.pumpSearch.d_wavefrontPathPrev, numPaths,	 
				mpp.hsmc.d_as, mpp.hsmc.d_bs, mpp.hsmc.d_countStoredHSs, mpp.pumpSearch.d_pathTime, mpp.roadmap.d_toptsEdge, mpp.dt, 
				mpp.hsmc.maxHSCount, mpp.hsmc.d_offsetsTime, mpp.hsmc.offsetMult,
				mpp.numMCParticles, mpp.hsmc.d_pathValidParticles,
				mpp.numHSMCParticles, d_debugOutput);
		} else {
			int numThreadParticles = 2;
			while ((numNewPaths*mpp.numHSMCParticles/numThreadParticles + blockPropCP - 1) / blockPropCP > 2147483647)
				numThreadParticles *= 2;

			const int gridPropCP = std::min(
				(numNewPaths*mpp.numHSMCParticles/numThreadParticles + blockPropCP - 1) / blockPropCP, 2147483647);
			std::cout << "launching propagateValidParticlesMany (" << blockPropCP << ", " << gridPropCP << ")" << " and numThreadParticles = " << numThreadParticles;
			propagateValidParticlesMany<<<gridPropCP, blockPropCP>>>(
				numNewPaths, mpp.numDisc+1, mpp.pumpSearch.d_wavefrontEdge, mpp.pumpSearch.d_wavefrontPathPrev, numPaths,	 
				mpp.hsmc.d_as, mpp.hsmc.d_bs, mpp.hsmc.d_countStoredHSs, mpp.pumpSearch.d_pathTime, mpp.roadmap.d_toptsEdge, mpp.dt, 
				mpp.hsmc.maxHSCount, mpp.hsmc.d_offsetsTime, mpp.hsmc.offsetMult, 
				mpp.numMCParticles, mpp.hsmc.d_pathValidParticles,
				mpp.numHSMCParticles, numThreadParticles, d_debugOutput);
		}

		// float debugOutput[numNewPaths*mpp.numHSMCParticles];
		// CUDA_ERROR_CHECK(cudaMemcpy(debugOutput, d_debugOutput, sizeof(float)*numNewPaths*mpp.numHSMCParticles, cudaMemcpyDeviceToHost));
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
			numNewPaths, numPaths, mpp.pumpSearch.d_pathCP, mpp.hsmc.d_pathValidParticles, mpp.numHSMCParticles);

		cudaDeviceSynchronize();
		float t_calcCP = (std::clock() - t_calcCPStart) / (double) CLOCKS_PER_SEC;
		std::cout << ", calc time: " << t_calcCP << " s" << std::endl;

		code = cudaPeekAtLastError();
		if (cudaSuccess != code) { std::cout << "ERROR on calculateCP_HSMC: " << cudaGetErrorString(code) << std::endl; }

		// float debugOutput[numNewPaths*mpp.numHSMCParticles];
		// CUDA_ERROR_CHECK(cudaMemcpy(debugOutput, d_debugOutput, sizeof(float)*numNewPaths*mpp.numHSMCParticles, cudaMemcpyDeviceToHost));
		// std::cout << "debugOutput = "; printArray(debugOutput,1,200,std::cout);

		// int npp = 20;
		// bool validParticles[mpp.numHSMCParticles*npp];
		// cudaMemcpy(validParticles, mpp.hsmc.d_pathValidParticles, sizeof(bool)*mpp.numHSMCParticles*npp, cudaMemcpyDeviceToHost);
		// std::cout << "valid particles"; printArray(validParticles, npp, mpp.numHSMCParticles, std::cout);

		// copy back CP, fill in CP, time, cost
		CUDA_ERROR_CHECK(cudaMemcpy(mpp.pumpSearch.pathCP + numPaths, mpp.pumpSearch.d_pathCP + numPaths, sizeof(float)*numNewPaths, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		// std::cout << "path CP = "; printArray(mpp.pumpSearch.pathCP,1,20,std::cout);

		for (int i = 0; i < numNewPaths; ++i) {
			mpp.pumpSearch.pathTime[numPaths + i] = mpp.pumpSearch.pathTime[mpp.pumpSearch.wavefrontPathPrev[i]] + mpp.roadmap.toptsEdge[mpp.pumpSearch.wavefrontEdge[i]];
			mpp.pumpSearch.pathCost[numPaths + i] = mpp.pumpSearch.pathCost[mpp.pumpSearch.wavefrontPathPrev[i]] + mpp.roadmap.coptsEdge[mpp.pumpSearch.wavefrontEdge[i]]; // cost = time currently
		}

		// ************************************** dominance check ************************************** 
		// load all new nodes into mpp.pumpSearch.P
		int PnewCount[NUM];
		for (int i = 0; i < NUM; ++i)
			PnewCount[i] = 0;
		for (int i = 0; i < numNewPaths; ++i) {
			int nodeIdx = mpp.pumpSearch.wavefrontNodeNext[i];
			int pathIdx = numPaths + i;

			if (mpp.pumpSearch.pathCP[pathIdx] > cpCutoff) { // CP cutoff reached
				mpp.pumpSearch.wavefrontNodeNext[i] = -1;
				continue;
			}

			mpp.pumpSearch.P[nodeIdx*NUM + mpp.pumpSearch.Pcounts[nodeIdx] + PnewCount[nodeIdx]] = pathIdx;
			PnewCount[nodeIdx]++;
		}

		// check new paths against stored paths
		for (int i = 0; i < numNewPaths; ++i) {
			int nodeIdx = mpp.pumpSearch.wavefrontNodeNext[i];
			// already eliminated or at the goal idx
			if (mpp.pumpSearch.wavefrontNodeNext[i] == -1 || nodeIdx == mpp.goalIdx) // don't check goal
				continue;
			int pathIdx = numPaths + i;

			for (int j = 0; j < mpp.pumpSearch.Pcounts[nodeIdx] + PnewCount[nodeIdx]; ++j) {
				int pathIdxCompare = mpp.pumpSearch.P[nodeIdx*NUM + j];
				// don't compare to self
				if (pathIdxCompare == pathIdx)
					continue;

				// eps comparison
				if (mpp.pumpSearch.pathCP[pathIdxCompare] < mpp.epsCP + mpp.pumpSearch.pathCP[pathIdx] && 
					mpp.pumpSearch.pathCost[pathIdxCompare] < mpp.epsCost + mpp.pumpSearch.pathCost[pathIdx]) {
					
					// check if paths are co-dominant, then keep the one with a lower path number
					if (mpp.pumpSearch.pathCP[pathIdxCompare] + mpp.epsCP >= mpp.pumpSearch.pathCP[pathIdx] && 
						mpp.pumpSearch.pathCost[pathIdxCompare] + mpp.epsCost >= mpp.pumpSearch.pathCost[pathIdx] &&
						pathIdx < pathIdxCompare) {
						// std::cout << "co dom: (" << mpp.pumpSearch.pathCP[pathIdxCompare] << ", " << mpp.pumpSearch.pathCost[pathIdxCompare] << ") vs ( " 
						// 	<< mpp.pumpSearch.pathCP[pathIdx] << ", " << mpp.pumpSearch.pathCost[pathIdx] << ")" << std::endl;
						continue;
					}

					mpp.pumpSearch.wavefrontNodeNext[i] = -1; // mark for removal
					break; 
				}
			}
		}

		// ************************************** store good paths ************************************** 
		int numValidNewPaths = 0;
		for (int i = 0; i < numNewPaths; ++i) {
			int nodeIdx = mpp.pumpSearch.wavefrontNodeNext[i];
			int pathIdx = numPaths + i;

			// TODO: if i break here, decrement path index and path count
			if (mpp.pumpSearch.wavefrontNodeNext[i] == -1 || mpp.pumpSearch.wavefrontNodeNext[i] == 0) { // either node is at the init or marked as bad
				mpp.pumpSearch.wavefrontNodeNext[i] = -1; // clear
				continue;
			}

			int pathIdxStore = numPaths + numValidNewPaths;
			mpp.pumpSearch.pathCP[pathIdxStore] = mpp.pumpSearch.pathCP[pathIdx];
			mpp.pumpSearch.pathTime[pathIdxStore] = mpp.pumpSearch.pathTime[pathIdx];
			mpp.pumpSearch.pathCost[pathIdxStore] = mpp.pumpSearch.pathCost[pathIdx];
			CUDA_ERROR_CHECK(cudaMemcpy(mpp.hsmc.d_pathValidParticles + mpp.numHSMCParticles*pathIdxStore,
				mpp.hsmc.d_pathValidParticles + mpp.numHSMCParticles*pathIdx, sizeof(bool)*mpp.numHSMCParticles, cudaMemcpyDeviceToDevice));

			float dCost = (mpp.pumpSearch.pathCost[pathIdxStore] + mpp.roadmap.h[nodeIdx] - costThreshold);
			dCost = std::max((float) 0, dCost); // if below zero, put into the next bucket
			int bucketIdx = (((int) (dCost / dr)) + Gidx + 1) % mpp.numBuckets;
			mpp.pumpSearch.G[bucketIdx*mpp.pumpSearch.maxPathCount + mpp.pumpSearch.sizeG[bucketIdx]] = pathIdxStore;
			mpp.pumpSearch.sizeG[bucketIdx]++;
			mpp.pumpSearch.pathPrev[pathIdxStore] = mpp.pumpSearch.wavefrontPathPrev[i];
			mpp.pumpSearch.pathNode[pathIdxStore] = mpp.pumpSearch.wavefrontNodeNext[i];

			mpp.pumpSearch.P[nodeIdx*NUM + mpp.pumpSearch.Pcounts[nodeIdx]] = pathIdxStore;
			mpp.pumpSearch.Pcounts[nodeIdx]++;
			mpp.pumpSearch.wavefrontNodeNext[i] = -1; // clear, TODO: uncessary, but nice for debugging purposes to have a fresh array

			numValidNewPaths++;
		}
		CUDA_ERROR_CHECK(cudaMemcpy(mpp.pumpSearch.d_pathCP + numPaths, mpp.pumpSearch.pathCP + numPaths, sizeof(float)*numValidNewPaths, cudaMemcpyHostToDevice));
		CUDA_ERROR_CHECK(cudaMemcpy(mpp.pumpSearch.d_pathTime + numPaths, mpp.pumpSearch.pathTime + numPaths, sizeof(float)*numValidNewPaths, cudaMemcpyHostToDevice));

		numPaths += numValidNewPaths;

		// update goal condition
		for (int i = 0; i < mpp.pumpSearch.Pcounts[mpp.goalIdx]; ++i) {
			int pathIdx = mpp.pumpSearch.P[mpp.goalIdx*NUM + i];
			if (mpp.pumpSearch.pathCP[pathIdx] <= cpMinSoln) {
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
		for (int b = 0; b < mpp.numBuckets; ++b)
			emptyOpenSet = emptyOpenSet && (mpp.pumpSearch.sizeG[b] == 0);
		if (emptyOpenSet) {
			std::cout << "emptyOpenSet met" << std::endl;
			break;
		}

		// update G index
		Gidx = (Gidx+1) % mpp.numBuckets;
		costThreshold += dr;

		// end and send out warning if mpp.pumpSearch.maxPathCount is exceeded
		if (numPaths >= mpp.pumpSearch.maxPathCount) {
			std::cout << "mpp.pumpSearch.maxPathCount reached, increase max number of paths" << std::endl;
			return;
		}
		
	}
	// std::cout << "path CP = "; printArray(mpp.pumpSearch.pathCP,1,numPaths,std::cout);
	float t_ccgmt = (std::clock() - t_ccgmtStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Explore took: " << t_ccgmt << " s" << std::endl;
	std::cout << "number of goal paths found = " << mpp.pumpSearch.Pcounts[mpp.goalIdx] << " with a total number of paths = " << numPaths << " of " << numConsideredPaths << std::endl;

	// ************************************** post exploration loop ************************************** 
	// load mpp.roadmap.samples into array
	std::vector<float> xs;

	int pathIdx;
	int nodeIdx;
	int pathNumSamples = 0;

	float splitPath[DIM*Tmax]; // empty array for the optimal path
	int pathLength = 0;

	float *d_path;
	CUDA_ERROR_CHECK(cudaMalloc(&d_path, sizeof(float)*Tmax*DIM));
	float *d_obstacles;
	CUDA_ERROR_CHECK(cudaMalloc(&d_obstacles, sizeof(float)*2*mpp.numObstacles*DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_obstacles, mpp.obstacles.data(), sizeof(float)*2*mpp.numObstacles*DIM, cudaMemcpyHostToDevice));

	std::cout << "*********************** output paths and check cp heursitic ********************" << std::endl;
	for (int i = 0; i < mpp.pumpSearch.Pcounts[mpp.goalIdx]; ++i) {
		xs.clear();
		int solnPathIdx = mpp.pumpSearch.P[mpp.goalIdx*NUM + i];
		pathIdx = solnPathIdx;
		nodeIdx = mpp.pumpSearch.pathNode[pathIdx];
		pathNumSamples = 0;
		while (pathIdx != -2) {
			++pathNumSamples;
			for (int d = DIM-1; d >= 0; --d)
				xs.insert(xs.begin(), mpp.roadmap.samples[nodeIdx*DIM+d]);
			pathIdx = mpp.pumpSearch.pathPrev[pathIdx];
			nodeIdx = mpp.pumpSearch.pathNode[pathIdx];
		}
		// printArray(&(xs[0]),pathNumSamples,DIM,std::cout);
		// std::cout << "path has " << int(xs.size()) << " elements" << std::endl;

		// solve 2pbvp
		pathLength = 0;
		findOptimalPath(mpp.dt, splitPath, &(xs[0]), pathNumSamples, &pathLength); // ignoring return of topt

		CUDA_ERROR_CHECK(cudaMemcpy(d_path, splitPath, sizeof(float)*Tmax*DIM, cudaMemcpyHostToDevice));
		
		float actualCP = collisionProbability(d_obstacles, mpp.numObstacles, mpp.hsmc.d_offsets, mpp.hsmc.offsetMult, d_path, pathLength);
		std::cout << "Path " << solnPathIdx << ": cost = " << mpp.pumpSearch.pathCost[solnPathIdx] << 
			", time = " << mpp.pumpSearch.pathTime[solnPathIdx] << 
			", cp = " << actualCP << " vs " << mpp.pumpSearch.pathCP[solnPathIdx] << " est, ";
		pathIdx = solnPathIdx;
		std::cout << "nodes = [" << mpp.pumpSearch.pathNode[pathIdx];
		while (mpp.pumpSearch.pathPrev[pathIdx] != -2) {
			pathIdx = mpp.pumpSearch.pathPrev[pathIdx];
			std::cout << ", " << mpp.pumpSearch.pathNode[pathIdx];
		}
		std::cout << "]"<< std::endl;
	}

	// ************************************** search for best path ************************************** 
	// not currently implemented as bisection, 
	// but rather choose best path's by cost and check if cp condition is met
	// put paths in a proirity queue by cost

	float t_searchStart = std::clock();
	int goalPathsCount = mpp.pumpSearch.Pcounts[mpp.goalIdx];
	std::vector< std::pair<float, int> > goalPaths;
	for (int i = 0; i < goalPathsCount; ++i) {
		int pathIdx = mpp.pumpSearch.P[mpp.goalIdx*NUM + i];
		std::pair<float, int> pathPair(mpp.pumpSearch.pathCost[pathIdx], pathIdx);
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
		nodeIdx = mpp.pumpSearch.pathNode[pathIdx];
		pathNumSamples = 0;
		while (pathIdx != -2) {
			++pathNumSamples;
			for (int d = DIM-1; d >= 0; --d)
				xs.insert(xs.begin(), mpp.roadmap.samples[nodeIdx*DIM+d]);
			pathIdx = mpp.pumpSearch.pathPrev[pathIdx];
			nodeIdx = mpp.pumpSearch.pathNode[pathIdx];
		}

		pathLength = 0;
		findOptimalPath(mpp.dt, splitPath, &(xs[0]), pathNumSamples, &pathLength); // ignoring return of topt

		CUDA_ERROR_CHECK(cudaMemcpy(d_path, splitPath, sizeof(float)*Tmax*DIM, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		solnPathCP = collisionProbability(d_obstacles, mpp.numObstacles, mpp.hsmc.d_offsets, mpp.hsmc.offsetMult, d_path, pathLength);
		solnPathCost = mpp.pumpSearch.pathCost[solnPathIdx];
		cudaDeviceSynchronize();
		searchItrs++;

		std::cout << "tried path " << solnPathIdx << ", cost = " << mpp.pumpSearch.pathCost[solnPathIdx] << ", cp = " << solnPathCP << std::endl;
		if (solnPathCP < mpp.cpTarget) {
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
		float optPathTopt = findOptimalPath(mpp.dt/dtMult, optPath, &(xsOpt[0]), 2, &optPathLength);

		// std::cout << " optimal path is: " << std::endl;
		// printArray(optPath,optPathLength,DIM,std::cout);

		std::cout << " from the original nodes: " << std::endl;
		for ( int i = 0; i < xs.size(); i++) {
            std::cout << xs[i] << " ";
            if ((i + 1) % 6 == 0)
            	std::cout << std::endl;
        }

		// find the path points that map to the nominal path
		std::vector<float> splitPathTopts(pathNumSamples,0);
		std::vector<int> splitPathIdxs(pathNumSamples,0);
		for (int i = 0; i < pathNumSamples-1; ++i)
			splitPathTopts[i+1] = splitPathTopts[i]+toptBisection(&xs[i*DIM], &xs[(i+1)*DIM], 2);
		for (int i = 0; i < pathNumSamples; ++i)
			splitPathIdxs[i] = (optPathLength-1)*splitPathTopts[i]/splitPathTopts[splitPathTopts.size()-1];

		// print out waypoint matches for the smoothing process
		// std::cout << "found optimal times as: ";
		// for ( int i = 0; i < splitPathTopts.size(); i++) {
  //           std::cout << splitPathTopts[i] << " ";
  //       }
  //       std::cout << std::endl;
  //       std::cout << "found indexes as: ";
		// for ( int i = 0; i < splitPathIdxs.size(); i++) {
  //           std::cout << splitPathIdxs[i] << " ";
  //       }
  //       std::cout << " of " << optPathLength;

        xsOpt.clear();
        // std::cout << " means we match: " << std::endl;
        for ( int i = 0; i < pathNumSamples; i++) {
        	// printArray(&xs[i*DIM],1,DIM,std::cout);
        	// std::cout << " with "; printArray(&optPath[splitPathIdxs[i]*DIM],1,DIM,std::cout);
        	for (int d = 0; d < DIM; ++d)
        		xsOpt.push_back(optPath[splitPathIdxs[i]*DIM+d]);
        }

        // printing the optimal unconstrained path 
        // std::cout << "verify creation of xsOpt: " << std::endl;
        // for ( int i = 0; i < xsOpt.size(); i++) {
        //     std::cout << xsOpt[i] << " ";
        //     if ((i + 1) % 6 == 0)
        //     	std::cout << std::endl;
        // }

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
			findOptimalPath(mpp.dt, smoothPath, &(xsSmooth[0]), pathNumSamples, &smoothPathLength); // ignoring return of topt
			for (int i = 0; i < pathNumSamples-1; ++i) {
				float tau = toptBisection(&xsSmooth[i*DIM], &xsSmooth[(i+1)*DIM], 2);
				smoothPathCost += cost(tau, &xsSmooth[i*DIM], &xsSmooth[(i+1)*DIM]);
			}

			CUDA_ERROR_CHECK(cudaMemcpy(d_path, smoothPath, sizeof(float)*Tmax*DIM, cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();

			solnPathCP = collisionProbability(d_obstacles, mpp.numObstacles, mpp.hsmc.d_offsets, mpp.hsmc.offsetMult, d_path, smoothPathLength);
			cudaDeviceSynchronize();

			mpp.verbose && std::cout << "-------------- iteration " << smoothItrs << ", alpha = " << alpha << ", cp = " << solnPathCP << ", cost = " << smoothPathCost << std::endl;

			if (solnPathCP < mpp.cpTarget)
				alphaL = alpha;
			if (solnPathCP >= mpp.cpTarget)
				alphaU = alpha;
			alpha = (alphaL + alphaU)/2;

			// go for path closest to the cp limit
			bool satisfiesCP = solnPathCP < mpp.cpTarget;
			bool closerToCP = solnPathCP > lastSmoothBelowCP;
			bool equalCPandLowerCost = (solnPathCP == lastSmoothBelowCP && smoothPathCost < lastSmoothBelowCost);
			if (satisfiesCP && (closerToCP || equalCPandLowerCost))
			{
				mpp.verbose && std::cout << "NEW BEST PATH!" << std::endl;	

				copyArray(lastSmoothBelowPath, smoothPath, DIM*Tmax);
				lastSmoothBelowPathLength = smoothPathLength;
				std::vector<float> xsLastSmoothBelow(xs.size());
				std::copy(xsSmooth.begin(), xsSmooth.end(), xsLastSmoothBelow.begin());
				lastSmoothBelowCost = smoothPathCost;
				lastSmoothBelowCP = solnPathCP;
			}

			if (std::abs(mpp.cpTarget - solnPathCP) < epsSmoothCP &&
				mpp.cpTarget - solnPathCP > 0) {
				std::cout << "breaking" << std::endl;
				break;
			}

			++smoothItrs;
		}

		std::cout << "smoothing gave a cost savings of " << 100-100*lastSmoothBelowCost/solnPathCost << "%" << std::endl;
		solnPathCost = lastSmoothBelowCost;
	}
	std::cout << "final smoothed path is: " << std::endl;
    for (int i = 0; i < xsSmooth.size(); i++) {
        std::cout << xsSmooth[i] << " ";
        if ((i + 1) % 6 == 0)
        	std::cout << std::endl;
    }
    
    // ******* discretize path for controller
 	float controlDt = 0.05; // dt for the discretization to give to the controller
 	float smoothedPathTime = 0;
 	int discretizedPathLength = 1;
 	for (int i = 0; i < pathNumSamples-1; ++i) {
    	float tau = toptBisection(&xsSmooth[i*DIM], &xsSmooth[(i+1)*DIM], 2);
    	smoothedPathTime += tau;
    	discretizedPathLength += tau/controlDt + 1;
    }
    std::vector<float> discretizedPath(discretizedPathLength*DIM);
 	int currentNumWaypoints = 0;
    for (int i = 0; i < pathNumSamples-1; ++i) {
    	float tau = toptBisection(&xsSmooth[i*DIM], &xsSmooth[(i+1)*DIM], 2);
    	int numWaypoints = tau/controlDt + 1;
    	findDiscretizedPath(&(discretizedPath[currentNumWaypoints*DIM]), &xsSmooth[i*DIM], &xsSmooth[(i+1)*DIM], numWaypoints);
    	currentNumWaypoints += numWaypoints;
    }

    std::ofstream discretizedPathStream;
	discretizedPathStream.open("discretizedPath.txt");
    for (int i = 0; i < discretizedPath.size(); i++) {
        discretizedPathStream << discretizedPath[i] << ", ";
        if ((i + 1) % 6 == 0)
        	discretizedPathStream << std::endl;
    }
    discretizedPathStream.close();


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
	int maxHSCount, float *offsets,  float offsetMult, int numMCParticles, bool *validParticles,
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
	int hsHeadIdx = edgeIdx*waypointsCount*maxHSCount;
	for (int i = 0; i < waypointsCount; ++i) {
		float time = times[pathIdxPrev] + i*dtLocal;
		int dtIdx = time/dt; // TODO: make this round, not floor

		// countStoredHSs[edgeIdx*waypointsCount+i];
		for (int o = 0; o < maxHSCount; ++o) { // loop through obstacles
			if (bs[hsHeadIdx + i*maxHSCount + o] >= 1000)
				continue; 

			float LHS = 0; // offset*a
			for (int d = 0; d < DIM/2; ++d) {
				LHS += offsets[dtIdx*numMCParticles*DIM/2 + particleIdx*DIM/2 + d]*   //particleIdx*Tmax*DIM/2 + dtIdx*DIM/2 +d]*
					as[hsHeadIdx*DIM/2 + i*maxHSCount*DIM/2 + o*DIM/2 + d];
			}
			validParticle &=
				(LHS*offsetMult < bs[hsHeadIdx + i*maxHSCount + o]);
		}
	}
	validParticles[pathIdx*numHSMCParticles + particleIdx] = validParticle;
}

// propogate valid particles for new halfspaces along the edge
__global__ void propagateValidParticlesMany(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, int *countStoredHSs, float *times, float *topts, float dt,
	int maxHSCount, float *offsets, float offsetMult, int numMCParticles, bool *validParticles,
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
		int hsHeadIdx = edgeIdx*waypointsCount*maxHSCount;

		// validParticles[pathIdx*numHSMCParticles + particleIdx] = validParticles[pathIdxPrev*numHSMCParticles + particleIdx];
		bool validParticle = validParticles[pathIdxPrev*numHSMCParticles + particleIdx];

		float dtLocal = topts[edgeIdx]/(waypointsCount-1);
		// loop through waypoints
		for (int i = 0; i < waypointsCount; ++i) {
			float time = times[pathIdxPrev] + i*dtLocal;
			int dtIdx = time/dt; // TODO: make this round, not floor

			// countStoredHSs[edgeIdx*waypointsCount + i]
			for (int o = 0; o < maxHSCount; ++o) { // loop through obstacles
				if (bs[hsHeadIdx + i*maxHSCount + o] >= 1000)
					continue; // TODO: or continue if using make all hs's

				float LHS = 0; // offset*a
				for (int d = 0; d < DIM/2; ++d) {
					LHS += offsets[dtIdx*numMCParticles*DIM/2 + particleIdx*DIM/2 + d]*   //particleIdx*Tmax*DIM/2 + dtIdx*DIM/2 +d]*
						as[hsHeadIdx*DIM/2 + i*maxHSCount*DIM/2 + o*DIM/2 + d];
				}
				validParticle &=
					(LHS*offsetMult < bs[hsHeadIdx + i*maxHSCount + o]);
			}
		}
		validParticles[pathIdx*numHSMCParticles + particleIdx] = validParticle;
	}
}

__global__ void propagateValidParticlesPathCentric(int numNewPaths, int waypointsCount,
	int *wavefrontEdge, int *wavefrontPathPrev, int pathBaseIdx,	 
	float *as, float *bs, float *times, float *topts, float dt,
	int maxHSCount, float *offsets, int numMCParticles, bool *validParticles,
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

			for (int o = 0; o < maxHSCount; ++o) { // loop through obstacles
				float LHS = 0; // offset*a
				for (int d = 0; d < DIM/2; ++d) {
					LHS += offsets[p*DIM/2*Tmax + dtIdx*DIM/2 + d]* //  [dtIdx*numMCParticles*DIM + p*DIM + d]*
						as[edgeIdx*(waypointsCount)*maxHSCount*DIM/2 + i*maxHSCount*DIM/2 + o*DIM/2 + d];
				}
				validParticles[pathIdx*numHSMCParticles + p] &=
					(LHS < bs[edgeIdx*waypointsCount*maxHSCount + i*maxHSCount + o]);
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