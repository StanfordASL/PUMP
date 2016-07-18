/*
testInflation.cu
author: Brian Ichter

This code is meant for running the GMT* motion planning algorithm on a Jetson TX1.
Specifically, it's model is a quadrotor with some precomputed data.

TODO: I should check for errors in kernel calls and memcpys
*/

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <algorithm> 
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>

#include "helper.cuh"
#include "2pBVP.cuh"
#include "obstacles.cuh"
#include "sampler.cuh"
#include "precomp.cuh"
#include "collisionProbability.cuh"
#include "hardCoded.cuh"
#include "GMT.cuh"

/******************************************
Run parameters
******************************************/

#ifndef DIM
#error Please define DIM.
#endif
#ifndef NUM
#error Please define NUM.
#endif
#ifndef RUNS
#error Please define RUNS.
#endif

float lambda = 1.0; // adjusts wavefront size, R(0,1]
float initial_loc = 0.1;
float goal_loc = 0.9; // and the y dimension updated below
bool activeKernel = true; // false = map cores to all samples, true = map cores to only active samples (requires a sort and find op)
float lo[DIM] = {0, 0, 0, -1, -1, -1};
float hi[DIM] = {1, 1, 1, 1, 1, 1};
int numDisc = 8; // number of discretizations of kinodynamic paths
float offsetMult = 0.8; 

float dt = 0.05;

/******************************************
Global memory
******************************************/

/******************************************
Solution Data Struct and functions
******************************************/

int main()
{
	if(DIM != 6) {
		std::cout << "DIM MUST BE 6, this solves the 3D double integrator only!" << std::endl;
		return EXIT_FAILURE;
	}

	int count = 0;
	cudaGetDeviceCount(&count);

	std::cout << "New run: DIM = " << DIM << ", NUM = " << NUM << ", RUNS = " << RUNS << std::endl;

    /***********************
	setup data (initial/goal states, obstacles, samples, etc.)
	***********************/

	// timing data
	double t_overall(0), t_sampleFree(0), t_setup(0), t_gmt(0);
	double ms = 1000;
	double t_overallStart = std::clock();

	// generate initial and goal states
	std::vector<float> initial(DIM, initial_loc);
	std::vector<float> goal(DIM, goal_loc);
	initial[0] = 0.7;
	goal[0] = 0.7;
	for (int d = DIM/2; d < DIM; ++d) {
	 	initial[d] = 0;
	 	goal[d] = 0;
	}

	/*********************** generate obstacles *************************************************/
	int obstaclesCount = getObstaclesCount();
	float obstacles[obstaclesCount*2*DIM];
	generateObstacles(obstacles, obstaclesCount*2*DIM);
	printArray(obstacles, obstaclesCount, 2*DIM, std::cout);

	// load obstacles on device
	float *d_obstacles;
	cudaMalloc(&d_obstacles, sizeof(float)*2*obstaclesCount*DIM);
	cudaMemcpy(d_obstacles, obstacles, sizeof(float)*2*obstaclesCount*DIM, cudaMemcpyHostToDevice);

	// entire runs loop

	/*********************** create array to return debugging information ***********************/
	float *d_debugOutput;
	cudaMalloc(&d_debugOutput, sizeof(float)*NUM);

	/*********************** generate samples ***************************************************/
	int seed = 1;
	float samplesAll[DIM*NUM];
	createSamplesIID(seed, samplesAll, &(initial[0]), &(goal[0]), lo, hi);
	// printArray(samplesAll,20,DIM,std::cout);

	// set number of nn
	// float r = calculateConnectionBallRadius(DIM, NUM);
	// float r2 = r*r;
	// std::cout << " r is " << r << std::endl;

	// float *d_samplesAll;
	// cudaMalloc(&d_samplesAll, sizeof(float)*DIM*NUM);
	// cudaMemcpy(d_samplesAll, samplesAll, sizeof(float)*DIM*NUM, cudaMemcpyHostToDevice);
	thrust::device_vector<float> d_samples_thrust(DIM*NUM);
	float *d_samples = thrust::raw_pointer_cast(d_samples_thrust.data());
	cudaMemcpy(d_samples, samplesAll, sizeof(float)*DIM*NUM, cudaMemcpyHostToDevice);

	// const int blockSizeNNSize = 192;
	// const int gridSizeNNSize = std::min((NUM + blockSizeNNSize - 1) / blockSizeNNSize, 65535);
	// thrust::device_vector<int> d_nnSizes(NUM);
	// int *d_nnSizes_ptr = thrust::raw_pointer_cast(d_nnSizes.data());
	// calculateNumberNN<<<gridSizeNNSize,blockSizeNNSize>>>(r2, d_samples, d_nnSizes_ptr, NUM);
	// int maxNNIndex = thrust::max_element(d_nnSizes.begin(), d_nnSizes.end()) - d_nnSizes.begin();
	// const int nnSize = d_nnSizes[maxNNIndex]; // array size for nn 
	// std::cout << "Number of nn is " << nnSize << std::endl;

	/*********************** find 2pbvp 10th quantile *********************************************/
	double t_calc10thStart = std::clock();
	std::vector<float> topts ((NUM-1)*(NUM));
	int percentile = 30;
	int rnIdx = (NUM-1)*(NUM)/percentile; // 10th quartile and number of NN
	std::cout << "rnIdx is " << rnIdx << std::endl;
	int idx = 0;
	float tmax = 5;
	for (int i = 0; i < NUM; ++i) {
		for (int j = 0; j < NUM; ++j) {
			if (j == i)
				continue;
			topts[idx] = toptBisection(&(samplesAll[i*DIM]), &(samplesAll[j*DIM]), tmax);
			idx++;
		}
	}
	std::vector<float> toptsSorted ((NUM-1)*(NUM));
	toptsSorted = topts;
	std::sort (toptsSorted.begin(), toptsSorted.end());
	float rn = toptsSorted[rnIdx];
	double t_calc10th = (std::clock() - t_calc10thStart) / (double) CLOCKS_PER_SEC;
	std::cout << percentile << "th quartile pre calc took: " << t_calc10th*ms << " ms for " << idx << " solves and cutoff is " 
		<< rn << " at " << rnIdx << std::endl;	
	// for (std::vector<float>::const_iterator i = topts.begin(); i != topts.begin()+1000; ++i)
 //    	std::cout << *i << ' ';

	double t_2pbvpTestStart = std::clock();
	// int numDisc = 4;
	float x0[DIM], x1[DIM];
	// float topt;

	double t_discMotionsStart = std::clock();
	// int nnIdxs[NUM*NUM];
	std::vector<int> nnIdxs(NUM*NUM);
	int nnGoSizes[NUM];
	int nnComeSizes[NUM];
	for (int i = 0; i < NUM; ++i) {
		nnGoSizes[i] = 0;
		nnComeSizes[i] = 0;
	}
	std::vector<float> discMotions (rnIdx*(numDisc+1)*DIM,0); // array of motions, but its a vector who idk for some reason it won't work as an array
	int nnIdx = 0; // index position in NN discretization array
	idx = 0; // index position in topts vector above

	for (int i = 0; i < NUM; ++i) {
		for (int d = 0; d < DIM; ++d)
			x0[d] = samplesAll[d + DIM*i];
		for (int j = 0; j < NUM; ++j) {
			if (j == i)
				continue;
			if (topts[idx] < rn) {
				for (int d = 0; d < DIM; ++d)
					x1[d] = samplesAll[d + DIM*j];
				nnIdxs[j*NUM+i] = nnIdx; // look up for discrete motions from i -> j
				findDiscretizedPath(&(discMotions[nnIdx*DIM*(numDisc+1)]), x0, x1, numDisc); // TODO: give topt
				nnGoSizes[i]++;
				nnComeSizes[j]++;
				nnIdx++;
			}
			idx++;
		}
	}
	double t_discMotions = (std::clock() - t_discMotionsStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Discretizing motions took: " << t_discMotions*ms << " ms for " << nnIdx << " solves" << std::endl;	

	int maxNNSize = 0;
	for (int i = 0; i < NUM; ++i) {
		if (maxNNSize < nnGoSizes[i])
			maxNNSize = nnGoSizes[i];
		if (maxNNSize < nnComeSizes[i])
			maxNNSize = nnComeSizes[i];
	}
	std::cout << "max number of nn is " << maxNNSize << std::endl;

	std::vector<float> distancesCome (NUM*maxNNSize, 0);
	std::vector<int> nnGoEdges (NUM*maxNNSize, -1); // edge gives indices (i,j) to check nnIdx to then find the discretized path
	std::vector<int> nnComeEdges (NUM*maxNNSize, -1); // edge gives indices (j,i) to check nnIdx to then find the discretized path
	idx = 0;
	for (int i = 0; i < NUM; ++i) {
		nnGoSizes[i] = 0; // clear nnSizes again
		nnComeSizes[i] = 0; // clear nnSizes again
	}
	for (int i = 0; i < NUM; ++i) {
		for (int j = 0; j < NUM; ++j) {
			if (j == i)
				continue;
			if (topts[idx] < rn) {
				nnGoEdges[i*maxNNSize + nnGoSizes[i]] = j;
				nnComeEdges[j*maxNNSize + nnComeSizes[j]] = i;
				distancesCome[j*maxNNSize + nnComeSizes[j]] = topts[idx];
				nnGoSizes[i]++;
				nnComeSizes[j]++;
			}
			idx++;
		}
	}	

	// std::cout << "nnIdxs: " << std::endl;
	// printArray(&nnIdxs[0], 10, NUM, std::cout);
	// std::cout << "nnEdges: " << std::endl;
	// printArray(&nnEdges[0], 5, maxNNSize, std::cout);
	// std::cout << "discMotions: " << std::endl;
	// printArray(&discMotions[0], 5, (numDisc+1)*DIM, std::cout);
	/*********************** precomputation of nn *********************************************/
	float *d_discMotions;
	cudaMalloc(&d_discMotions, sizeof(float)*rnIdx*(numDisc+1)*DIM);
	cudaMemcpy(d_discMotions, &discMotions[0], sizeof(float)*rnIdx*(numDisc+1)*DIM, cudaMemcpyHostToDevice);

	int *d_nnIdxs;
	cudaMalloc(&d_nnIdxs, sizeof(int)*NUM*NUM);
	cudaMemcpy(d_nnIdxs, &nnIdxs[0], sizeof(int)*NUM*NUM, cudaMemcpyHostToDevice);

	float *d_distancesCome;
	cudaMalloc(&d_distancesCome, sizeof(float)*NUM*maxNNSize);
	cudaMemcpy(d_distancesCome, &distancesCome[0], sizeof(float)*NUM*maxNNSize, cudaMemcpyHostToDevice);

	int *d_nnGoEdges;
	cudaMalloc(&d_nnGoEdges, sizeof(int)*NUM*maxNNSize);
	cudaMemcpy(d_nnGoEdges, &nnGoEdges[0], sizeof(int)*NUM*maxNNSize, cudaMemcpyHostToDevice);

	int *d_nnComeEdges;
	cudaMalloc(&d_nnComeEdges, sizeof(int)*NUM*maxNNSize);
	cudaMemcpy(d_nnComeEdges, &nnComeEdges[0], sizeof(int)*NUM*maxNNSize, cudaMemcpyHostToDevice);
	
	/***********************
	run GPU code
	***********************/
	/*********************** setup cost and edge arrays *****************************************/
	double t_setupStart = std::clock();
	float *d_costs;
	cudaMalloc(&d_costs, sizeof(float)*NUM);
	thrust::device_vector<int> d_edges(NUM);
	int* d_edges_ptr = thrust::raw_pointer_cast(d_edges.data());

	// print statement variables

	t_setup = (std::clock() - t_setupStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Setup took: " << t_setup*ms << " ms" << std::endl;

	// combined state uncertainty (loaded in from hardCoded.cu)
	float *d_xcomb;
	cudaMalloc(&d_xcomb, sizeof(float)*Tmax*numMCSamples*DIM);
	cudaMemcpy(d_xcomb, xcomb, sizeof(float)*Tmax*numMCSamples*DIM, cudaMemcpyHostToDevice);
	float *d_path;
	cudaMalloc(&d_path, sizeof(float)*Tmax*DIM);

	/***********************
	// begin inflate obstacles here 
	***********************/
	float cp = 1;
	int itrs = 0;
	float obstaclesInflated[obstaclesCount*2*DIM];
	float *d_obstaclesInflated;
	cudaMalloc(&d_obstaclesInflated, sizeof(float)*2*DIM*obstaclesCount);
	thrust::device_vector<bool> d_isFreeSamples(NUM);
	bool* d_isFreeSamples_ptr = thrust::raw_pointer_cast(d_isFreeSamples.data());

	double t_inflateStart = std::clock();

	float dObsInfl = 0.005;
	int numInflItrs = 30;
	std::vector<float> inflateFactors(numInflItrs);
	std::vector<float> costsInfl(numInflItrs);
	std::vector<float> cps(numInflItrs);

	// output to matlab readable file
	std::ofstream matlabData;
	matlabData.open ("matlabInflationData.txt");
	matlabData << "obstacles = ["; 
	printArray(obstacles, 2*obstaclesCount, DIM, matlabData); 
	matlabData << "];" << std::endl;

	for (int infl = 0; infl < numInflItrs; ++infl)
	{
		double t_obsInfStart = std::clock();
		float inflateFactor = infl*dObsInfl;
		inflateObstacles(obstacles, obstaclesInflated, inflateFactor, obstaclesCount);
		std::cout << "New obstacle set for iteration " << itrs << " are inflated " << inflateFactor << std::endl;
		printArray(obstaclesInflated, obstaclesCount, 2*DIM, std::cout);
		cudaMemcpy(d_obstaclesInflated, obstaclesInflated, sizeof(float)*2*obstaclesCount*DIM, cudaMemcpyHostToDevice);
		// load obstacles on device
		float t_obsInf = (std::clock() - t_obsInfStart) / (double) CLOCKS_PER_SEC;
		std::cout << "Obs inflate took: " << t_obsInf*ms << " ms" << std::endl;

		/*********************** sample free ********************************************************/
		double t_sampleFreeStart = std::clock();
		const int blockSizeSF = 192;
		const int gridSizeSF = std::min((NUM + blockSizeSF - 1) / blockSizeSF, 65535);

		sampleFree<<<gridSizeSF, blockSizeSF>>>(
			d_obstaclesInflated, obstaclesCount, d_samples, d_isFreeSamples_ptr, d_debugOutput);
		cudaDeviceSynchronize();

		int goalIdx = NUM-1;
		int initialIdx = 0;

		t_sampleFree += (std::clock() - t_sampleFreeStart) / (double) CLOCKS_PER_SEC;

		std::cout << "Sample free took: " << t_sampleFree*ms << " ms" << std::endl;

		/********************** call GMT ************************************************************/
		double t_gmtStart = std::clock();
		std::cout << "Running wavefront expansion GMT" << std::endl;
		GMTwavefront(&(initial[0]), &(goal[0]), d_obstaclesInflated, obstaclesCount,
			d_distancesCome, d_nnGoEdges, d_nnComeEdges, maxNNSize, d_discMotions, d_nnIdxs,
			d_samples, NUM, d_isFreeSamples_ptr, rn, numDisc,
			d_costs, d_edges_ptr, initialIdx, goalIdx) ;
	
		t_gmt += (std::clock() - t_gmtStart) / (double) CLOCKS_PER_SEC;
		float costGoal = 0;
		cudaMemcpy(&costGoal, d_costs+goalIdx, sizeof(float), cudaMemcpyDeviceToHost);
		std::cout << "Solution cost: " << costGoal << std::endl;

		/*********************** solve 2pBVP ********************************************************/
		int numWaypoints = 0;
		int currentEdge = goalIdx;
		while (currentEdge >= 0 && ++numWaypoints) {
			currentEdge = d_edges[currentEdge];
		}
		std::cout << "num waypoints = " << numWaypoints << " and solution is: " << std::endl;

		std::vector<float> xs(DIM*numWaypoints);
		currentEdge = goalIdx;
		int edgeItr = 0;
		while (currentEdge >= 0) {
			for (int d = 0; d < DIM; ++d) 
				xs[(numWaypoints-edgeItr-1)*DIM + d] = samplesAll[currentEdge*DIM + d];
			currentEdge = d_edges[currentEdge];
			++edgeItr;
		}
		printArray(&(xs[0]),numWaypoints,DIM,std::cout);

		int pathLength = 0;
		float splitPath[DIM*Tmax]; // empty array for the optimal path

		double t_2pbvpStart = std::clock();
		float topt = findOptimalPath(dt, splitPath, &(xs[0]), numWaypoints, &pathLength);
		double t_2pbvp = (std::clock() - t_2pbvpStart) / (double) CLOCKS_PER_SEC;
		std::cout << "2pbvp took: " << t_2pbvp*ms << " ms and path length is " << pathLength << " for path:" << std::endl;
		
		/*********************** call Collision Probability *****************************************/	
		cudaMemcpy(d_path, splitPath, sizeof(float)*Tmax*DIM, cudaMemcpyHostToDevice);

		int T = pathLength; // will be = pathLength
		cp = collisionProbability(d_obstacles, obstaclesCount, d_xcomb, offsetMult, d_path, T);
		std::cout << "Collision Probability = " << cp << " at inflation = " << inflateFactor<< std::endl;
		
		cps[itrs] = cp;
		costsInfl[itrs] = costGoal;
		inflateFactors[itrs] = inflateFactor;
		itrs++;

		matlabData << "pathSoln{" << itrs << "} = ["; 
		printArray(splitPath,pathLength,DIM,matlabData); 
		matlabData << "];" << std::endl;

	}
	double t_inflate = (std::clock() - t_inflateStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Inflate took: " << t_inflate*ms << " ms for " << itrs << " iterations" << std::endl;
	std::cout << "GMT took: " << t_gmt*ms/((double)itrs) << " ms per run" << std::endl;

	// output ready to copy pasted into matlabViz/vizSoln.m
	true && printSolution(NUM, d_samples, d_edges_ptr, d_costs);
	matlabData << "cps = [" << std::endl; 
	printArray(&cps[0], 1, numInflItrs, matlabData); 
	matlabData << "];" << std::endl;
	matlabData << "inflateFactors = ["; 
	printArray(&inflateFactors[0], 1, numInflItrs, matlabData); 
	matlabData << "];" << std::endl;
	matlabData << "costsInfl = ["; 
	printArray(&costsInfl[0], 1, numInflItrs, matlabData); 
	matlabData << "];" << std::endl;

	matlabData.close();

	/***********************
	free memory and post processing
	***********************/
	cudaFree(d_costs);
	cudaFree(d_debugOutput);
	cudaFree(d_discMotions);
	cudaFree(d_distancesCome);
	cudaFree(d_nnGoEdges);
	cudaFree(d_nnComeEdges);
	cudaFree(d_nnIdxs);
	cudaFree(d_obstacles);
	cudaFree(d_obstaclesInflated);
	cudaFree(d_path);
	cudaFree(d_xcomb);

	t_overall += (std::clock() - t_overallStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Overall took: " << t_overall*ms << " ms" << std::endl;

	return EXIT_SUCCESS;
}
