/*
testPUMP>cu
author: Brian Ichter

Runs the PUMP algorithm.
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
#include <thrust/reduce.h>
#include <fstream>

#include "helper.cuh"
#include "obstacles.cuh"
#include "sampler.cuh"
#include "bvls.cuh"
#include "hsmc.cuh"
#include "collisionCheck.cuh"
#include "2pBVP.cuh"
#include "CCGMT.cuh"
#include "hardCoded.cuh"

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

float initial_loc = 0.1;
float goal_loc = 0.9; // other positions changed, see goal setting below
bool activeKernel = true; // false = map cores to all samples, true = map cores to only active samples (requires a sort and find op)
// float lo[DIM] = {0, 0, 0.01, -1, -1, -0.5}; // indoor lo
// float hi[DIM] = {1, 1, 0.3, 1, 1, 0.5}; // indoor hi
float lo[DIM] = {0, 0, 0, -1, -1, -1};
float hi[DIM] = {1, 1, 1, 1, 1, 1};
int numDisc = 8; // number of discretizations of kinodynamic paths

float ms = 1000;
float dt = 0.05;

float offsetMult = 1.0; // scale error
float maxd2 = 0.1*offsetMult*offsetMult; 
// don't store halfspaces who's obs point distance than any particle (0.2 calculated from xcomb)

const float lambda = 0.5;
const int numBuckets = 2; // = 1/lambda
const int numHSMCParticles = 128;
const int preMaxHSCount = 5;

const float epsCost = 0.0002;
const float epsCP = 0.0002;
float cpTarget = 0.01; // NOTE: ALSO UPDATE EPS_CP for smoothing in CCGMT.cu
float cpFactor = 5;

/******************************************
Global memory
******************************************/

/******************************************
Solution Data Struct and functions
******************************************/

int main(int argc, char **argv)
{
	if(DIM != 6) {
		std::cout << "DIM MUST BE 6, this solves the 3D double integrator only!" << std::endl;
		return EXIT_FAILURE;
	}

	if(numHSMCParticles > numMCSamples) {
		std::cout << "numHSMCParticles must be less than or equal to numMCSamples!" << std::endl;
		return EXIT_FAILURE;
	}

	if (argc == 3) {
		cpTarget = atof(argv[1]);
		cpFactor = atof(argv[2]);
	}

	int count = 0;
	cudaGetDeviceCount(&count);
	cudaError_t code;

	std::cout << "New run: DIM = " << DIM << ", NUM = " << NUM << ", RUNS = " << RUNS << std::endl;

    /***********************
	setup for hsmc test
	***********************/

	// *********************** generate obstacles *************************************************
	int obstaclesCount = getObstaclesCount();
	float obstacles[obstaclesCount*2*DIM];
	generateObstacles(obstacles, obstaclesCount*2*DIM);
	// inflateObstacles(obstacles, obstacles, 0.025, obstaclesCount); // inflate for quad rotor geometry
	std::cout << "the " << obstaclesCount << " obstacles are = " << std::endl;
	printArray(obstacles, obstaclesCount, 2*DIM, std::cout);

	// load obstacles on device
	float *d_obstacles;
	CUDA_ERROR_CHECK(cudaMalloc(&d_obstacles, sizeof(float)*2*obstaclesCount*DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_obstacles, obstacles, sizeof(float)*2*obstaclesCount*DIM, cudaMemcpyHostToDevice));

	int maxHSCount = obstaclesCount > preMaxHSCount ? preMaxHSCount : obstaclesCount;
	std::cout << "Allowing " << maxHSCount << " half-spaces (not yet implemented)" << std::endl;

	// ********************** create array to return debugging information ***********************
	float *d_debugOutput;
	CUDA_ERROR_CHECK(cudaMalloc(&d_debugOutput, sizeof(float)*NUM));
	
	// *********************** generate samples ***************************************************
	// generate initial and goal states
	std::vector<float> initial(DIM, initial_loc);
	std::vector<float> goal(DIM, goal_loc); 
	// (0.05, 0.05, 0.2) -> (0.5, 0.95, 0.2) for indoor
	// initial[2] = 0.2;
	// goal[0] = 0.5;
	// goal[2] = 0.2;
	// initial[2] = 0.2;
	// initial[0] = 0.7;
	// goal[0] = 0.7;
	// 3obs new (0.75 0.1 0.1) -> (0.75 0.9 0.9),
	// initial[0] = 0.75;
	// goal[0] = 0.75;
	for (int d = DIM/2; d < DIM; ++d) {
	 	initial[d] = 0;
	 	goal[d] = 0;
	}
	int goalIdx = NUM-1;
	int initIdx = 0;

	std::vector<float> samplesAll (DIM*NUM);
	createSamplesHalton(0, samplesAll.data(), &(initial[0]), &(goal[0]), lo, hi);
	thrust::device_vector<float> d_samples_thrust(DIM*NUM);
	float *d_samples = thrust::raw_pointer_cast(d_samples_thrust.data());
	CUDA_ERROR_CHECK(cudaMemcpy(d_samples, samplesAll.data(), sizeof(float)*DIM*NUM, cudaMemcpyHostToDevice));

	// printArray(&samplesAll[0], NUM, DIM, std::cout);

	// calculate nn
	double t_calc10thStart = std::clock();
	std::vector<float> topts ((NUM-1)*(NUM));
	std::vector<float> copts ((NUM-1)*(NUM));
	int percentile = 20;
	int rnIdx = (NUM-1)*(NUM)/percentile; // 10th quartile and number of NN
	int numEdges = rnIdx;
	std::cout << "rnIdx is " << rnIdx << std::endl;
	int idx = 0;
	float tmax = 5;
	for (int i = 0; i < NUM; ++i) {
		for (int j = 0; j < NUM; ++j) {
			if (j == i)
				continue;
			topts[idx] = toptBisection(&(samplesAll[i*DIM]), &(samplesAll[j*DIM]), tmax);
			copts[idx] = cost(topts[idx], &(samplesAll[i*DIM]), &(samplesAll[j*DIM]));
			idx++;
		}
	}
	std::vector<float> coptsSorted ((NUM-1)*(NUM));
	coptsSorted = copts;
	std::sort (coptsSorted.begin(), coptsSorted.end());
	float rn = coptsSorted[rnIdx];
	double t_calc10th = (std::clock() - t_calc10thStart) / (double) CLOCKS_PER_SEC;
	std::cout << percentile << "th percentile pre calc took: " << t_calc10th*ms << " ms for " << idx << " solves and cutoff is " 
		<< rn << " at " << rnIdx << std::endl;	

	double t_2pbvpTestStart = std::clock();
	// int numDisc = 4;
	float x0[DIM], x1[DIM];
	// float copt;

	double t_discMotionsStart = std::clock();
	// int nnIdxs[NUM*NUM];
	std::vector<int> nnIdxs(NUM*NUM,-3);
	int nnGoSizes[NUM];
	int nnComeSizes[NUM];
	for (int i = 0; i < NUM; ++i) {
		nnGoSizes[i] = 0;
		nnComeSizes[i] = 0;
	}
	std::vector<float> discMotions (numEdges*(numDisc+1)*DIM,0); // array of motions, but its a vector who idk for some reason it won't work as an array
	int nnIdx = 0; // index position in NN discretization array
	idx = 0; // index position in copts vector above

	std::vector<float> coptsEdge (numEdges); // edge index accessed copts
	std::vector<float> toptsEdge (numEdges); // edge index accessed topts
	for (int i = 0; i < NUM; ++i) {
		for (int d = 0; d < DIM; ++d)
			x0[d] = samplesAll[d + DIM*i];
		for (int j = 0; j < NUM; ++j) {
			if (j == i)
				continue;
			if (copts[idx] < rn) {
				coptsEdge[nnIdx] = copts[idx];
				toptsEdge[nnIdx] = topts[idx];
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

	// calculate heuristic h
	float h[NUM];
	for (int i = 0; i < NUM-1; ++i) {
		float topt = toptBisection(&samplesAll[i*DIM], goal.data(), tmax);
		h[i] = cost(topt, &samplesAll[i*DIM], goal.data());
	}

	float *d_toptsEdge;
	CUDA_ERROR_CHECK(cudaMalloc(&d_toptsEdge, sizeof(float)*numEdges));
	CUDA_ERROR_CHECK(cudaMemcpy(d_toptsEdge, toptsEdge.data(), sizeof(float)*numEdges, cudaMemcpyHostToDevice));
	float *d_coptsEdge;
	CUDA_ERROR_CHECK(cudaMalloc(&d_coptsEdge, sizeof(float)*numEdges));
	CUDA_ERROR_CHECK(cudaMemcpy(d_coptsEdge, coptsEdge.data(), sizeof(float)*numEdges, cudaMemcpyHostToDevice));

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
			if (copts[idx] < rn) {
				nnGoEdges[i*maxNNSize + nnGoSizes[i]] = j;
				nnComeEdges[j*maxNNSize + nnComeSizes[j]] = i;
				distancesCome[j*maxNNSize + nnComeSizes[j]] = copts[idx];
				nnGoSizes[i]++;
				nnComeSizes[j]++;
			}
			idx++;
		}
	}	

	// put NN onto device
	float *d_discMotions;
	CUDA_ERROR_CHECK(cudaMalloc(&d_discMotions, sizeof(float)*numEdges*(numDisc+1)*DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_discMotions, &discMotions[0], sizeof(float)*numEdges*(numDisc+1)*DIM, cudaMemcpyHostToDevice));
	// std::cout << "**** disc motions = " << std::endl;
	// printArray(&discMotions[0], 30, DIM, std::cout);

	int *d_nnIdxs;
	CUDA_ERROR_CHECK(cudaMalloc(&d_nnIdxs, sizeof(int)*NUM*NUM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_nnIdxs, &(nnIdxs[0]), sizeof(int)*NUM*NUM, cudaMemcpyHostToDevice));

	float *d_distancesCome;
	CUDA_ERROR_CHECK(cudaMalloc(&d_distancesCome, sizeof(float)*NUM*maxNNSize));
	CUDA_ERROR_CHECK(cudaMemcpy(d_distancesCome, &(distancesCome[0]), sizeof(float)*NUM*maxNNSize, cudaMemcpyHostToDevice));

	int *d_nnGoEdges;
	CUDA_ERROR_CHECK(cudaMalloc(&d_nnGoEdges, sizeof(int)*NUM*maxNNSize));
	CUDA_ERROR_CHECK(cudaMemcpy(d_nnGoEdges, &(nnGoEdges[0]), sizeof(int)*NUM*maxNNSize, cudaMemcpyHostToDevice));

	int *d_nnComeEdges;
	CUDA_ERROR_CHECK(cudaMalloc(&d_nnComeEdges, sizeof(int)*NUM*maxNNSize));
	CUDA_ERROR_CHECK(cudaMemcpy(d_nnComeEdges, &(nnComeEdges[0]), sizeof(int)*NUM*maxNNSize, cudaMemcpyHostToDevice));

	// instantiate memory for online comp
	thrust::device_vector<bool> d_isFreeSamples_thrust(NUM);
	bool* d_isFreeSamples = thrust::raw_pointer_cast(d_isFreeSamples_thrust.data());
	thrust::device_vector<bool> d_isFreeEdges_thrust(numEdges);
	bool* d_isFreeEdges = thrust::raw_pointer_cast(d_isFreeEdges_thrust.data());
	bool isFreeEdges[numEdges];
	bool isFreeSamples[NUM];

	// calculate path offsets from hardcoded
	/* find the largest distance from the path in xcomb
	float maxFoundD2 = 0;
	for (int i = 0; i < numMCSamples; ++i) {
		for (int t = 0; t < Tmax; ++t) {
			float d2 = 0;
			for (int d = 0; d < DIM/2; ++d) {
				d2 += xcomb[i*Tmax*DIM/2 + t*DIM/2 + d]*xcomb[i*Tmax*DIM/2 + t*DIM/2 + d];
			}
			if (d2 > maxFoundD2)
				maxFoundD2 = d2;
		}
	}
	std::cout << "found a d2 of " << maxFoundD2 << std::endl;
	*/

	float *d_offsets;
	CUDA_ERROR_CHECK(cudaMalloc(&d_offsets, sizeof(float)*Tmax*numMCSamples*DIM/2));
	CUDA_ERROR_CHECK(cudaMemcpy(d_offsets, xcomb, sizeof(float)*Tmax*numMCSamples*DIM/2, cudaMemcpyHostToDevice));
	float *d_offsetsTime;
	CUDA_ERROR_CHECK(cudaMalloc(&d_offsetsTime, sizeof(float)*Tmax*numMCSamples*DIM/2));
	CUDA_ERROR_CHECK(cudaMemcpy(d_offsetsTime, xcombTime, sizeof(float)*Tmax*numMCSamples*DIM/2, cudaMemcpyHostToDevice));

	// *********************** massively parallel online computation ******************************
	// sample free points (fast)
	double t_sampleFreeStart = std::clock();
	const int blockSizeSF = 192;
	const int gridSizeSF = std::min((NUM + blockSizeSF - 1) / blockSizeSF, 2147483647);
	if (gridSizeSF == 2147483647)
		std::cout << "...... ERROR: increase grid size for sampleFree" << std::endl;
	sampleFree<<<gridSizeSF, blockSizeSF>>>(
		d_obstacles, obstaclesCount, d_samples, d_isFreeSamples, d_debugOutput);
	cudaDeviceSynchronize();

	float t_sampleFree = (std::clock() - t_sampleFreeStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Sample free took: " << t_sampleFree << " s" << std::endl;

	// remove edges in collision (fast)
	// track these edges and potentially remove from storage?
	double t_edgesValidStart = std::clock();
	const int blockSizeEdgesValid = 192;
	const int gridSizeEdgesValid = std::min((numEdges + blockSizeEdgesValid - 1) / blockSizeEdgesValid, 2147483647);
	if (gridSizeEdgesValid == 2147483647)
		std::cout << "...... ERROR: increase grid size for freeEdges" << std::endl;
	freeEdges<<<gridSizeEdgesValid,blockSizeEdgesValid>>>(
		d_obstacles, obstaclesCount, d_samples, 
		d_isFreeSamples, numDisc, d_discMotions, 
		d_isFreeEdges, numEdges, d_debugOutput);
	cudaDeviceSynchronize();
	float t_edgesValid = (std::clock() - t_edgesValidStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Edge validity check took: " << t_edgesValid << " s";

	std::cout << " (" << blockSizeEdgesValid << ", " << gridSizeEdgesValid << ")" << std::endl;
	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on freeEdges: " << cudaGetErrorString(code) << std::endl; }

	CUDA_ERROR_CHECK(cudaMemcpy(isFreeEdges, d_isFreeEdges, sizeof(bool)*numEdges, cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(isFreeSamples, d_isFreeSamples, sizeof(bool)*NUM, cudaMemcpyDeviceToHost));

	// find the half-spaces and store
	int halfspaceCount = numEdges*(numDisc+1)*obstaclesCount;
	float *d_as;
	float *d_bs;
	int *d_countStoredHSs;
	CUDA_ERROR_CHECK(cudaMalloc(&d_as, sizeof(float)*DIM/2*halfspaceCount));
	CUDA_ERROR_CHECK(cudaMalloc(&d_bs, sizeof(float)*halfspaceCount));
	CUDA_ERROR_CHECK(cudaMalloc(&d_countStoredHSs, sizeof(int)*numEdges*(numDisc+1)));

	const int blockSetupBs = 1024;
	const int gridSetupBs = std::min(
		(halfspaceCount + blockSetupBs - 1) / blockSetupBs, 2147483647);
	if (gridSetupBs == 2147483647)
		std::cout << "...... ERROR: increase grid size for setupBs" << std::endl;
	setupBs<<<gridSetupBs,blockSetupBs>>>(
		halfspaceCount, d_bs, d_as);
	std::cout << "setupBs (" << blockSetupBs << ", " << gridSetupBs << ")" << std::endl;
	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on setupBs: " << cudaGetErrorString(code) << std::endl; }

	// store only relevant half-spaces
	const int blockHalfspace = 1024;
	if (blockHalfspace < 2*DIM*20) {std::cout << "...... WARNING: blockHalfspace may need to be larger for shared obstacles" << std::endl;}
	const int gridHalfspace = std::min(
		(numEdges*(numDisc+1) + blockHalfspace - 1) / blockHalfspace, 2147483647);
	if (gridHalfspace == 2147483647) {std::cout << "...... ERROR: increase grid size for cacheHalfspaces" << std::endl;}
	double t_hsStart = std::clock();
	cacheHalfspaces<<<gridHalfspace, blockHalfspace>>>(
			numEdges, d_discMotions, 
			d_isFreeEdges,
			obstaclesCount, d_obstacles, 
			numDisc+1, d_as, d_bs, d_countStoredHSs,
			NULL, maxd2);
	cudaDeviceSynchronize();
	double t_hs = (std::clock() - t_hsStart) / (double) CLOCKS_PER_SEC;
	std::cout << "HS calculation took: " << t_hs << "s for " << halfspaceCount << " computations";
	std::cout << " (" << blockHalfspace << ", " << gridHalfspace << ")" << std::endl;

	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on cacheHalfspaces: " << cudaGetErrorString(code) << std::endl; }

	int skipEdges = 0;
	int skipIdx = skipEdges*(numDisc+1);
	int numOuputEdges = 1;
	int numOutputPoints = numOuputEdges*(numDisc+1);
	int numOutputHS = numOuputEdges*(numDisc+1)*obstaclesCount;

	std::vector<float> as (DIM/2*numOutputHS);
	std::vector<float> bs (numOutputHS);

	CUDA_ERROR_CHECK(cudaMemcpy(&as[0], d_as+skipIdx*obstaclesCount*DIM/2, sizeof(float)*DIM/2*numOutputHS, cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(&bs[0], d_bs+skipIdx*obstaclesCount, sizeof(float)*numOutputHS, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	std::ofstream matlabDataVan;
	matlabDataVan.open("../referenceCode/matlabHSdataVan.m");
	matlabDataVan << "%% " << std::endl << "as = ["; 
	printArray(&as[0], numOutputHS, DIM/2, matlabDataVan);
	matlabDataVan << "];" << std::endl;
	matlabDataVan << "%% " << std::endl << "bs = ["; 
	printArray(&bs[0], numOutputHS, 1, matlabDataVan);
	matlabDataVan << "];" << std::endl;
	matlabDataVan << "%% " << std::endl << "discMotions = ["; 
	printArray(&discMotions[skipIdx*DIM], numOutputPoints, DIM, matlabDataVan);
	matlabDataVan << "];" << std::endl << "%% end of file" ;
	matlabDataVan.close();

	// store only relevant half-spaces, edge centric
	const int blockHalfspaceEdge = 1024;
	const int gridHalfspaceEdge = std::min(
		(numEdges + blockHalfspaceEdge - 1) / blockHalfspaceEdge, 2147483647);
	if (gridHalfspaceEdge == 2147483647) {std::cout << "...... ERROR: increase grid size for cacheHalfspacesEdge" << std::endl;}
	double t_hsEdgeStart = std::clock();
	cacheHalfspacesEdge<<<gridHalfspaceEdge, blockHalfspaceEdge>>>(
			numEdges, d_discMotions, 
			d_isFreeEdges, d_isFreeSamples,
			obstaclesCount, d_obstacles, 
			numDisc+1, d_as, d_bs, d_countStoredHSs,
			NULL, maxd2);
	cudaDeviceSynchronize();
	double t_hsEdge = (std::clock() - t_hsEdgeStart) / (double) CLOCKS_PER_SEC;
	std::cout << "HS calculation edge based took: " << t_hsEdge << "s for " << halfspaceCount << " computations";
	std::cout << " (" << blockHalfspaceEdge << ", " << gridHalfspaceEdge << ")" << std::endl;

	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on cacheHalfspaces: " << cudaGetErrorString(code) << std::endl; }

	CUDA_ERROR_CHECK(cudaMemcpy(&as[0], d_as+skipIdx*obstaclesCount*DIM/2, sizeof(float)*DIM/2*numOutputHS, cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(&bs[0], d_bs+skipIdx*obstaclesCount, sizeof(float)*numOutputHS, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	std::ofstream matlabDataEdge;
	matlabDataEdge.open("../referenceCode/matlabHSdataEdge.m");
	matlabDataEdge << "%% " << std::endl << "as = ["; 
	printArray(&as[0], numOutputHS, DIM/2, matlabDataEdge);
	matlabDataEdge << "];" << std::endl;
	matlabDataEdge << "%% " << std::endl << "bs = ["; 
	printArray(&bs[0], numOutputHS, 1, matlabDataEdge);
	matlabDataEdge << "];" << std::endl;
	matlabDataEdge << "%% " << std::endl << "discMotions = ["; 
	printArray(&discMotions[skipIdx*DIM], numOutputPoints, DIM, matlabDataEdge);
	matlabDataEdge << "];" << std::endl << "%% end of file" ;
	matlabDataEdge.close();


	// enable to print a specific edge between two nodes
	// viz with z_checkBadEdges.m
	/*
	if (NUM == 4000) {
		int nodesList[] = {3999, 2309, 1857, 270, 810, 0};
		int numNodes = 6;

		// print path
		std::cout << "pathNodes = [";
		printArray(&samplesAll[goalIdx*DIM],1,DIM,std::cout);
		for (int i = 0; i < numNodes-1; ++i)
		{
			int nodeFrom = nodesList[i+1];
			int nodeTo = nodesList[i];
			int myEdge = nnIdxs[nodeTo*NUM+nodeFrom];
			printArray(&samplesAll[nodeFrom*DIM],1,DIM,std::cout);
		}
		std::cout << "]; " << std::endl;

		std::cout << "path = [";
		for (int i = 0; i < numNodes-1; ++i)
		{
			int nodeFrom = nodesList[i+1];
			int nodeTo = nodesList[i];
			int myEdge = nnIdxs[nodeTo*NUM+nodeFrom];
			printArray(&discMotions[myEdge*(numDisc+1)*DIM],numDisc+1,DIM,std::cout);
		}
		std::cout << "]; " << std::endl;
		// print as
		std::cout << "as = [";
		for (int i = 0; i < numNodes-1; ++i)
		{
			int nodeFrom = nodesList[i+1];
			int nodeTo = nodesList[i];
			int myEdge = nnIdxs[nodeTo*NUM+nodeFrom];

			std::vector<float> asCheck (obstaclesCount*(numDisc+1)*DIM/2);
			CUDA_ERROR_CHECK(cudaMemcpy(&asCheck[0], d_as + myEdge*obstaclesCount*(numDisc+1)*DIM/2, 
				sizeof(float)*obstaclesCount*(numDisc+1)*DIM/2, cudaMemcpyDeviceToHost));

			printArray(&asCheck[0], obstaclesCount*(numDisc+1), DIM/2, std::cout);
		}
		std::cout << "]; " << std::endl;
	}
	*/

	// ********************** exploration ******************************
	// setup arrays
	int maxPathCount = 2*NUM * std::sqrt(NUM);
	std::vector<int> P (NUM*NUM); // P[i*NUM + j] index of path that exists at node i
	int Pcounts[NUM]; // Pcounts[i] number of paths currently through node i
	for (int i = 0; i < NUM; ++i) {
		for (int j =0; j < NUM; ++j) 
			P[i*NUM+j] = -1; // no path exists
		Pcounts[i] = 0;
	}

	std::vector<int> pathPrev (maxPathCount);
	std::vector<int> pathNode (maxPathCount);
	std::vector<float> pathCost (maxPathCount);
	std::vector<float> pathTime (maxPathCount);
	std::vector<float> pathCP (maxPathCount);

	// TODO: declare only once
	// int numBuckets = 2;
	// int numHSMCParticles = 512;

	int sizeG[numBuckets];
	std::vector<int> G (numBuckets*maxPathCount); // G[Gidx*maxPathCount + i] path i is in group Gidx
	// instantiate arrays (probably should do this before and pass them in)
	for (int b = 0; b < numBuckets; ++b)
		sizeG[b] = 0;
	for (int i = 0; i < numBuckets*maxPathCount; ++i)
		G[i] = -1;

	float *d_pathCP;
	float *d_pathTime;
	CUDA_ERROR_CHECK(cudaMalloc(&d_pathCP, sizeof(float)*maxPathCount));
	CUDA_ERROR_CHECK(cudaMalloc(&d_pathTime, sizeof(float)*maxPathCount));

	std::cout << "number of particles tracked " << numHSMCParticles*maxPathCount << std::endl;
	// thrust::device_vector<bool> d_pathValidParticles_thrust(numHSMCParticles*maxPathCount);
	// bool *d_pathValidParticles = thrust::raw_pointer_cast(d_pathValidParticles_thrust.data());
	bool *d_pathValidParticles;
	CUDA_ERROR_CHECK(cudaMalloc(&d_pathValidParticles, sizeof(bool)*numHSMCParticles*maxPathCount));

	const int blockSetupVP = 128;
	const int gridSetupVP = std::min(
		(maxPathCount + blockSetupVP - 1) / blockSetupVP, 2147483647);
	setupValidParticles<<<gridSetupVP,blockSetupVP>>>(
		d_pathValidParticles, numHSMCParticles, maxPathCount,
		d_pathCP, d_pathTime);

	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on setupValidParticles: " << cudaGetErrorString(code) << std::endl; }

	std::vector<int> wavefrontPathPrev_vec(maxPathCount);
	std::vector<int> wavefrontNodeNext_vec(maxPathCount);
	std::vector<int> wavefrontEdge_vec(maxPathCount);
	int *wavefrontPathPrev = wavefrontPathPrev_vec.data();
	int *wavefrontNodeNext = wavefrontNodeNext_vec.data();
	int *wavefrontEdge = wavefrontEdge_vec.data();
	int *d_wavefrontPathPrev;
	int *d_wavefrontNodeNext;
	int *d_wavefrontEdge;
	CUDA_ERROR_CHECK(cudaMalloc(&d_wavefrontPathPrev, sizeof(int)*maxPathCount));
	CUDA_ERROR_CHECK(cudaMalloc(&d_wavefrontNodeNext, sizeof(int)*maxPathCount));
	CUDA_ERROR_CHECK(cudaMalloc(&d_wavefrontEdge, sizeof(int)*maxPathCount));
	
	// call actual function
	// TODO/WARNING: significantly higher speeds cna be obtained with preallocated arrays
	// instead of vectors, but they come with memory instability
	// perhaps allocate the array statically
	CCGMTcpu(rn, cpTarget, initIdx, goalIdx, dt, numMCSamples,
		h, nnGoSizes, nnGoEdges.data(), nnIdxs.data(), maxNNSize, isFreeSamples, isFreeEdges,
		numDisc, obstaclesCount, d_toptsEdge, &toptsEdge[0], d_coptsEdge, &coptsEdge[0], 
		d_as, d_bs, d_countStoredHSs, d_offsets, d_offsetsTime, offsetMult,
		P, Pcounts, pathPrev.data(), pathNode.data(), pathCost.data(), pathTime.data(), pathCP.data(), maxPathCount,
		G, sizeG, d_pathValidParticles, d_pathCP, d_pathTime,
		wavefrontPathPrev, wavefrontNodeNext, wavefrontEdge,
		d_wavefrontPathPrev, d_wavefrontNodeNext, d_wavefrontEdge,
		epsCost, epsCP, lambda, numBuckets, cpFactor, numHSMCParticles,
		samplesAll.data(), obstacles, Tmax);

	std::cout << "Free time: \t\t" << (t_sampleFree+t_edgesValid)*ms << " ms" << std::endl;
	std::cout << "HS time: \t\t" << t_hs*ms << " ms" << std::endl;

	// *********************** free memory ******************************
	cudaFree(d_discMotions);
	cudaFree(d_offsets);
	cudaFree(d_offsetsTime);
	cudaFree(d_nnIdxs);
	cudaFree(d_distancesCome);
	cudaFree(d_nnGoEdges);
	cudaFree(d_nnComeEdges);
	cudaFree(d_obstacles);
	cudaFree(d_debugOutput);
	cudaFree(d_as);
	cudaFree(d_bs);
	cudaFree(d_countStoredHSs);
	cudaFree(d_pathCP);
	cudaFree(d_pathTime);
	cudaFree(d_wavefrontPathPrev);
	cudaFree(d_wavefrontNodeNext);
	cudaFree(d_wavefrontEdge);
	cudaFree(d_pathValidParticles);

	return EXIT_SUCCESS;
}
