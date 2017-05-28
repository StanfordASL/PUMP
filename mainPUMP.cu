/*
mainPUMP.cu
author: Brian Ichter

This code is meant for running the PUMP motion planning algorithm.

TODO: I should check for errors in kernel calls and memcpys
TODO: costs should be the full quadratic costs, not just time
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
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <fstream>
#include <sstream>

#include "helper.cuh"
#include "obstacles.cuh"
#include "sampler.cuh"
#include "bvls.cuh"
#include "hsmc.cuh"
#include "collisionCheck.cuh"
#include "2pBVP.cuh"
#include "PUMP.cuh"
#include "hardCoded.cuh"
#include "motionPlanningProblem.cuh"
#include "roadMap.cuh"

/******************************************
Run parameters
******************************************/

#ifndef DIM
#error Please define DIM.
#endif
#ifndef NUM
#error Please define NUM.
#endif

float init_loc = 0.1;
float goal_loc = 0.9; // other positions changed, see goal setting below
// float lo[DIM] = {0, 0, 0, -1, -1, -1};
// float hi[DIM] = {1, 1, 1, 1, 1, 1};
float lo[DIM] = {0, 0, 0.01, -1, -1, -0.5}; // indoor lo
float hi[DIM] = {1, 1, 0.3, 1, 1, 0.5}; // indoor hi
int numDisc = 8; // number of discretizations of kinodynamic paths

float ms = 1000;
float dt = 0.05; // the dt used for the spacing on the precomputed path offsets

float offsetMult = 0.5; // scale error
float maxd2 = 0.1*offsetMult*offsetMult; 
// don't store halfspaces who's obs point distance than any particle (0.2 calculated from xcomb)

const float lambda = 0.5;
const int numBuckets = 2; // = 1/lambda
const int numHSMCParticles = 128;
const int preMaxHSCount = 5;

const float epsCost = 0.0002;
const float epsCP = 0.0002;

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

	if(numHSMCParticles > numMCParticles) {
		std::cout << "numHSMCParticles must be less than or equal to numMCParticles!" << std::endl;
		return EXIT_FAILURE;
	}

	if (argc != 4) {
		std::cout << "Must specify an online setting filename, i.e. $ ./ground <file.txt> <cp target> <cp factor>" << std::endl;
		return EXIT_FAILURE;
	}

	int count = 0;
	cudaGetDeviceCount(&count);
	cudaError_t code;

	MotionPlanningProblem mpp;
	mpp.verbose = false;

	// problem settings
	mpp.filename = argv[1];
	mpp.dimW = DIM/2; // state space for double integrator is 6D, workspace 3D
	mpp.dimX = DIM; // use DIM instead for compiler allocation, this is just for printing really, and maybe for later
	mpp.cpTarget = atof(argv[2]);
	mpp.cpFactor = atof(argv[3]);

	mpp.hi.resize(DIM);
	mpp.lo.resize(DIM);
	for (int i = 0; i < DIM; ++i) {
		mpp.hi[i] = hi[i];
		mpp.lo[i] = lo[i];
	}

	mpp.init.resize(DIM);
	mpp.goal.resize(DIM);
	for (int i = 0; i < mpp.dimW; ++i) {
		mpp.init[i] = init_loc;
		mpp.goal[i] = goal_loc;
	}
	for (int d = mpp.dimW; d < DIM; ++d) {
	 	mpp.init[d] = 0;
	 	mpp.goal[d] = 0;
	}
	mpp.initIdx = 0;
	mpp.goalIdx = NUM-1;

	// program settings
	mpp.numDisc = numDisc;
	mpp.dt = dt;

	// algorithm settings
	mpp.numHSMCParticles = numHSMCParticles;
	mpp.numMCParticles = numMCParticles;
	mpp.numBuckets = numBuckets;
	mpp.lambda = lambda;
	mpp.epsCost = epsCost;
	mpp.epsCP = epsCP;
	mpp.roadmap.numSamples = NUM;

	// *********************** generate obstacles *************************************************
	generateObstacles(mpp.obstacles);
	mpp.numObstacles = mpp.obstacles.size()/(2*DIM);
	// inflateObstacles(obstacles.data(), obstacles.data(), 0.025, mpp.numObstacles); // inflate for quad rotor geometry

	// *********************** overwrite inputs from file *************************************************
	std::ifstream file (argv[1]);
	std::string readValue;
	if (file.good()) {
		double t_readInStart = std::clock(); 
		if (mpp.verbose) {
			std::cout << "***** Inputs from " << mpp.filename << " are: *****" << std::endl;
		}

		for (int d = 0; d < DIM; ++d) {
			getline(file, readValue, ',');	
	        std::stringstream convertorInit(readValue);
	        convertorInit >> mpp.init[d];
		}
		if (mpp.verbose) {
			std::cout << "init is: "; printArray(&mpp.init[0],1,DIM,std::cout);
		}

		for (int d = 0; d < DIM; ++d) {
			getline(file, readValue, ',');	
	        std::stringstream convertorGoal(readValue);
	        convertorGoal >> mpp.goal[d];
		}
		if (mpp.verbose) {
			std::cout << "goal is: "; printArray(&mpp.goal[0],1,DIM,std::cout);
		}

		// read in hi and lo
		for (int d = 0; d < DIM; ++d) {
			getline(file, readValue, ',');	
	        std::stringstream convertorLo(readValue);
	        convertorLo >> mpp.lo[d];
		}
		if (mpp.verbose) {
			std::cout << "lo is: "; printArray(&mpp.lo[0],1,DIM,std::cout);
		}

		for (int d = 0; d < DIM; ++d) {
			getline(file, readValue, ',');	
	        std::stringstream convertorHi(readValue);
	        convertorHi >> mpp.hi[d];
		}
		if (mpp.verbose) {
			std::cout << "hi is: "; printArray(&mpp.hi[0],1,DIM,std::cout);
		}

		// offset multiplier
		getline(file, readValue, ',');
		std::stringstream convertorOffsetMult(readValue);	
		convertorOffsetMult >> offsetMult;
		maxd2 = 0.1*offsetMult*offsetMult;
		if (mpp.verbose) {
			std::cout << "offsetmult = " << offsetMult << std::endl;
		}

		// obstacles
		getline(file, readValue, ',');
		std::stringstream convertorNumObs(readValue);	
		convertorNumObs >> mpp.numObstacles;
		if (mpp.verbose) {
			std::cout << "obstacle count = " << mpp.numObstacles << std::endl;
		}

		mpp.obstacles.resize(mpp.numObstacles*DIM*2);
		for (int obs = 0; obs < mpp.numObstacles; ++obs) {
			for (int d = 0; d < DIM*2; ++d) {
				getline(file, readValue, ',');	
				std::stringstream convertorObs(readValue);
				convertorObs >> mpp.obstacles[obs*DIM*2 + d];
			}
		}
		if(mpp.verbose) {
			printArray(&mpp.obstacles[0],mpp.numObstacles,2*DIM,std::cout);
		}

		double t_readIn = (std::clock() - t_readInStart) / (double) CLOCKS_PER_SEC;
		std::cout << "***** Reading in init, goal, obs took: " << t_readIn*ms << " ms *****" << std::endl << std::endl;
	} else {
		std::cout << "file ain't good, keeping set parameters" << std::endl << std::endl;
	}

	printMotionPlanningProblem(mpp, std::cout);

	// load mpp.obstacles on device
	float *d_obstacles;
	CUDA_ERROR_CHECK(cudaMalloc(&d_obstacles, sizeof(float)*2*mpp.numObstacles*DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_obstacles, mpp.obstacles.data(), sizeof(float)*2*mpp.numObstacles*DIM, cudaMemcpyHostToDevice));

	// have not implemented actually using this yet, but would take the maxHSCount closest half-spaces. 
	// would work well for complex obstacle sets
	mpp.hsmc.maxHSCount = mpp.numObstacles > preMaxHSCount ? preMaxHSCount : mpp.numObstacles;
	std::cout << "Allowing " << mpp.hsmc.maxHSCount << " half-spaces (not yet implemented)" << std::endl;

	// ********************** create array to return debugging information ***********************
	float *d_debugOutput;
	CUDA_ERROR_CHECK(cudaMalloc(&d_debugOutput, sizeof(float)*NUM));
	
	// *********************** generate samples ***************************************************	
	mpp.roadmap.samples.resize(DIM*NUM);
	createSamplesHalton(0, mpp.roadmap.samples.data(), mpp.init.data(), mpp.goal.data(), mpp.lo.data(), mpp.hi.data());
	thrust::device_vector<float> d_samples_thrust(DIM*NUM);
	float *d_samples = thrust::raw_pointer_cast(d_samples_thrust.data());
	CUDA_ERROR_CHECK(cudaMemcpy(d_samples, mpp.roadmap.samples.data(), sizeof(float)*DIM*NUM, cudaMemcpyHostToDevice));

	// printArray(&mpp.roadmap.samples[0], NUM, DIM, std::cout);

	// calculate nn
	double t_calc10thStart = std::clock();
	std::vector<float> topts ((NUM-1)*(NUM));
	std::vector<float> copts ((NUM-1)*(NUM));
	mpp.rnPerc = 0.05;
	int rnIdx = (int) ((NUM-1)*(NUM)*mpp.rnPerc); // 10th quartile and number of NN
	int numEdges = rnIdx;
	std::cout << "rnIdx is " << rnIdx << std::endl;
	int idx;
	float tmax = 5;

	// compute graph edge lengths and rn (note minor variations between values from GPU and CPU versions, likely due to FP computations)
	// CPU version of finding rn and costs/times
	// for (int i = 0; i < NUM; ++i) {
	// 	for (int j = 0; j < NUM; ++j) {
	// 		if (i == j)
	// 			continue;
			
	// 		idx = i*(NUM-1) + j;
	// 		if (i < j) 
 	//        		idx--;
        	
	// 		topts[idx] = toptBisection(&(mpp.roadmap.samples[i*DIM]), &(mpp.roadmap.samples[j*DIM]), tmax);
	// 		copts[idx] = cost(topts[idx], &(mpp.roadmap.samples[i*DIM]), &(mpp.roadmap.samples[j*DIM]));
	// 	}
	// }
	// std::vector<float> coptsSorted ((NUM-1)*(NUM));
	// coptsSorted = copts;
	// std::sort (coptsSorted.begin(), coptsSorted.end());
	// mpp.rn = coptsSorted[rnIdx];

	// GPU version of finding rn and costs/times
	thrust::device_vector<float> d_copts_thrust((NUM-1)*(NUM));
	float* d_copts = thrust::raw_pointer_cast(d_copts_thrust.data());
	thrust::device_vector<float> d_topts_thrust((NUM-1)*(NUM));
	float* d_topts = thrust::raw_pointer_cast(d_topts_thrust.data());
	const int blockSizeFillCoptsTopts = 192;
	const int gridSizeFillCoptsTopts = std::min((NUM*NUM + blockSizeFillCoptsTopts - 1) / blockSizeFillCoptsTopts, 2147483647);
	if (gridSizeFillCoptsTopts == 2147483647)
		std::cout << "...... ERROR: increase grid size for fillCoptsTopts" << std::endl;
	fillCoptsTopts<<<gridSizeFillCoptsTopts, blockSizeFillCoptsTopts>>>(
		d_samples, d_copts, d_topts, tmax);
	cudaDeviceSynchronize();

	CUDA_ERROR_CHECK(cudaMemcpy(topts.data(), d_topts, sizeof(float)*((NUM-1)*(NUM)), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(copts.data(), d_copts, sizeof(float)*((NUM-1)*(NUM)), cudaMemcpyDeviceToHost));
	thrust::sort(d_copts_thrust.begin(), d_copts_thrust.end());
	mpp.rn = d_copts_thrust[rnIdx];
	
	double t_calc10th = (std::clock() - t_calc10thStart) / (double) CLOCKS_PER_SEC;
	std::cout << mpp.rnPerc << "th percentile pre calc took: " << t_calc10th*ms << " ms for " << (NUM-1)*(NUM) << " solves and cutoff is " 
		<< mpp.rn << " at " << rnIdx << std::endl;	

	double t_2pbvpTestStart = std::clock();
	float x0[DIM], x1[DIM];

	double t_discMotionsStart = std::clock();
	// int nnIdxs[NUM*NUM];
	mpp.roadmap.nnIdxs.resize(NUM*NUM,-3);
	int nnComeSizes[NUM];
	for (int i = 0; i < NUM; ++i) {
		mpp.roadmap.nnGoSizes[i] = 0;
		nnComeSizes[i] = 0;
	}
	std::vector<float> discMotions (numEdges*(numDisc+1)*DIM,0); // array of motions, but its a vector who idk for some reason it won't work as an array
	int nnIdx = 0; // index position in NN discretization array
	idx = 0; // index position in copts vector above

	mpp.roadmap.coptsEdge.resize(numEdges); // edge index accessed copts
	mpp.roadmap.toptsEdge.resize(numEdges); // edge index accessed topts
	for (int i = 0; i < NUM; ++i) {
		for (int d = 0; d < DIM; ++d)
			x0[d] = mpp.roadmap.samples[d + DIM*i];
		for (int j = 0; j < NUM; ++j) {
			if (j == i)
				continue;
			if (copts[idx] < mpp.rn) {
				mpp.roadmap.coptsEdge[nnIdx] = copts[idx];
				mpp.roadmap.toptsEdge[nnIdx] = topts[idx];
				for (int d = 0; d < DIM; ++d)
					x1[d] = mpp.roadmap.samples[d + DIM*j];
				mpp.roadmap.nnIdxs[j*NUM+i] = nnIdx; // look up for discrete motions from i -> j
				findDiscretizedPath(&(discMotions[nnIdx*DIM*(numDisc+1)]), x0, x1, numDisc); // TODO: give topt
				mpp.roadmap.nnGoSizes[i]++;
				nnComeSizes[j]++;
				nnIdx++;
			}
			idx++;
		}
	}
	double t_discMotions = (std::clock() - t_discMotionsStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Discretizing motions took: " << t_discMotions*ms << " ms for " << nnIdx << " solves" << std::endl;	

	// calculate heuristic h
	mpp.roadmap.h.resize(NUM);
	for (int i = 0; i < NUM-1; ++i) {
		float topt = toptBisection(&mpp.roadmap.samples[i*DIM], mpp.goal.data(), tmax);
		mpp.roadmap.h[i] = cost(topt, &mpp.roadmap.samples[i*DIM], mpp.goal.data());
	}

	CUDA_ERROR_CHECK(cudaMalloc(&mpp.roadmap.d_toptsEdge, sizeof(float)*numEdges));
	CUDA_ERROR_CHECK(cudaMemcpy(mpp.roadmap.d_toptsEdge, mpp.roadmap.toptsEdge.data(), sizeof(float)*numEdges, cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMalloc(&mpp.roadmap.d_coptsEdge, sizeof(float)*numEdges));
	CUDA_ERROR_CHECK(cudaMemcpy(mpp.roadmap.d_coptsEdge, mpp.roadmap.coptsEdge.data(), sizeof(float)*numEdges, cudaMemcpyHostToDevice));

	mpp.roadmap.maxNNSize = 0;
	for (int i = 0; i < NUM; ++i) {
		if (mpp.roadmap.maxNNSize < mpp.roadmap.nnGoSizes[i])
			mpp.roadmap.maxNNSize = mpp.roadmap.nnGoSizes[i];
		if (mpp.roadmap.maxNNSize < nnComeSizes[i])
			mpp.roadmap.maxNNSize = nnComeSizes[i];
	}
	std::cout << "max number of nn is " << mpp.roadmap.maxNNSize << std::endl;

	std::vector<float> distancesCome (NUM*mpp.roadmap.maxNNSize, 0);
	mpp.roadmap.nnGoEdges.resize(NUM*mpp.roadmap.maxNNSize, -1); // edge gives indices (i,j) to check nnIdx to then find the discretized path
	std::vector<int> nnComeEdges (NUM*mpp.roadmap.maxNNSize, -1); // edge gives indices (j,i) to check nnIdx to then find the discretized path
	idx = 0;
	for (int i = 0; i < NUM; ++i) {
		mpp.roadmap.nnGoSizes[i] = 0; // clear nnSizes again
		nnComeSizes[i] = 0; // clear nnSizes again
	}
	for (int i = 0; i < NUM; ++i) {
		for (int j = 0; j < NUM; ++j) {
			if (j == i)
				continue;
			if (copts[idx] < mpp.rn) {
				mpp.roadmap.nnGoEdges[i*mpp.roadmap.maxNNSize + mpp.roadmap.nnGoSizes[i]] = j;
				nnComeEdges[j*mpp.roadmap.maxNNSize + nnComeSizes[j]] = i;
				distancesCome[j*mpp.roadmap.maxNNSize + nnComeSizes[j]] = copts[idx];
				mpp.roadmap.nnGoSizes[i]++;
				nnComeSizes[j]++;
			}
			idx++;
		}
	}	

	// put nearest neighbors onto device
	float *d_discMotions;
	CUDA_ERROR_CHECK(cudaMalloc(&d_discMotions, sizeof(float)*numEdges*(numDisc+1)*DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_discMotions, &discMotions[0], sizeof(float)*numEdges*(numDisc+1)*DIM, cudaMemcpyHostToDevice));

	CUDA_ERROR_CHECK(cudaMalloc(&mpp.roadmap.d_nnIdxs, sizeof(int)*NUM*NUM));
	CUDA_ERROR_CHECK(cudaMemcpy(mpp.roadmap.d_nnIdxs, mpp.roadmap.nnIdxs.data(), sizeof(int)*NUM*NUM, cudaMemcpyHostToDevice));

	float *d_distancesCome;
	CUDA_ERROR_CHECK(cudaMalloc(&d_distancesCome, sizeof(float)*NUM*mpp.roadmap.maxNNSize));
	CUDA_ERROR_CHECK(cudaMemcpy(d_distancesCome, &(distancesCome[0]), sizeof(float)*NUM*mpp.roadmap.maxNNSize, cudaMemcpyHostToDevice));

	int *d_nnGoEdges;
	CUDA_ERROR_CHECK(cudaMalloc(&d_nnGoEdges, sizeof(int)*NUM*mpp.roadmap.maxNNSize));
	CUDA_ERROR_CHECK(cudaMemcpy(d_nnGoEdges, &(mpp.roadmap.nnGoEdges[0]), sizeof(int)*NUM*mpp.roadmap.maxNNSize, cudaMemcpyHostToDevice));

	int *d_nnComeEdges;
	CUDA_ERROR_CHECK(cudaMalloc(&d_nnComeEdges, sizeof(int)*NUM*mpp.roadmap.maxNNSize));
	CUDA_ERROR_CHECK(cudaMemcpy(d_nnComeEdges, &(nnComeEdges[0]), sizeof(int)*NUM*mpp.roadmap.maxNNSize, cudaMemcpyHostToDevice));

	// instantiate memory for online computation
	thrust::device_vector<bool> d_isFreeSamples_thrust(NUM);
	bool* d_isFreeSamples = thrust::raw_pointer_cast(d_isFreeSamples_thrust.data());
	thrust::device_vector<bool> d_isFreeEdges_thrust(numEdges);
	bool* d_isFreeEdges = thrust::raw_pointer_cast(d_isFreeEdges_thrust.data());
	bool isFreeEdges[numEdges];

	CUDA_ERROR_CHECK(cudaMalloc(&mpp.hsmc.d_offsets, sizeof(float)*Tmax*numMCParticles*DIM/2));
	CUDA_ERROR_CHECK(cudaMemcpy(mpp.hsmc.d_offsets, xcomb, sizeof(float)*Tmax*numMCParticles*DIM/2, cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMalloc(&mpp.hsmc.d_offsetsTime, sizeof(float)*Tmax*numMCParticles*DIM/2));
	CUDA_ERROR_CHECK(cudaMemcpy(mpp.hsmc.d_offsetsTime, xcombTime, sizeof(float)*Tmax*numMCParticles*DIM/2, cudaMemcpyHostToDevice));
	mpp.hsmc.offsetMult = offsetMult;

	// *********************** massively parallel online computation ******************************
	// sample free points (fast)
	double t_sampleFreeStart = std::clock();
	const int blockSizeSF = 192;
	const int gridSizeSF = std::min((NUM + blockSizeSF - 1) / blockSizeSF, 2147483647);
	if (gridSizeSF == 2147483647)
		std::cout << "...... ERROR: increase grid size for sampleFree" << std::endl;
	sampleFree<<<gridSizeSF, blockSizeSF>>>(
		d_obstacles, mpp.numObstacles, d_samples, d_isFreeSamples, d_debugOutput);
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
		d_obstacles, mpp.numObstacles, d_samples, 
		d_isFreeSamples, numDisc, d_discMotions, 
		d_isFreeEdges, numEdges, d_debugOutput);
	cudaDeviceSynchronize();
	float t_edgesValid = (std::clock() - t_edgesValidStart) / (double) CLOCKS_PER_SEC;
	std::cout << "Edge validity check took: " << t_edgesValid << " s";

	std::cout << " (" << blockSizeEdgesValid << ", " << gridSizeEdgesValid << ")" << std::endl;
	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on freeEdges: " << cudaGetErrorString(code) << std::endl; }

	CUDA_ERROR_CHECK(cudaMemcpy(isFreeEdges, d_isFreeEdges, sizeof(bool)*numEdges, cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(mpp.roadmap.isFreeSamples, d_isFreeSamples, sizeof(bool)*NUM, cudaMemcpyDeviceToHost));
	mpp.roadmap.isFreeEdges.resize(numEdges);
	mpp.roadmap.isFreeEdges.assign(isFreeEdges, isFreeEdges + sizeof(isFreeEdges)/sizeof(isFreeEdges[0]));

	// find the half-spaces and store
	int halfspaceCount = numEdges*(numDisc+1)*mpp.hsmc.maxHSCount;
	CUDA_ERROR_CHECK(cudaMalloc(&mpp.hsmc.d_as, sizeof(float)*DIM/2*halfspaceCount));
	CUDA_ERROR_CHECK(cudaMalloc(&mpp.hsmc.d_bs, sizeof(float)*halfspaceCount));
	CUDA_ERROR_CHECK(cudaMalloc(&mpp.hsmc.d_countStoredHSs, sizeof(int)*numEdges*(numDisc+1)));

	const int blockSetupBs = 1024;
	const int gridSetupBs = std::min(
		(halfspaceCount + blockSetupBs - 1) / blockSetupBs, 2147483647);
	if (gridSetupBs == 2147483647)
		std::cout << "...... ERROR: increase grid size for setupBs" << std::endl;
	setupAsBs<<<gridSetupBs,blockSetupBs>>>(
		halfspaceCount, mpp.hsmc.d_bs, mpp.hsmc.d_as);
	std::cout << "setupAsBs (" << blockSetupBs << ", " << gridSetupBs << ")" << std::endl;
	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on setupBs: " << cudaGetErrorString(code) << std::endl; }

	// ******* find half spaces with a waypoint centric kernel
	// store only relevant half-spaces
	const int blockHalfspace = 1024;
	if (blockHalfspace < 2*DIM*mpp.numObstacles) {
		std::cout << "...... WARNING: blockHalfspace needs to be larger for shared obstacles" << std::endl;
	}
	const int gridHalfspace = std::min(
		(numEdges*(numDisc+1) + blockHalfspace - 1) / blockHalfspace, 2147483647);
	if (gridHalfspace == 2147483647) {std::cout << "...... ERROR: increase grid size for cacheHalfspaces" << std::endl;}
	double t_hsStart = std::clock();
	cacheHalfspaces<<<gridHalfspace, blockHalfspace, mpp.numObstacles*2*DIM*sizeof(float)>>>(
			numEdges, d_discMotions, 
			d_isFreeEdges,
			mpp.numObstacles, mpp.hsmc.maxHSCount, d_obstacles, 
			numDisc+1, mpp.hsmc.d_as, mpp.hsmc.d_bs, mpp.hsmc.d_countStoredHSs,
			NULL, maxd2);
	cudaDeviceSynchronize();
	double t_hs = (std::clock() - t_hsStart) / (double) CLOCKS_PER_SEC;
	std::cout << "HS calculation took: " << t_hs << "s for " << halfspaceCount << " computations";
	std::cout << " (" << blockHalfspace << ", " << gridHalfspace << ")" << std::endl;

	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on cacheHalfspaces: " << cudaGetErrorString(code) << std::endl; }

	// output some half-spaces for debugging and plotting
	/*
	int skipEdges = 0;
	int skipIdx = skipEdges*(numDisc+1);
	int numOuputEdges = 1;
	int numOutputPoints = numOuputEdges*(numDisc+1);
	int numOutputHS = numOuputEdges*(numDisc+1)*mpp.hsmc.maxHSCount;

	std::vector<float> as (DIM/2*numOutputHS);
	std::vector<float> bs (numOutputHS);

	CUDA_ERROR_CHECK(cudaMemcpy(&as[0], mpp.hsmc.d_as+skipIdx*mpp.hsmc.maxHSCount*DIM/2, sizeof(float)*DIM/2*numOutputHS, cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(&bs[0], mpp.hsmc.d_bs+skipIdx*mpp.hsmc.maxHSCount, sizeof(float)*numOutputHS, cudaMemcpyDeviceToHost));
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
	*/

	// ******* find half spaces with an edge centric kernel
	/*
	// store only relevant half-spaces, edge centric
	const int blockHalfspaceEdge = 1024;
	const int gridHalfspaceEdge = std::min(
		(numEdges + blockHalfspaceEdge - 1) / blockHalfspaceEdge, 2147483647);
	if (gridHalfspaceEdge == 2147483647) {std::cout << "...... ERROR: increase grid size for cacheHalfspacesEdge" << std::endl;}
	double t_hsEdgeStart = std::clock();
	cacheHalfspacesEdge<<<gridHalfspaceEdge, blockHalfspaceEdge>>>(
			numEdges, d_discMotions, 
			d_isFreeEdges, d_isFreeSamples,
			mpp.numObstacles, mpp.hsmc.maxHSCount, d_obstacles, 
			numDisc+1, mpp.hsmc.d_as, mpp.hsmc.d_bs, mpp.hsmc.d_countStoredHSs,
			NULL, maxd2);
	cudaDeviceSynchronize();
	double t_hsEdge = (std::clock() - t_hsEdgeStart) / (double) CLOCKS_PER_SEC;
	std::cout << "HS calculation edge based took: " << t_hsEdge << "s for " << halfspaceCount << " computations";
	std::cout << " (" << blockHalfspaceEdge << ", " << gridHalfspaceEdge << ")" << std::endl;

	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on cacheHalfspaces: " << cudaGetErrorString(code) << std::endl; }

	CUDA_ERROR_CHECK(cudaMemcpy(&as[0], mpp.hsmc.d_as+skipIdx*mpp.numObstacles*DIM/2, sizeof(float)*DIM/2*numOutputHS, cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(&bs[0], mpp.hsmc.d_bs+skipIdx*mpp.numObstacles, sizeof(float)*numOutputHS, cudaMemcpyDeviceToHost));
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
	*/

	// enable to print a specific edge between two nodes
	// viz with z_checkBadEdges.m
	/*
	if (NUM == 4000) {
		int nodesList[] = {3999, 2309, 1857, 270, 810, 0};
		int numNodes = 6;

		// print path
		std::cout << "pathNodes = [";
		printArray(&mpp.roadmap.samples[goalIdx*DIM],1,DIM,std::cout);
		for (int i = 0; i < numNodes-1; ++i)
		{
			int nodeFrom = nodesList[i+1];
			int nodeTo = nodesList[i];
			int myEdge = mpp.roadmap.nnIdxs[nodeTo*NUM+nodeFrom];
			printArray(&mpp.roadmap.samples[nodeFrom*DIM],1,DIM,std::cout);
		}
		std::cout << "]; " << std::endl;

		std::cout << "path = [";
		for (int i = 0; i < numNodes-1; ++i)
		{
			int nodeFrom = nodesList[i+1];
			int nodeTo = nodesList[i];
			int myEdge = mpp.roadmap.nnIdxs[nodeTo*NUM+nodeFrom];
			printArray(&discMotions[myEdge*(numDisc+1)*DIM],numDisc+1,DIM,std::cout);
		}
		std::cout << "]; " << std::endl;
		// print as
		std::cout << "as = [";
		for (int i = 0; i < numNodes-1; ++i)
		{
			int nodeFrom = nodesList[i+1];
			int nodeTo = nodesList[i];
			int myEdge = mpp.roadmap.nnIdxs[nodeTo*NUM+nodeFrom];

			std::vector<float> asCheck (mpp.numObstacles*(numDisc+1)*DIM/2);
			CUDA_ERROR_CHECK(cudaMemcpy(&asCheck[0], mpp.hsmc.d_as + myEdge*mpp.numObstacles*(numDisc+1)*DIM/2, 
				sizeof(float)*mpp.numObstacles*(numDisc+1)*DIM/2, cudaMemcpyDeviceToHost));

			printArray(&asCheck[0], mpp.numObstacles*(numDisc+1), DIM/2, std::cout);
		}
		std::cout << "]; " << std::endl;
	}
	*/

	// ********************** exploration ******************************
	// setup arrays
	mpp.pumpSearch.maxPathCount = 2*NUM * std::sqrt(NUM);
	mpp.pumpSearch.P.resize(NUM*NUM); // mpp.pumpSearch.P[i*NUM + j] index of path that exists at node i
	for (int i = 0; i < NUM; ++i) {
		for (int j =0; j < NUM; ++j) 
			mpp.pumpSearch.P[i*NUM+j] = -1; // no path exists
		mpp.pumpSearch.Pcounts[i] = 0;
	}

	std::vector<int> pathPrev(mpp.pumpSearch.maxPathCount);
	std::vector<int> pathNode(mpp.pumpSearch.maxPathCount);
	std::vector<float> pathCost(mpp.pumpSearch.maxPathCount);
	std::vector<float> pathTime(mpp.pumpSearch.maxPathCount);
	std::vector<float> pathCP(mpp.pumpSearch.maxPathCount);
	mpp.pumpSearch.pathPrev = pathPrev.data();
	mpp.pumpSearch.pathNode = pathNode.data();
	mpp.pumpSearch.pathCost = pathCost.data();
	mpp.pumpSearch.pathTime = pathTime.data();
	mpp.pumpSearch.pathCP = pathCP.data();

	// TODO: declare only once
	// int numBuckets = 2;
	// int numHSMCParticles = 512;

	int sizeG[numBuckets];
	mpp.pumpSearch.G.resize(numBuckets*mpp.pumpSearch.maxPathCount); // mpp.pumpSearch.G[Gidx*mpp.pumpSearch.maxPathCount + i] path i is in group Gidx
	// instantiate arrays (probably should do this before and pass them in)
	for (int b = 0; b < numBuckets; ++b)
		sizeG[b] = 0;
	for (int i = 0; i < numBuckets*mpp.pumpSearch.maxPathCount; ++i)
		mpp.pumpSearch.G[i] = -1;
	mpp.pumpSearch.sizeG = sizeG;

	CUDA_ERROR_CHECK(cudaMalloc(&mpp.pumpSearch.d_pathCP, sizeof(float)*mpp.pumpSearch.maxPathCount));
	CUDA_ERROR_CHECK(cudaMalloc(&mpp.pumpSearch.d_pathTime, sizeof(float)*mpp.pumpSearch.maxPathCount));

	std::cout << "number of particles tracked " << numHSMCParticles*mpp.pumpSearch.maxPathCount << std::endl;
	// thrust::device_vector<bool> d_pathValidParticles_thrust(numHSMCParticles*mpp.pumpSearch.maxPathCount);
	// bool *mpp.hsmc.d_pathValidParticles = thrust::raw_pointer_cast(d_pathValidParticles_thrust.data());
	CUDA_ERROR_CHECK(cudaMalloc(&mpp.hsmc.d_pathValidParticles, sizeof(bool)*numHSMCParticles*mpp.pumpSearch.maxPathCount));

	const int blockSetupVP = 128;
	const int gridSetupVP = std::min(
		(mpp.pumpSearch.maxPathCount + blockSetupVP - 1) / blockSetupVP, 2147483647);
	setupValidParticles<<<gridSetupVP,blockSetupVP>>>(
		mpp.hsmc.d_pathValidParticles, numHSMCParticles, mpp.pumpSearch.maxPathCount,
		mpp.pumpSearch.d_pathCP, mpp.pumpSearch.d_pathTime);

	code = cudaPeekAtLastError();
	if (cudaSuccess != code) { std::cout << "ERROR on setupValidParticles: " << cudaGetErrorString(code) << std::endl; }

	std::vector<int> wavefrontPathPrev_vec(mpp.pumpSearch.maxPathCount);
	std::vector<int> wavefrontNodeNext_vec(mpp.pumpSearch.maxPathCount);
	std::vector<int> wavefrontEdge_vec(mpp.pumpSearch.maxPathCount);
	mpp.pumpSearch.wavefrontPathPrev = wavefrontPathPrev_vec.data();
	mpp.pumpSearch.wavefrontNodeNext = wavefrontNodeNext_vec.data();
	mpp.pumpSearch.wavefrontEdge = wavefrontEdge_vec.data();
	CUDA_ERROR_CHECK(cudaMalloc(&mpp.pumpSearch.d_wavefrontPathPrev, sizeof(int)*mpp.pumpSearch.maxPathCount));
	CUDA_ERROR_CHECK(cudaMalloc(&mpp.pumpSearch.d_wavefrontNodeNext, sizeof(int)*mpp.pumpSearch.maxPathCount));
	CUDA_ERROR_CHECK(cudaMalloc(&mpp.pumpSearch.d_wavefrontEdge, sizeof(int)*mpp.pumpSearch.maxPathCount));
	
	// call actual function
	// TODO/WARNING: significantly higher speeds cna be obtained with preallocated arrays
	// instead of vectors, but they come with memory instability
	// perhaps allocate the array statically
	PUMP(mpp);

	std::cout << "Free time: \t\t" << (t_sampleFree+t_edgesValid)*ms << " ms" << std::endl;
	std::cout << "HS time: \t\t" << t_hs*ms << " ms" << std::endl;

	// *********************** free memory ******************************
	cudaFree(d_discMotions);
	cudaFree(mpp.hsmc.d_offsets);
	cudaFree(mpp.hsmc.d_offsetsTime);
	cudaFree(mpp.roadmap.d_nnIdxs);
	cudaFree(d_distancesCome);
	cudaFree(d_nnGoEdges);
	cudaFree(d_nnComeEdges);
	cudaFree(d_obstacles);
	cudaFree(d_debugOutput);
	cudaFree(mpp.hsmc.d_as);
	cudaFree(mpp.hsmc.d_bs);
	cudaFree(mpp.hsmc.d_countStoredHSs);
	cudaFree(mpp.pumpSearch.d_pathCP);
	cudaFree(mpp.pumpSearch.d_pathTime);
	cudaFree(mpp.pumpSearch.d_wavefrontPathPrev);
	cudaFree(mpp.pumpSearch.d_wavefrontNodeNext);
	cudaFree(mpp.pumpSearch.d_wavefrontEdge);
	cudaFree(mpp.hsmc.d_pathValidParticles);

	return EXIT_SUCCESS;
}