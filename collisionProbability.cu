
#include "collisionProbability.cuh"

float collisionProbability(float *d_obstacles, int obstaclesCount, float *d_xcomb, float offsetMult, float *d_path, int T)
{
	float *d_debugOutput;
	CUDA_ERROR_CHECK(cudaMalloc(&d_debugOutput, sizeof(float)*numMCSamples*DIM));

	// double t_cpStart = std::clock();
	thrust::device_vector<int> d_isCollision(numMCSamples);
	int* d_isCollision_ptr = thrust::raw_pointer_cast(d_isCollision.data());

	const int blockSize = 64;
	const int gridSize = std::min((numMCSamples + blockSize - 1) / blockSize, 2147483647);
	MCCP<<<gridSize,blockSize>>>(d_isCollision_ptr, d_path, d_xcomb, offsetMult, d_obstacles, obstaclesCount, T, d_debugOutput);
	cudaDeviceSynchronize();
	cudaError_t code = cudaPeekAtLastError();
		if (cudaSuccess != code) { std::cout << "ERROR on MCCP: " << cudaGetErrorString(code) << std::endl; }

	// int isCollision[numMCSamples];
	// cudaMemcpy(isCollision, d_isCollision_ptr, sizeof(int)*numMCSamples, cudaMemcpyDeviceToHost);
	// printArray(isCollision, 1, numMCSamples, std::cout);

	// float debugOutput[numMCSamples*DIM];
	// cudaMemcpy(debugOutput, d_debugOutput, sizeof(float)*DIM*numMCSamples, cudaMemcpyDeviceToHost);
	// printArray(debugOutput, numMCSamples, DIM, std::cout);

	int numCollisions = thrust::reduce(d_isCollision.begin(), d_isCollision.end());
	float cp = ((float) numCollisions)/((float) numMCSamples);

	// double t_cp = (std::clock() - t_cpStart) / (double) CLOCKS_PER_SEC;
	// double ms = 1000;
	// std::cout << "CP estimate took: " << t_cp*ms << " ms and number of collisions: " << numCollisions << std::endl;

	cudaFree(d_debugOutput);
	return cp;
}

__global__ 
void MCCP(int *isCollision, float *path, float *xcomb, float offsetMult, 
	float *obstacles, int obstaclesCount, int T, float *debugOutput) 
{
	int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIdx >= numMCSamples) 
		return;
	isCollision[particleIdx] = 0;

	float v[DIM/2], w[DIM/2];
	float bbMin[DIM/2], bbMax[DIM/2];
	for (int t = 0; t < T-2; ++t) {
		for (int d = 0; d < DIM/2; ++d) {
			v[d] = path[t*DIM + d] + xcomb[particleIdx*DIM/2*Tmax + t*DIM/2 + d]*offsetMult;
			w[d] = path[(t+1)*DIM + d] + xcomb[particleIdx*DIM/2*Tmax + (t+1)*DIM/2 + d]*offsetMult;
			// debugOutput[particleIdx*DIM+d] = v[d]; // gives collision location

			if (v[d] > w[d]) {
				bbMin[d] = w[d];
				bbMax[d] = v[d];
			} else {
				bbMin[d] = v[d];
				bbMax[d] = w[d];
			}
		}

		/* check if points are within an obstacle (not necessary for CC and still have to run below routine)
		I think only useful if an entire warp were to have collisions, and even then, barely
		otherwise its just extra checks that aren't necessary
		with a small enough discretization I could maybe just do this though, which would have large gains
		timing shows more of less no change 
		*/
		// for (int obs_idx = 0; obs_idx < obstaclesCount; ++obs_idx) {
		// 	bool notFree = true;
		// 	for (int d = 0; d < DIM; ++d) {
		// 		notFree = notFree && 
		// 		w[d] > obstacles[obs_idx*2*DIM + d] && 
		// 		w[d] < obstacles[obs_idx*2*DIM + DIM + d];
		// 		if (!notFree)
		// 			break;
		// 	}
		// 	if (notFree) {
		// 		isCollision[particleIdx] = 1;
		// 		return;
		// 	}
		// }	

		bool motionValid = isMotionValid(v, w, bbMin, bbMax, obstaclesCount, obstacles, debugOutput);
		if (!motionValid) {
			isCollision[particleIdx] = 1;
			return;
		}
	}
}