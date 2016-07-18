#include "hsmc.cuh"

/*
Given discMotions and obstacles
Calculate the closest intersection using bvls for discMotions/obstacles pairs
(later to expanded to waypoints)
Return as and bs
*/

__global__ 
void cacheHalfspaces(int numEdges, float *discMotions, 
	bool *isFreeEdges,
	int obstaclesCount, float *obstacles, 
	int waypointsCount, float *as, float *bs, int *countStoredHSs, 
	float *sigma, float maxd2)
{
	// put obstacles into shared memory
	const int obstacleSize = 3*2*DIM;
	__shared__ float s_obstacles[obstacleSize];
	int obsStorageIdx = threadIdx.x;
	if (obsStorageIdx < obstacleSize)
		s_obstacles[obsStorageIdx] = obstacles[obsStorageIdx];
	__syncthreads();

	int motionWaypointIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (motionWaypointIdx >= numEdges*waypointsCount)
		return;

	int edgeIdx = motionWaypointIdx/(waypointsCount);
	if (!isFreeEdges[edgeIdx])
		return;

	// TODO: put obstacles into shared?
	// how is it shared between blocks

	int dim = DIM/2; 

	float pos[DIM/2]; // sample point
	float vel[DIM/2]; // velocity at sample point
	for (int d = 0; d < dim; ++d) {
		pos[d] = discMotions[motionWaypointIdx*DIM + d];
		vel[d] = discMotions[motionWaypointIdx*DIM + DIM/2 + d];
	}

	// float l[DIM/2]; // lower obstacle corner
	// float u[DIM/2]; // upper obstacle corner 
	float x[DIM/2]; // returned a result

	int hsIdx = motionWaypointIdx*obstaclesCount;
	int countStoredHS = 0;
	for (int obsIdx = 0; obsIdx < obstaclesCount; ++obsIdx) {
		// for (int d = 0; d < dim; ++d) {
		// 	l[d] = s_obstacles[obsIdx*2*DIM + d];
		// 	u[d] = s_obstacles[obsIdx*2*DIM + DIM + d];
		// }

		bvlsShortcut(x, NULL, pos, &s_obstacles[obsIdx*2*DIM], &s_obstacles[obsIdx*2*DIM + DIM], NULL);

		/*
		check distance
		*/
		for (int d = 0; d < dim; ++d)
			x[d] = x[d] - pos[d];

		float d2 = 0;
		for (int d = 0; d < dim; ++d)
			d2 += x[d]*x[d];

		if (d2 > maxd2)
			continue;
		
		float xdotv = 0;
		float vdotv = 0;
		for (int d = 0; d < dim; ++d) {
			xdotv += x[d]*vel[d];
			vdotv += vel[d]*vel[d];
		}
		if (vdotv == 0) // if the sample point is not moving
			vdotv = 1e-5;

		float dotratio = xdotv/vdotv;
		for (int d = 0; d < dim; ++d)
			x[d] = x[d] - dotratio*vel[d];
		float xdotx = 0;
		for (int d = 0; d < dim; ++d) {
			as[hsIdx*dim + d] = x[d];
			xdotx += x[d]*x[d];
		}
		bs[hsIdx] = xdotx;
		countStoredHS++;
		hsIdx++;
	}
	countStoredHSs[motionWaypointIdx] = countStoredHS;
}

__global__ 
void cacheHalfspacesEdge(int numEdges, float *discMotions, 
	bool *isFreeEdges, bool *isFreeSamples,
	int obstaclesCount, float *obstacles, 
	int waypointsCount, float *as, float *bs, int *countStoredHSs, 
	float *sigma, float maxd2)
{
	// put obstacles into shared memory
	const int obstacleSize = 3*2*DIM;
	__shared__ float s_obstacles[obstacleSize];
	int obsStorageIdx = threadIdx.x;
	if (obsStorageIdx < obstacleSize)
		s_obstacles[obsStorageIdx] = obstacles[obsStorageIdx];
	__syncthreads();

	int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (edgeIdx >= numEdges)
		return;

	if (!isFreeEdges[edgeIdx])
		return;

	// TODO: put obstacles into shared?
	// how is it shared between blocks

	int dim = DIM/2; 

	float pos[DIM/2]; // sample point
	float vel[DIM/2]; // velocity at sample point
	// float l[DIM/2]; // lower obstacle corner
	// float u[DIM/2]; // upper obstacle corner 
	float x[DIM/2]; // returned a result

	// set the number of stored half spaces to zero
	for (int i = 0; i < waypointsCount; ++i) 
 		countStoredHSs[edgeIdx*waypointsCount + i] = 0;

	for (int obsIdx = 0; obsIdx < obstaclesCount; ++obsIdx) {
		// for (int d = 0; d < dim; ++d) { // load in obstacle
		// 	l[d] = obstacles[obsIdx*2*DIM + d];
		// 	u[d] = obstacles[obsIdx*2*DIM + DIM + d];
		// }

		for (int i = 0; i < waypointsCount; ++i) {
 			int motionWaypointIdx = edgeIdx*waypointsCount + i;
 			int hsIdx = motionWaypointIdx*obstaclesCount + countStoredHSs[motionWaypointIdx];

 			for (int d = 0; d < dim; ++d) {
				pos[d] = discMotions[motionWaypointIdx*DIM + d];
				vel[d] = discMotions[motionWaypointIdx*DIM + DIM/2 + d];
			}

			// solve the BVLS with last points solution (if it exists)
			// i == 0 ? bvls(x, NULL, pos, &s_obstacles[obsIdx*2*DIM], &s_obstacles[obsIdx*2*DIM + DIM], NULL) 
			// 	: bvls(x, NULL, pos, &s_obstacles[obsIdx*2*DIM], &s_obstacles[obsIdx*2*DIM + DIM], x);
			bvlsShortcut(x, NULL, pos, &s_obstacles[obsIdx*2*DIM], &s_obstacles[obsIdx*2*DIM + DIM], NULL); 

			// check if the half-space is relevant
			for (int d = 0; d < dim; ++d)
				x[d] = x[d] - pos[d];

			float d2 = 0;
			for (int d = 0; d < dim; ++d)
				d2 += x[d]*x[d];

			if (d2 > maxd2) // TODO: perhaps check at a distance covered by dt
				continue;

			float xdotv = 0;
			float vdotv = 0;
			for (int d = 0; d < dim; ++d) {
				xdotv += x[d]*vel[d];
				vdotv += vel[d]*vel[d];
			}
			if (vdotv == 0) // if the sample point is not moving
				vdotv = 1e-5;

			float dotratio = xdotv/vdotv;
			for (int d = 0; d < dim; ++d)
				x[d] = x[d] - dotratio*vel[d];
			float xdotx = 0;
			for (int d = 0; d < dim; ++d) {
				as[hsIdx*dim + d] = x[d];
				xdotx += x[d]*x[d];
			}
			bs[hsIdx] = xdotx;
			countStoredHSs[motionWaypointIdx]++;
		}
	}
}



__global__ 
void cacheHalfspacesAll(int halfspaceCount, float *discMotions, int obstaclesCount, 
	float *obstacles, int waypointsCount, float *as, float *bs, float *sigma, float maxd2)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= halfspaceCount)
		return;

	// TODO: put obstacles into shared?

	int motionWaypointIdx = tid / obstaclesCount;
	int obsIdx = tid % obstaclesCount;
	int dim = DIM/2; 

	float sample[DIM/2]; // sample point
	float v[DIM/2]; // velocity at sample point
	float l[DIM/2]; // lower obstacle corner
	float u[DIM/2]; // upper obstacle corner 
	float x[DIM/2]; // returned a result

	for (int d = 0; d < dim; ++d) {
		sample[d] = discMotions[motionWaypointIdx*DIM + d];
		v[d] = discMotions[motionWaypointIdx*DIM + DIM/2 + d];
		l[d] = obstacles[obsIdx*2*DIM + d];
		u[d] = obstacles[obsIdx*2*DIM + DIM + d];
	}

	bvls(x, NULL, sample, l, u, NULL);
	
	for (int d = 0; d < dim; ++d)
		x[d] = x[d] - sample[d];

	float d2 = 0;
	for (int d = 0; d < dim; ++d)
		d2 += x[d]*x[d];

	float xdotv = 0;
	float vdotv = 0;
	for (int d = 0; d < dim; ++d) {
		xdotv += x[d]*v[d];
		vdotv += v[d]*v[d];
	}
	if (vdotv == 0) // if the sample point is not moving
		vdotv = 1e-5;

	/*
	check mahalanobis distance
	*/

	float dotratio = xdotv/vdotv;
	for (int d = 0; d < dim; ++d)
		x[d] = x[d] - dotratio*v[d];

	float xdotx = 0;
	for (int d = 0; d < dim; ++d)
		xdotx += x[d]*x[d];
	for (int d = 0; d < dim; ++d)
		as[tid*dim + d] = x[d];
	bs[tid] = xdotx;

		// set as really far away
	if (d2 > maxd2)
		bs[tid] = 1112;
}

__global__ 
void cacheHalfspacesEdgeCentric(int numEdges, float *discMotions, int obstaclesCount, 
	float *obstacles, int waypointsCount, float *as, float *bs, float *sigma)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numEdges*obstaclesCount)
		return;

	// TODO: put obstacles into shared?

	int edgeIdx = tid / obstaclesCount;
	int obsIdx = tid % obstaclesCount;
	int dim = DIM/2; 

	float sample[DIM/2]; // sample point
	float v[DIM/2]; // velocity at sample point
	float l[DIM/2]; // lower obstacle corner
	float u[DIM/2]; // upper obstacle corner 
	float x[DIM/2]; // returned a result

	for (int i = 0; i < waypointsCount; ++i) {
		for (int d = 0; d < dim; ++d) {
			sample[d] = discMotions[edgeIdx*waypointsCount*DIM + i*DIM + d];
			v[d] = discMotions[edgeIdx*waypointsCount*DIM + DIM/2 + i*DIM + d];
			l[d] = obstacles[obsIdx*2*DIM + d];
			u[d] = obstacles[obsIdx*2*DIM + DIM + d];
		}

		bvls(x, NULL, sample, l, u, NULL);

		float xdotv = 0;
		float vdotv = 0;
		for (int d = 0; d < dim; ++d) {
			x[d] = x[d] - sample[d];
			xdotv += x[d]*v[d];
			vdotv += v[d]*v[d];
		}
		if (vdotv == 0) // if the sample point is not moving
			vdotv = 1e-8;

		/*
		check mahalanobis distance
		*/

		float dotratio = xdotv/vdotv;
		for (int d = 0; d < dim; ++d)
			x[d] = x[d] - dotratio*v[d];

		float xdotx = 0;	
		for (int d = 0; d < dim; ++d)
			xdotx += x[d]*x[d];
		for (int d = 0; d < dim; ++d)
			as[edgeIdx*waypointsCount*obstaclesCount*DIM/2 
				+ i*DIM/2*obstaclesCount + obsIdx*DIM/2 + d] = x[d];
		bs[edgeIdx*waypointsCount*obstaclesCount + i*obstaclesCount + obsIdx ] = xdotx;
	}
}

__global__ void setupBs(int halfspaceCount, float *bs, float *as) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= halfspaceCount)
		return;

	for (int d = 0; d < DIM/2; ++d)
		as[tid*DIM/2+d] = -1;

	bs[tid] = 1111;

}