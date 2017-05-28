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
	int numObstacles, int maxHSCount, float *obstacles, 
	int waypointsCount, float *as, float *bs, int *countStoredHSs, 
	float *sigma, float maxd2)
{
	// put obstacles into shared memory
	extern __shared__ float s_obstacles[];

	int obsStorageIdx = threadIdx.x; // see warrning before kernel call
	if (obsStorageIdx < numObstacles*2*DIM)
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

	int dimW = DIM/2; 

	float pos[DIM/2]; // sample point
	float vel[DIM/2]; // velocity at sample point
	for (int d = 0; d < dimW; ++d) {
		pos[d] = discMotions[motionWaypointIdx*DIM + d];
		vel[d] = discMotions[motionWaypointIdx*DIM + DIM/2 + d];
	}

	// float l[DIM/2]; // lower obstacle corner
	// float u[DIM/2]; // upper obstacle corner 
	float x[DIM/2]; // returned a result

	int hsIdx = motionWaypointIdx*maxHSCount;
	int countStoredHS = 0;
	for (int obsIdx = 0; obsIdx < numObstacles; ++obsIdx) {
		// for (int d = 0; d < dimW; ++d) {
		// 	l[d] = s_obstacles[obsIdx*2*DIM + d];
		// 	u[d] = s_obstacles[obsIdx*2*DIM + DIM + d];
		// }

		bvlsShortcut(x, NULL, pos, &s_obstacles[obsIdx*2*DIM], &s_obstacles[obsIdx*2*DIM + DIM], NULL);

		/*
		check distance
		*/
		for (int d = 0; d < dimW; ++d)
			x[d] = x[d] - pos[d];

		float d2 = 0;
		for (int d = 0; d < dimW; ++d)
			d2 += x[d]*x[d];

		if (d2 > maxd2)
			continue;
		
		float xdotv = 0;
		float vdotv = 0;
		for (int d = 0; d < dimW; ++d) {
			xdotv += x[d]*vel[d];
			vdotv += vel[d]*vel[d];
		}
		if (vdotv == 0) // if the sample point is not moving, avoid divide by zero
			vdotv = 1e-5;

		float dotratio = xdotv/vdotv;
		for (int d = 0; d < dimW; ++d)
			x[d] = x[d] - dotratio*vel[d];
		float xdotx = 0;
		for (int d = 0; d < dimW; ++d) {
			as[hsIdx*dimW + d] = x[d];
			xdotx += x[d]*x[d];
		}
		bs[hsIdx] = xdotx;
		countStoredHS++;
		hsIdx++;

		// so we don't exceed the number of half-spaces, ideally this would actually check if the 
		// new half-space is closer than any previously stored
		if (countStoredHS == maxHSCount) 
			break;
	}
	countStoredHSs[motionWaypointIdx] = countStoredHS;
}

__global__ 
void cacheHalfspacesEdge(int numEdges, float *discMotions, 
	bool *isFreeEdges, bool *isFreeSamples,
	int numObstacles, int maxHSCount, float *obstacles, 
	int waypointsCount, float *as, float *bs, int *countStoredHSs, 
	float *sigma, float maxd2)
{
	// put obstacles into shared memory
	// const int obstacleSize = 15*2*DIM; // MUST UPDATE THIS
	extern __shared__ float s_obstacles[];

	int obsStorageIdx = threadIdx.x;
	if (obsStorageIdx < numObstacles*2*DIM)
		s_obstacles[obsStorageIdx] = obstacles[obsStorageIdx];
	__syncthreads();

	int edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (edgeIdx >= numEdges)
		return;

	if (!isFreeEdges[edgeIdx])
		return;

	// TODO: put obstacles into shared?
	// how is it shared between blocks

	int dimW = DIM/2; 

	float pos[DIM/2]; // sample point
	float vel[DIM/2]; // velocity at sample point
	// float l[DIM/2]; // lower obstacle corner
	// float u[DIM/2]; // upper obstacle corner 
	float x[DIM/2]; // returned a result

	// set the number of stored half spaces to zero
	for (int i = 0; i < waypointsCount; ++i) 
 		countStoredHSs[edgeIdx*waypointsCount + i] = 0;

	for (int obsIdx = 0; obsIdx < numObstacles; ++obsIdx) {
		// for (int d = 0; d < dimW; ++d) { // load in obstacle
		// 	l[d] = obstacles[obsIdx*2*DIM + d];
		// 	u[d] = obstacles[obsIdx*2*DIM + DIM + d];
		// }

		for (int i = 0; i < waypointsCount; ++i) {
			// so we don't exceed the number of half-spaces, ideally this would actually check if the 
			// new half-space is closer than any previously stored

 			int motionWaypointIdx = edgeIdx*waypointsCount + i;
 			int hsIdx = motionWaypointIdx*maxHSCount + countStoredHSs[motionWaypointIdx];

			if (countStoredHSs[motionWaypointIdx] >= maxHSCount) {
				break;
			}


 			for (int d = 0; d < dimW; ++d) {
				pos[d] = discMotions[motionWaypointIdx*DIM + d];
				vel[d] = discMotions[motionWaypointIdx*DIM + DIM/2 + d];
			}

			// solve the BVLS with last points solution (if it exists)
			// i == 0 ? bvls(x, NULL, pos, &s_obstacles[obsIdx*2*DIM], &s_obstacles[obsIdx*2*DIM + DIM], NULL) 
			// 	: bvls(x, NULL, pos, &s_obstacles[obsIdx*2*DIM], &s_obstacles[obsIdx*2*DIM + DIM], x);
			bvlsShortcut(x, NULL, pos, &s_obstacles[obsIdx*2*DIM], &s_obstacles[obsIdx*2*DIM + DIM], NULL); 

			// check if the half-space is relevant
			for (int d = 0; d < dimW; ++d)
				x[d] = x[d] - pos[d];

			float d2 = 0;
			for (int d = 0; d < dimW; ++d)
				d2 += x[d]*x[d];

			if (d2 > maxd2) // TODO: perhaps check at a distance covered by dt
				continue;

			float xdotv = 0;
			float vdotv = 0;
			for (int d = 0; d < dimW; ++d) {
				xdotv += x[d]*vel[d];
				vdotv += vel[d]*vel[d];
			}
			if (vdotv == 0) // if the sample point is not moving
				vdotv = 1e-5;

			float dotratio = xdotv/vdotv;
			for (int d = 0; d < dimW; ++d)
				x[d] = x[d] - dotratio*vel[d];
			float xdotx = 0;
			for (int d = 0; d < dimW; ++d) {
				as[hsIdx*dimW + d] = x[d];
				xdotx += x[d]*x[d];
			}
			bs[hsIdx] = xdotx;
			countStoredHSs[motionWaypointIdx]++;
		}
	}
}

__global__ void setupAsBs(int halfspaceCount, float *bs, float *as) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= halfspaceCount)
		return;

	for (int d = 0; d < DIM/2; ++d)
		as[tid*DIM/2+d] = -1;

	bs[tid] = 1111;

}