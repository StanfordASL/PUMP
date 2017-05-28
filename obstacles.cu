#include "obstacles.cuh"

void generateObstacles(std::vector<float>& obstacles) 
{
	float obstaclesTmp[] = {
        0.05, 0.20, -3.1, -3.1, -3.1, -3.1, 0.45, 0.35, 3.1, 3.1, 3.1, 3.1,  
		0.70, 0.30, -3.1, -3.1, -3.1, -3.1, 0.90, 0.50, 3.1, 3.1, 3.1, 3.1,  
		0.30, 0.60, -3.1, -3.1, -3.1, -3.1, 0.80, 0.75, 3.1, 3.1, 3.1, 3.1};
    obstacles.assign(obstaclesTmp, obstaclesTmp + sizeof(obstaclesTmp)/sizeof(obstaclesTmp[0]));
}

void inflateObstacles(float *obstacles, float *obstaclesInflated, float inflateFactor, int obstaclesCount)
{
	for (int obs = 0; obs < obstaclesCount; ++obs) {
		for (int d = 0; d < 2*DIM; ++d) {
			if (d < DIM)
				obstaclesInflated[obs*2*DIM + d] = obstacles[obs*2*DIM + d] - inflateFactor;
			else
				obstaclesInflated[obs*2*DIM + d] = obstacles[obs*2*DIM + d] + inflateFactor;
		}
	}
}