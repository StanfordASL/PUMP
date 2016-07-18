/* 
obstacles.cuh
author: Brian Ichter

Generates hyperrectangular obstacle sets
*/

#pragma once

#include <stdio.h>
#include <vector>
#include <iostream>

#include "helper.cuh"

int getObstaclesCount(); // return number of obstacles
void generateObstacles(float* obstacles, int obstaclesLength); // create obstacle array
void inflateObstacles(float *obstacles, float *obstaclesInflated, float inflateFactor, int obstaclesCount);