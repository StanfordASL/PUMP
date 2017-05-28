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

void generateObstacles(std::vector<float>& obstacles); // fill obstacle vector
void inflateObstacles(float *obstacles, float *obstaclesInflated, float inflateFactor, int obstaclesCount);