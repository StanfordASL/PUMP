/*
motionPlanningProblem.cuh
Author: Brian Ichter

Struct definition for a motion planning problem.
*/

#pragma once

#include "roadMap.cuh"
#include "hsmc.cuh"
#include "helper.cuh"

#include <iostream>
#include <vector>

typedef struct MotionPlanningProblem {
	const char *filename;
	int dimW; // workspace dimension
	int dimX; // state space dimension
	std::vector<float> hi; // high bound on configuration space
	std::vector<float> lo; // low bound on configuration space

	float cpTarget; // target collision probability
	float cpFactor; // search factor around cp target
	float epsCost; // epsilon for dominated cost
	float epsCP; // epsilon for dominated CP
	int numHSMCParticles; // number of particles used for HSMC
	int numMCParticles; // number of Monte Carlo samples for CP calculation
	float lambda; // expansion cost threshold expansion rate (threshold += rn*lambda at each step)
	int numBuckets;
	float dt;

	int numDisc; // for collision checks, number of discretizations in edge representation
	int numObstacles; // 
	std::vector<float> obstacles;
	float *d_obstacles;

	std::vector<float> init; // initial state
	std::vector<float> goal; // goal state
	int initIdx; // index in samples of 
	int goalIdx; 

	// calculated
	float rnPerc; // percentile cutoff of for nn connections (e.g. 0.1 means 10th percentile)
	float rn; // nearest neighbor radius, as 100*rnPerc percentile connection cost

	RoadMap roadmap;
	HSMC hsmc;
	PUMPSearch pumpSearch;

	bool verbose;

	} MotionPlanningProblem;

void printMotionPlanningProblem(MotionPlanningProblem mpp, std::ostream& stream);
// sizeMotionPlanningProblem // once all ints for sizing are initialized, we size the vectors