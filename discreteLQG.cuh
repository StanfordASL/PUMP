/*
discreteLQG.cuh
author: Brian Ichter

This file has the discrete LQG controller for a double integrator
*/

#pragma once

// class DiscreteLQG {
// public:
// 	// Settings
// 	float dt = 0.05;
// 	float tmax = 10:
// 	float nsf = 1.0;
// 	int Tmax = 200; // tmax/dt;

// 	// Double integrator dynamics
// 	float A[DIM][DIM]; 		// 6x6
// 	float B[DIM][DIM/2]; 	// 6x3
// 	float C[DIM/2][DIM]; 	// Controller output, 3x6
// 	float c[DIM];			// Drift, 6x1
// 	float Cws[DIM/2][DIM]; 	// C-Space to Workspace, 3x6

// 	// Controller costs
// 	float r;
// 	float R[DIM/2][DIM/2]; 	// Control effort, 6x6
// 	float Q[DIM][DIM];				// State regulator, 6x6
// 	float F[DIM][DIM];				// Final state penalty, 6x6

// 	// Double integrator noise
// 	float V[DIM][DIM]; 		// Process noise, 6x6
// 	float W[DIM][DIM];		// Measurement noise, 6x6
// 	float P0[DIM][DIM];		// Initial Uncertainty, 6x6

// 	// Calculated gains and uncertainties
// 	float P[DIM][DIM][Tmax+1];			
// 	float S[DIM][DIM][Tmax+1];
// 	float K[DIM][DIM][Tmax];
// 	float L[DIM/2][DIM][Tmax];			
// 	float Ncomb[2*DIM][2*DIM][Tmax+1]; 	// combined uncertainty
// 	float Acomb[2*DIM][2*DIM][Tmax]; 	// will only use T of this where T is the length of the followed path
// };