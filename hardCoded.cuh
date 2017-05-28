/*
collisionProbability.cuh
author: Brian Ichter

Hard coded array values for path and offset from path according
to an LQG method (calculated in Matlab code in the folder 
referenceCode).
*/

#pragma once

const int Tmax = 100;
const int numMCParticles = 2048;

extern float xcomb[numMCParticles*Tmax*DIM/2]; // offset from nominal path
// accessed as xcomb[particleIdx*Tmax*DIM/2 + timeIdx*DIM/2 + dimensionIdx]

extern float xcombTime[Tmax*numMCParticles*DIM/2]; // offset from nominal path, stored flipped from xcomb
// accessed as xcombTime[timeIdx*numMCParticles*DIM/2 + particleIdx*DIM/2 + dimensionIdx]