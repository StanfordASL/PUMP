/*
collisionProbability.cuh
author: Brian Ichter

Hard coded array values for path and offset from path according
to an LQG method. These particles track a nominal motion plans, which 
is then tracked by all candidate motion plans.
*/

#pragma once

const int Tmax = 100;
const int numMCSamples = 2048;

extern float xcomb[numMCSamples*Tmax*DIM/2]; // offset from nominal path
// accessed as xcomb[particleIdx*Tmax*DIM/2 + timeIdx*DIM/2 + dimensionIdx]

extern float xcombTime[Tmax*numMCSamples*DIM/2]; // offset from nominal path, stored flipped from xcomb
// accessed as xcombTime[timeIdx*numMCSamples*DIM/2 + particleIdx*DIM/2 + dimensionIdx]