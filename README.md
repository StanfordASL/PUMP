# Parallel Uncertainty-aware Multiobjective Planning

This repo contains the code for "Real-Time Stochastic Kinodynamic Motion Planning via Multiobjective Search on GPUs," which presents the parallel uncertainty-aware multiobjective planning algorithm submitted to ICRA '17 for solving the stochastic kinodynamic motion planning problem.

It is written in CUDA C and setup to run a 3D double integrator (in a 6D configuration space). PUMP can be compiled with 
`$ make pump NUM=<sample count>` (e.g. `$ make pump NUM=4000`) 
and run with 
`$ ./pump <input file name> <cp target> <eta>` (e.g., `$ ./pump input.txt 0.01 2`). This runs out of the mainPUMP.cu main file. 

The input file contains all the necessary run settings for a planning problem. It is structured as a csv where each line respectively represents initial state, goal state, state space lower bound, state space upper bound, noise multiplier, number of obstacles, and the obstacles.
The obstacles are enumerated as bounding boxes in 6D with the lowest corner listed first and the highest corner listed second.

# Disclaimer

Though it can be run it is intended primarily for reference purposes and is currently a mess. Feel free to reach out to me if you're interested in the code and I may be able to provide a more up to date code, or at least some advice on what should be expanded on and what was written quickly just to get the job done. It may be best to reference this [disclaimer](https://github.com/schmrlng/MotionPlanning.jl), then add a bit more alpha. 

# License
Copyright 2017 Brian Ichter

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
