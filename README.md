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
