/*
bvls.cuh
author: Brian Ichter

This file containers a solver for the Bounded-Variable Least-Squares problem.
The solution methodology is adapted for CUDA and C from,
http://www.math.unl.edu/~tshores1/Public/Math496S06/MatlabTools/Examples/chap7/examp1/

The initial algorithm is based on the bounded variables least squares algorithm 
of Stark and Parker.  See P.B. Stark and R. L. Parker, "Bounded-Variable Least-Squares: An Algorithm
and Applications", Computational Statistics 10:129-141, 1995.
*/

#pragma once

#include "helper.cuh"

// x=bvls(A,b,l,u)

/* 
x is returned vector
A is transformation matrix, can be identity for just distance, 
 	or L inverse for Mahalanobis distance transformation
b is the point
l and u are upper and lower limits of the obstacle
xi is initial guess
*/
__device__
void bvls(float *x, float *A, float *b, float *l, float *u, float *xi);
__device__
void bvlsShortcut(float *x, float *A, float *b, float *l, float *u, float *xi);