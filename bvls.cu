#include "bvls.cuh"

// may be able to limit divergence by splitting out dimensions or perhaps obstacles/waypoints
// TODO: go through the succcessive if loops and combine to limit divergence
// TODO: speed ups because we use A as identity

__device__
void bvls(float *x, float *A, float *b, float *l, float *u, float *xi) {
	// dim is the dimension for the bvls problem (i.e. double integrator in 3D is 3, not 6)
	int dim = DIM/2;

	bool oopslist[DIM/2];
	int state[DIM/2];
	bool atbound[DIM/2];
	bool between[DIM/2];
	int criti = -1;
	int crits = 0;
	float eps = 1.0e-8;
	for (int d = 0; d < dim; ++d) {
		oopslist[d] = false;
		state[d] = 0;
		atbound[d] = false;
		between[d] = false;
	}	

	// setup initial solution if xi is unfilled	
	if (xi == NULL) {
		for (int d = 0; d < dim; ++d) {
			// initial implementation checks if bounds are at infinity, but we do not expect this behavior for obstacles
			if (abs(l[d]) <= abs(u[d])) {
			// if (b[d] <= l[d]) {
				x[d] = l[d];
				state[d] = 1;
				atbound[d] = true;
			} else {
				x[d] = u[d];
				state[d] = 2;
				atbound[d] = true;
			}
		}
	} else { // if xi is filled, set the other arrays, TODO: there appears to be an error here, i'd guess with how I set the state
		for (int d = 0; d < dim; ++d) {
			x[d] = xi[d];
			state[d] = 2;
			if (x[d] == l[d]) {
				state[d] = 1;
				atbound[d] = true;
			} else if (x[d] == u[d]) {
				state[d] = 2;
				atbound[d] = true;
			}
		}
	}
	
	// solve bvls loop
	int iter = 0;
	while (iter < 10*dim) {
		++iter;

		// float Atrans[DIM/2*DIM/2];
		// float Ax[DIM/2];
		// float Axmb[DIM/2];
		float grad[DIM/2];// = transpose(A)*(A*x-b); 
		// TODO: shortcutted since A = eye(DIM/2), grad = A'*(A*x-b) = (A*x-b) = x-b
		// transpose(Atrans, A, dim, dim);
		// multiplyArrays(A, x, Ax, dim, dim, dim, 1);
		// subtractArrays(Ax, b, Axmb, dim, 1);
		// multiplyArrays(Atrans, Axmb, grad, dim, dim, dim, 1);
		for (int d = 0; d < dim; ++d)
			grad[d] = x[d] - b[d];
		for (int d = 0; d < dim; ++d) // ignore variables already tried and failed
			if (oopslist[d]) 
				grad[d] = 0;
	
		// check for optimality
		int done = 1;
		for (int d = 0; d < dim; ++d) {
			float normb = 0;
			for (int i = 0; i < dim; ++i)
				normb += b[i]*b[i];
			if ((abs(grad[d]) > (1+normb)*eps) && state[d] == 0)
				done = 0;
			else if (grad[d] < 0 && state[d] == 1)
				done = 0;
			else if (grad[d] > 0 && state[d] == 2)
				done = 0;
		}
		if (done == 1)
			return;

		// not optimal
		int newi = -1;
		float newg = 0;

		for (int d = 0; d < dim; ++d){
			if (atbound[d] == false)
				continue;
			if (d == criti) // don't free if just locked at bound
				continue;

			if (grad[d] > 0 && state[d] == 2) {
				if (abs(grad[d]) > newg) {
					newi = d;
					newg = abs(grad[d]);
				}
			}
			if (grad[d] < 0 && state[d] == 1) {
				if (abs(grad[d]) > newg) {
					newi = d;
					newg = abs(grad[d]);
				}
			}
		}

		// free locked variable with largest gradient
		if (newi != -1) {
			atbound[newi] = false;
			state[newi] = 0;
			between[newi] = true;
		}

		// construct the new projected problem
		int countBetween = 0;
		int countAtbound = 0;
		for (int d = 0; d < dim; ++d) {
			if (between[d])
				++countBetween;
			if (atbound[d])
				++countAtbound;
		}

		// load up arrays for the projected problem
		// may not use all rows, but probably better to preallocate since it is small
		// float Aproj[DIM/2*DIM/2]; 
		// float An[DIM/2*DIM/2];
		// int rowAproj = 0;
		// int rowAn = 0;
		// // TODO fix to put columns in Aproj and An
		// for (int d = 0; d < dim; ++d) {
		// 	if (between[d]) {
		// 		for (int i = 0; i < dim; ++i) {
		// 			Aproj[d*rowAproj + i] = A[d*dim + i];
		// 		}
		// 		++rowAproj;
		// 	}
		// 	if (atbound[d]) {
		// 		for (int i = 0; i < dim; ++i) {
		// 			An[d*rowAn + i] = A[d*dim + i];
		// 		}
		// 		++rowAn;
		// 	}
		// }

		float bproj[DIM/2];
		for (int d = 0; d < dim; ++d)
			bproj[d] = 0;
		// TODO, correctly implement bproj = b - An*x(atbound)
		// currently using shortcut since A is eye(dim)
		if (countAtbound > 0)
			for (int d = 0; d < dim; ++d)
				if (atbound[d])
					bproj[d] = b[d] - x[d];
		else
			for (int d = 0; d < dim; ++d)
				bproj[d] = b[d];

		// solve the projected problem
		float z[DIM/2];
		for (int d = 0; d < dim; ++d)
			z[d] = bproj[d];

		// z = Aproj\bproj; // TODO = inv(Aproj)*bproj
		// TODO: implement correctlly currently using shortcut since Aproj is identity minus rows
		float xnew[DIM/2];
		for (int d = 0; d < dim; ++d) {
			if (atbound[d])
				xnew[d] = x[d];
			else if (between[d])
				xnew[d] = z[d];
			else 
				xnew[d] = 0;
		}

		// freed variable is trying to go beyond bound, find new one
		if ((newi != -1) && (((xnew[newi] <= l[newi]) && (x[newi]==l[newi])) ||
      		((xnew[newi] >= u[newi]) && (x[newi]==u[newi])))) {
			oopslist[newi] = true;
			if (xnew[newi] <= l[newi] && state[newi] == 1) {
				state[newi] = 1;
				x[newi] = l[newi];
			}
			if (xnew[newi] >= u[newi] && state[newi] == 2) {
				state[newi] = 2;
				x[newi] = u[newi];
			}
			atbound[newi] = true;
			between[newi] = false;
			continue;
		}

		// good variable freed, reset the oopslist
		for (int d = 0; d < dim; ++d)
			oopslist[d] = false;

		// move towards the optimal solution to the proj problem
		float alpha = 1;
		float newalpha = 0;
		for (int d = 0; d < dim; ++d) {
			if (!between[d])
				continue;

			if (xnew[d] > u[d]) {
				newalpha = alpha < (u[d] - x[d]) / (xnew[d] - x[d]) ? alpha : (u[d] - x[d]) / (xnew[d] - x[d]);
				if (newalpha > alpha) {
					criti = d;
					crits = 2;
					alpha = newalpha;
				}
			}
			if (xnew[d] < l[d]) {
				newalpha = alpha < (l[d] - x[d]) / (xnew[d] - x[d]) ? alpha : (l[d] - x[d]) / (xnew[d] - x[d]);
				if (newalpha < alpha) {
					criti = d;
					crits = 1;
					alpha = newalpha;
				}
			}
		}

		// take the step: x = x + alpha*(xnew - x);
		for (int d = 0; d < dim; ++d)
			x[d] += alpha*(xnew[d] - x[d]);

		// update the state of variables
		if (alpha < 1) {
			between[criti] = false;
			atbound[criti] = true;
			state[criti] = crits;
		}

		for (int d = 0; d < dim; ++d) {
			if (x[d] >= u[d] || x[d] <= l[d]) {
				x[d] = (x[d] >= u[d]) ? u[d] : l[d];
				state[d] = (x[d] >= u[d]) ? 2 : 1;
				bool betweenEmpty = true;
				for (int i = 0; i < dim; ++i) {
					betweenEmpty = betweenEmpty && !between[i];
				}
				if (betweenEmpty || !between[d])
					between[d] = false;
				if (!between[d])
					atbound[d] = true;
			}
		}
	}
}


__device__
void bvlsShortcut(float *x, float *A, float *b, float *l, float *u, float *xi) {
	// dim is the dimension for the bvls problem (i.e. double integrator in 3D is 3, not 6)
	int dim = DIM/2;

	for (int d = 0; d < dim; ++d) {
		if (b[d] <= l[d]) { // if it is below the lower side of the obs
			x[d] = l[d];
		} else if (b[d] >= u[d]) { // if it is above the upper side of the obs
			x[d] = u[d];
		} else { // in between the side of the obstacle
			x[d] = b[d];
		}
	}
}