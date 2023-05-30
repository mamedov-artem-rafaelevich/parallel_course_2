#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>

#define MAX_ITER 1000000
#define EPSILON 1e-6

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <grid_size> <max_iter> <epsilon>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int size = atoi(argv[1]);
    const int max_iter = atoi(argv[2]);
    const double epsilon = atof(argv[3]);

    double *u = (double *) malloc(size * size * sizeof(double));
    double *u_new = (double *) malloc(size * size * sizeof(double));
    double *f = (double *) malloc(size * size * sizeof(double));
    double *boundary = (double *) malloc(size * size * sizeof(double));

    // Convert 2D indexing to 1D indexing
    #define IDX(i, j) ((i) * size + (j))

    // Set boundary conditions
    const double top_left = 10.0;
    const double top_right = 20.0;
    const double bottom_left = 30.0;
    const double bottom_right = 20.0;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == 0 && j == 0) {
                boundary[IDX(i, j)] = top_left;
            } else if (i == 0 && j == size - 1) {
                boundary[IDX(i, j)] = top_right;
            } else if (i == size - 1 && j == 0) {
                boundary[IDX(i, j)] = bottom_left;
            } else if (i == size - 1 && j == size - 1) {
                boundary[IDX(i, j)] = bottom_right;
            } else {
                boundary[IDX(i, j)] = NAN;
            }
        }
    }

    // Initialize u and f
    const double dx = 1.0 / (size - 1);
    const double dy = 1.0 / (size - 1);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            u[IDX(i, j)] = 0.0;
            u_new[IDX(i, j)] = 0.0;
            f[IDX(i, j)] = 0.0;
        }
    }

    // Set f
    for (int i = 1; i < size - 1; i++) {
        for (int j = 1; j < size - 1; j++) {
            f[IDX(i, j)] = -2.0 * M_PI * M_PI * sin(M_PI * i * dx) * sin(M_PI * j * dy);
        }
    }

    // Initialize iteration variables
    int iter = 0;
    double error = NAN;
    double local_error = NAN;

    // Start iterations
    while (iter < max_iter && (isnan(error) || error > epsilon)) {
        // Copy u to u_new
        #pragma acc enter data copyin(u[:size*size]) create(u_new[:size*size])
        #pragma acc parallel loop collapse(2) present(u[:size*size], u_new[:size*size])
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                u_new[IDX(i, j)] = u[IDX(i, j)];
            }
        }

        // Update u_new
        error = 0.0;
        #pragma acc parallel loop collapse(2) present(u[:size*size], u_new[:size*size], f[:size*size], boundary[:size*size]) reduction(max:error) private(local_error)
        for (int i = 1; i < size - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                double u_old = u[IDX(i, j)];
                u_new[IDX(i, j)] = (u[IDX(i - 1, j)] + u[IDX(i + 1, j)] + u[IDX(i, j - 1)] + u[IDX(i, j + 1)] + dx * dx * f[IDX(i, j)]) / 4.0;
                if (!isnan(boundary[IDX(i, j)])) {
                    u_new[IDX(i, j)] = boundary[IDX(i, j)];
                }
                local_error = fabs(u_new[IDX(i, j)] - u_old);
                error = fmax(error, local_error);
            }
        }

        // Copy u_new to u
        #pragma acc parallel loop collapse(2) present(u[:size*size], u_new[:size*size])
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                u[IDX(i, j)] = u_new[IDX(i, j)];
            }
        }

        #pragma acc exit data copyout(u[:size*size]) delete(u_new[:size*size])

        iter++;
    }

    // Print results
    printf("Iterations: %d\n", iter);
    printf("Error: %e\n", error);

    // Free memory
    free(u);
    free(u_new);
    free(f);
    free(boundary);

    return EXIT_SUCCESS;
}