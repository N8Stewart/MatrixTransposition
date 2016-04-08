/*
 * Name: Nate Steawrt
 * Date: 04-04-16
 * Description: Serial implementation of Matrix morphism
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define RANDOM_VALUE_MIN 1.0
#define RANDOM_VALUE_MAX 2.0

#define NUM_ROWS 4097
#define NUM_COLS 4097

/*
 * Calculate and return a random value between min and max.
 */
double randDouble(double min, double max) {
	double range =  max - min;
	double dist = RAND_MAX / range;
	return min + (rand() / dist);
}

/*
 * Output the matrix to fout
 */
void outputMatrix(FILE *fout, double *matrix, int rows, int cols) {
	int i, j;

	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			fprintf(fout,  "%lf ", *(matrix + i * cols + j));
		}
		fprintf(fout, "\n");
	}
}

int main(void) {

	// Declare the needed variables
	int i, j;
	
	// Variables for timing
	time_t startTime, endTime;
	clock_t clockTime;

	// Seed the random number generator
	srand(time(NULL));
    
    // Define thread hierarchy
    int nblocks = 1;
    int dimX = 512;
    int dimY = 8;
    
    // Declare the memory pointers
    double *h_matrix, *d_matrix;
    
    // Allocate memory for host and device
    size_t memSize = nblocks * dimX * dimY * sizeof(*h_matrix);
	
	// Create space on the host and device for matrix
    h_matrix = malloc(memSize);
    cudaMalloc( (void**) &d_matrix, memSize);

	// Initialize the matrix and copy values into device
	double *f_ptr = matrix; // Setup a traversal pointer
	for (i = 0; i < NUM_ROWS; i++) {
		for (j = 0; j < NUM_COLS; j++, f_ptr++) {
			*f_ptr = randDouble(RANDOM_VALUE_MIN, RANDOM_VALUE_MAX);
		}
	}
    cudaMemcpy(d_matrix, h_matrix, memSize, cudaMemcpyHostToDevice);
    
    // Set up grid and block structure
    dim3 dimGrid(nblocks);
    dim3 dimBlock(dimX, dimY);
    
    // Launch the kernel and begin timer 
    time(&startTime);
	clockTime = clock();
    for (i = 0; i < 100; i++) {
        computeMath<<< dimGrid, dimBlock >>>(d_matrix);
    }
    
    // stop timer and retrieve results
    time(&endTime);
	clockTime = clock() - clockTime;
    cudaMemcpy(h_matrix, d_matrix, memSize, cudaMemcpyDeviceToHost);
    
    // Compute estimated GFlops
	unsigned long long numFloatingPointOperations = 100 * (NUM_ROWS-1) * (NUM_COLS-1);
	double gflops = numFloatingPointOperations / ((double)clockTime/1000000) / 1000000000;
	printf("*********************************************************************\n");
	printf("Number of floating point operations:%ld\n", numFloatingPointOperations);
	printf("Estimated GFlops:%lf GFlops\n\n", gflops);
	printf("elapsed convergence loop time\t(clock): %lu\n", clockTime);
	printf("elapsed convergence loop time\t (time): %.f\n", difftime(endTime, startTime));
	printf("*********************************************************************\n");

	free(h_matrix);
    cudaFree(d_matrix);
}

__global__ void computeMath(double *matrix) {
    
    int i, j;
    
    // Grab id of thread
    int threadId = blockDim.x * threadIdx.y + threadIdx.x + 1;
    
    // Declare pointers to the two arguments of the addition
	double *f_ptr, *first_ptr, *second_ptr;
	
	// Grab starting points for pointers
    f_ptr = matrix + threadId * 4097;
    first_ptr = matrix + (threadId - 1) * 4097 + 1;
    second_ptr = f_ptr + 1;
    
    // Compute a single row
    for (j = 0; j < 4096; j++, f_ptr++, first_ptr++, second_ptr++) {
        *f_ptr = *first_ptr + *second_ptr;
    }
}
