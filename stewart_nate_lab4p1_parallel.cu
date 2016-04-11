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

__global__ void computeMath(double *matrix) {
    
    int i;

    // Grab id of thread
    int threadId = blockDim.x * threadIdx.y + threadIdx.x + 1;
    
    // Declare pointers to the two arguments of the addition and the result pointer
	double *f_ptr, *first_ptr, *second_ptr;
	
	// Grab starting points for pointers
    f_ptr = matrix + threadId * NUM_COLS;
    first_ptr = matrix + (threadId - 1) * NUM_COLS + 1;
    second_ptr = f_ptr + 1;
    
    // Compute a single row
    for (i = 0; i < NUM_COLS - 1; i++, f_ptr++, first_ptr++, second_ptr++) {
        *f_ptr = *first_ptr + *second_ptr;
    }
}

/*
 * Check if an error occurred during the last CUDA command
 */
void checkError() {
	int errorCode = cudaGetLastError();

	if (errorCode != 0) {
		printf("Error %d occurred during last operation.\n", errorCode);
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
    int nblocks = 16;
    int dimX = 128;
    int dimY = 2;

    // Declare the memory pointers
    double *h_matrix, *d_matrix;
    
    // Allocate memory for host and device
    size_t memSize = NUM_ROWS * NUM_COLS * sizeof(*h_matrix);
	
	// Create space on the host and device for matrix
    h_matrix = (double *)malloc(memSize);
    cudaMalloc( (void**) &d_matrix, memSize);
	checkError();

	// Initialize the matrix and copy values into device
	double *f_ptr = h_matrix; // Setup a traversal pointer
	for (i = 0; i < NUM_ROWS; i++) {
		for (j = 0; j < NUM_COLS; j++, f_ptr++) {
			*f_ptr = randDouble(RANDOM_VALUE_MIN, RANDOM_VALUE_MAX);
		}
	}
    cudaMemcpy(d_matrix, h_matrix, memSize, cudaMemcpyHostToDevice);
	checkError();
    
    // Set up grid and block structure
    dim3 dimGrid(nblocks);
    dim3 dimBlock(dimX, dimY);
    
    // Launch the kernel and begin timer 
    time(&startTime);
	clockTime = clock();
    for (i = 0; i < 100; i++) {
		computeMath<<< dimGrid, dimBlock >>>(d_matrix);
		checkError();
	}
    
    // stop timer and retrieve results
    cudaMemcpy(h_matrix, d_matrix, memSize, cudaMemcpyDeviceToHost);
	checkError();
    time(&endTime);
	clockTime = clock() - clockTime;
    
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
	checkError();
}

