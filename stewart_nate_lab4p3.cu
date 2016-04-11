/*
 * Name: Nate Steawrt
 * Date: 04-04-16
 * Description: Serial implementation of Matrix multiplication with transpose 
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define RANDOM_VALUE_MIN 1
#define RANDOM_VALUE_MAX 1000

#define MATRIX_DIM 1024 

/*
 * Calculate and return a random value between min and max.
 */
int randInt(int min, int max) {
	return rand() % max + min;
}

/*
 * Output the matrix to fout
 */
void outputMatrix(FILE *fout, int *matrix, int rows, int cols) {
	int i, j;

	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			fprintf(fout,  "%d ", *(matrix + i * cols + j));
		}
		fprintf(fout, "\n");
	}
}

__global__ void computeMath(int *matrix) {

	// Grab the two indices dependent on the block/thread structure
	int i = blockIdx.x;
	int k = blockIdx.y * blockDim.x + threadIdx.x;
	int register j = 0;

	// Declare pointers to the two arguments of the addition and the result pointer
	int register *result_ptr;
	int register *second_ptr;
	int register first_val;
	
	// Grab the initial values of the pointers and first val
	first_val = *(matrix + k * MATRIX_DIM + i);
	second_ptr = matrix + k * MATRIX_DIM + j;
	result_ptr = matrix + i * MATRIX_DIM + j;

	// Row traverse an entire row of two matrices
	for (; j < MATRIX_DIM; j++, result_ptr++, second_ptr++) {
		*result_ptr += first_val * *second_ptr;
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
	int nblocksX = 1024;
	int nblocksY = 4;
	int dimX = 256;

	// Declare the memory pointers
	int *h_matrix, *d_matrix;

	// Allocate memory for host and device
	size_t memSize = MATRIX_DIM * MATRIX_DIM * sizeof(*h_matrix);

	// Create space on the host and device for matrix
	h_matrix = (int *)malloc(memSize);
	cudaMalloc( (void**) &d_matrix, memSize);
	checkError();

	// Initialize the array
	int *m_ptr = h_matrix; // Setup a traversal pointer for the matrix
	for (i = 0; i < MATRIX_DIM; i++) {
		for (j = 0; j < MATRIX_DIM; j++, m_ptr++) {
			*m_ptr = randInt(RANDOM_VALUE_MIN, RANDOM_VALUE_MAX);
		}
	}
	cudaMemcpy(d_matrix, h_matrix, memSize, cudaMemcpyHostToDevice);
	checkError();

	// Set up grid and block structure
	dim3 dimGrid(nblocksX, nblocksY);
	dim3 dimBlock(dimX);

	// Launch the kernel and begin timer	
	time(&startTime);
	clockTime = clock();
	computeMath<<< dimGrid, dimBlock >>>(d_matrix);
	
	// Stop timer and retrieve results
	cudaMemcpy(h_matrix, d_matrix, memSize, cudaMemcpyDeviceToHost);
	checkError();
	time(&endTime);
	clockTime = clock() - clockTime;

	long numFloatingPointOperations = MATRIX_DIM * MATRIX_DIM;
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

