/*
 * Name: Nate Steawrt
 * Date: 04-04-16
 * Description: Serial implementation of Matrix multiplication with transpose 
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define RANDOM_VALUE_MIN 1.0
#define RANDOM_VALUE_MAX 2.0

#define MATRIX_DIM 4096 

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

__global__ void computeMath(double *matrix, double *result) {

	// Grab the two indices dependent on the block/thread structure
	int i = blockIdx.x;
	int k = blockIdx.y * blockDim.x + threadIdx.x;
	int register j = 0;

	// Declare pointers to the two arguments of the addition and the result pointer
	double register *result_ptr;
	double register *second_ptr;
	double register first_val;
	
	// Grab the initial values of the pointers and first val
	first_val = *(matrix + k * MATRIX_DIM + i);
	second_ptr = matrix + k * MATRIX_DIM + j;
	result_ptr = result + i * MATRIX_DIM + j;

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
	
	// Define thread hierarchy
	int nblocksX = 4096;
	int nblocksY = 128;
	int dimX = 32;

	// Declare the memory pointers
	double *h_matrix, *d_matrix, *h_result, *d_result;

	// Allocate memory for host and device
	size_t memSize = MATRIX_DIM * MATRIX_DIM;

	// Create space on the host and device for matrix
	h_matrix = (double *)malloc(memSize * sizeof(*h_matrix));
	h_result = (double *)calloc(memSize, sizeof(*h_matrix));
	cudaMalloc( (void**) &d_matrix, memSize * sizeof(*h_matrix));
	checkError();
	cudaMalloc( (void**) &d_result, memSize * sizeof(*h_matrix));
	checkError();

	// Initialize the array
	double *m_ptr = h_matrix; // Setup a traversal pointer for the matrix
	for (i = 0; i < MATRIX_DIM; i++) {
		for (j = 0; j < MATRIX_DIM; j++, m_ptr++) {
			*m_ptr = randDouble(RANDOM_VALUE_MIN, RANDOM_VALUE_MAX);
		}
	}
	cudaMemcpy(d_matrix, h_matrix, memSize * sizeof(*h_matrix), cudaMemcpyHostToDevice);
	checkError();
	cudaMemcpy(d_result, h_result, memSize * sizeof(*h_matrix), cudaMemcpyHostToDevice);
	checkError();

	// Set up grid and block structure
	dim3 dimGrid(nblocksX, nblocksY);
	dim3 dimBlock(dimX);

	// Launch the kernel
	computeMath<<< dimGrid, dimBlock >>>(d_matrix, d_result);
	checkError();
	
	// Retrieve results
	cudaMemcpy(h_result, d_result, memSize * sizeof(*h_matrix), cudaMemcpyDeviceToHost);
	checkError();

	// Free memory and make sure it completes without error
	free(h_matrix);
	free(h_result);
	cudaFree(d_matrix);
	checkError();
	cudaFree(d_result);
	checkError();
}

