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
	int col = blockIdx.x;
	int row = blockIdx.y * blockDim.x + threadIdx.x;

	// Only transpose if the column id is greater than the row id
	if (col > row) {
		int *transpose = matrix + col * MATRIX_DIM + row;
		int *result = matrix + row * MATRIX_DIM + col;
		int temp = *transpose;
		*transpose = *result;
		*result = temp;
	}

}

/*
 * Verify the transpose is correct and output to console if it is/is not
 */ 
void verifyTranspose(int *matrix, int *results) {
	int i, j;

	int *m_ptr = matrix; // Setup a traversal pointer for the matrix
	for (i = 0; i < MATRIX_DIM; i++) {
		for (j = 0; j < MATRIX_DIM; j++, m_ptr++) {
			if (*m_ptr != *(results + j * MATRIX_DIM + i)) {
				printf("Transpose Incorrect.\n");
				return;
			}
		}
	}
	printf("Transpose Correct.\n");
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
	int nblocksX = 1024;
	int nblocksY = 32;
	int dimX = 32;

	// Declare the memory pointers
	int *h_matrix, *d_matrix, *h_results;

	// Allocate memory for host and device
	size_t memSize = MATRIX_DIM * MATRIX_DIM * sizeof(*h_matrix);

	// Create space on the host and device for matrix
	h_matrix = (int *)malloc(memSize);
	h_results = (int *)malloc(memSize);
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

	// Compute the transpose
	computeMath<<< dimGrid, dimBlock >>>(d_matrix);
	
	// Stop timer and retrieve results
	cudaMemcpy(h_results, d_matrix, memSize, cudaMemcpyDeviceToHost);
	checkError();

	// Verify transpose and free memory
	verifyTranspose(h_matrix, h_results);
	free(h_matrix);
	free(h_results);
	cudaFree(d_matrix);
	checkError();
}

