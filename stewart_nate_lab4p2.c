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

int main(void) {

	// Declare the needed variables
	int i, j, k;
	
	// Variables for timing
	time_t startTime, endTime;
	clock_t clockTime;

	// Seed the random number generator
	srand(time(NULL));
	
	// Create space on the heap for matrix and transpose
	double *matrix = malloc(sizeof(*matrix) * MATRIX_DIM * MATRIX_DIM); 
	double *result = malloc(sizeof(*result) * MATRIX_DIM * MATRIX_DIM); 

	// Initialize the array
	double *m_ptr = matrix; // Setup a traversal pointer for the matrix
	for (i = 0; i < MATRIX_DIM; i++) {
		for (j = 0; j < MATRIX_DIM; j++, m_ptr++) {
			*m_ptr = randDouble(RANDOM_VALUE_MIN, RANDOM_VALUE_MAX);
		}
	}

	// Declare pointers to the two arguments of the addition
	double *first_ptr, *second_ptr;
	
	// Set the starting location for the pointers
	first_ptr = second_ptr = matrix;
	m_ptr = result;

	// Do math
	time(&startTime);
	clockTime = clock();
	for (i = 0; i < MATRIX_DIM; i++) {
		printf("Iteration %5d%5d%5d | result = %p\n", i, j, k, m_ptr);
		for (j = 0; j < MATRIX_DIM; j++, m_ptr++) {
			first_ptr = matrix + i;
			second_ptr = matrix + j;
			for (k = 0; k < MATRIX_DIM; k++, (first_ptr += MATRIX_DIM), (second_ptr += MATRIX_DIM)) {
				*m_ptr += *first_ptr * *second_ptr;
			}
		}
	}

	// Get end times and output results
	time(&endTime);
	clockTime = clock() - clockTime;

	//outputMatrix(stdout, matrix, MATRIX_DIM, MATRIX_DIM);
	//outputMatrix(stdout, result, MATRIX_DIM, MATRIX_DIM);

	unsigned long long numFloatingPointOperations = (MATRIX_DIM - 1);
	numFloatingPointOperations *= (MATRIX_DIM-1);
	numFloatingPointOperations *= (MATRIX_DIM-1);
	numFloatingPointOperations *= 2;
	double gflops = numFloatingPointOperations / ((double)clockTime/1000000) / 1000000000;

	printf("*********************************************************************\n");
	printf("Number of floating point operations:%ld\n", numFloatingPointOperations);
	printf("Estimated GFlops:%lf GFlops\n\n", gflops);
	printf("elapsed convergence loop time\t(clock): %lu\n", clockTime);
	printf("elapsed convergence loop time\t (time): %.f\n", difftime(endTime, startTime));
	printf("*********************************************************************\n");

	free(matrix);
	free(result);

}

