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
	int i, j, k;
	
	// Create space on the heap for matrix
	double *matrix = malloc(sizeof(*matrix) * NUM_ROWS * NUM_COLS); 

	// Initialize the array
	double *f_ptr = matrix; // Setup a traversal pointer
	for (i = 0; i < NUM_ROWS; i++) {
		for (j = 0; j < NUM_COLS; j++, f_ptr++) {
			*f_ptr = randDouble(RANDOM_VALUE_MIN, RANDOM_VALUE_MAX);
		}
	}

	// Declare pointers to the two arguments of the addition
	double *first_ptr, *second_ptr;

	// Compute math
	for (k = 0; k < 100; k++) {
		for (i = 1; i < NUM_ROWS; i++) {
			// Hold pointers to the importan pieces to cut down on flop
			f_ptr = matrix + i * NUM_COLS;
			first_ptr = matrix + (i - 1) * NUM_COLS + 1;
			second_ptr = f_ptr + 1;
			for (j = 0; j < NUM_COLS - 1; j++, f_ptr++, first_ptr++, second_ptr++) {
				*f_ptr = *first_ptr + *second_ptr;
			}
		}
	}

	free(matrix);

}

