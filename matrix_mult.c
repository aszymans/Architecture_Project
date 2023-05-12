#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100 // Change N to adjust the size of the arrays

int main() {
  int A[N][N], B[N][N], C[N][N];
  int i, j, k;

  // Seed the random number generator
  srand(time(NULL));

  // Initialize the arrays with random values
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      A[i][j] = rand() % 10;
      B[i][j] = rand() % 10;
    }
  }

  // Multiply the arrays element-wise and store the result in C
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i][j] = 0;
      for (k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return 0;
}
