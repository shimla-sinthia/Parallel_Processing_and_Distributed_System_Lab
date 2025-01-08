#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to print a matrix
void display(int rows, int cols, int matrix[rows][cols]) {
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int K = 100, M = 50, N = 50, P = 50;
    // if(rank == 0) {
    //     printf("Enter Number of Matrices: ");
    //     scanf("%d", &K);
    //     printf("Enter Number of Rows in Matrix A: ");
    //     scanf("%d", &M);
    //     printf("Enter Number of Columns in Matrix A: ");
    //     scanf("%d", &N);
    //     printf("Enter Number of Columns in Matrix B: ");
    //     scanf("%d", &P);
    // }

    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(K % size != 0) {
        printf("Number of matrices must be divisible by the number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    int A[K][M][N], B[K][N][P], R[K][M][P];

    // Initialize the matrices in the root process
    if(rank == 0) {
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < M; i++) {
                for(int j = 0; j < N; j++) {
                    A[k][i][j] = rand() % 100;
                }
            }
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < P; j++) {
                    B[k][i][j] = rand() % 100;
                }
            }
        }
    }    

    // Buffer to store portion of the matrices assigned to each process
    int localA[K / size][M][N], localB[K / size][N][P], localR[K / size][M][P];
    MPI_Scatter(A, (K / size) * M * N, MPI_INT, localA, (K / size) * M * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, (K / size) * N * P, MPI_INT, localB, (K / size) * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    // Matrix multiplication
    for(int k = 0; k < (K / size); k++) {
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < P; j++) {
                localR[k][i][j] = 0;
                for(int l = 0; l < N; l++) {
                    localR[k][i][j] += (localA[k][i][l] * localB[k][l][j]) % 100;
                }
                localR[k][i][j] %= 100;
            }
        }
    }

    double endTime = MPI_Wtime();

    // Gather result matrices from all processes to the root process
    MPI_Gather(localR, (K / size) * M * P, MPI_INT, R, (K / size) * M * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Remove the comment to print result matrices
    //Print all the result matrices
    // if(rank == 0) {
    //     for(int k = 0; k < K; k++) {
    //         printf("Result Matrix R%d\n", k);
    //         display(M, P, R[k]);
    //     }
    // }

    // Barrier to synchronize all processes before timing starts
    MPI_Barrier(MPI_COMM_WORLD);

    // Print timing information for each process
    printf("Process %d: Time taken = %f seconds\n", rank, endTime - startTime);

    MPI_Finalize();
    return 0;
}