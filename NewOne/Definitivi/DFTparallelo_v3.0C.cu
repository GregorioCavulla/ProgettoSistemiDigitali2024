#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846

// Struttura per rappresentare un numero complesso con allineamento
typedef struct __align__(8) {
    float real;
    float imag;
} Complesso;

// Kernel per la trasformata discreta di Fourier (DFT)
__global__ void dftKernel(const float *x, Complesso *X, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        X[i].real = 0;
        X[i].imag = 0;
        for (int j = 0; j < N; j++) {
            float angle = 2 * PI * i * j / N;
            X[i].real += x[j] * cos(angle);
            X[i].imag -= x[j] * sin(angle);
        }
    }
}

// Kernel per applicare un filtro passa-basso
__global__ void filtro(Complesso *X, int N, int fc, int fs) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < N) {
        int cutoffIndex = (fc * N) / fs;
        if (k > cutoffIndex && k < N - cutoffIndex) {
            X[k].real = 0;
            X[k].imag = 0;
        }
    }
}

// Kernel per la trasformata inversa discreta di Fourier (IDFT)
__global__ void idftKernel(const Complesso *X, float *x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] = 0;
        for (int j = 0; j < N; j++) {
            float angle = 2 * PI * i * j / N;
            x[i] += X[j].real * cos(angle) - X[j].imag * sin(angle);
        }
        x[i] /= N;
    }
}

// Funzione principale
int main(int argc, char *argv[]) {
    float *x, *y;
    Complesso *X;
    int N = 1024; // Dimensione di esempio
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Allocazione memoria host
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));
    X = (Complesso *)malloc(N * sizeof(Complesso));

    // Inizializzazione dati di esempio
    for (int i = 0; i < N; i++) {
        x[i] = sin(2 * PI * i / N);
    }

    // Allocazione memoria device
    float *d_x, *d_y;
    Complesso *d_X;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_X, N * sizeof(Complesso));

    // Copia dati host -> device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Esecuzione DFT
    dftKernel<<<gridSize, blockSize>>>(d_x, d_X, N);
    cudaDeviceSynchronize();

    // Applicazione filtro
    filtro<<<gridSize, blockSize>>>(d_X, N, 1000, 44100);
    cudaDeviceSynchronize();

    // Esecuzione IDFT
    idftKernel<<<gridSize, blockSize>>>(d_X, d_y, N);
    cudaDeviceSynchronize();

    // Copia dati device -> host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Pulizia memoria
    free(x);
    free(y);
    free(X);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_X);

    printf("Calcolo completato con struttura allineata.\n");
    return 0;
}
