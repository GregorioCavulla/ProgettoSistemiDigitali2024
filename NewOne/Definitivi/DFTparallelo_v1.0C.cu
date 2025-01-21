//filtro host con i double

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sys/stat.h>
#include <cuda.h>

#define PI 3.14159265358979323846

// Struttura per rappresentare un numero complesso
typedef struct {
    double real;
    double imag;
} Complesso;

// Funzione per creare una stringa di timestamp
void createTimestamp(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    strftime(buffer, size, "%Y%m%d_%H%M%S", t);
}

// Funzione per leggere l'intestazione di un file .wav e determinare la lunghezza
int getWavFileLength(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Errore nell'apertura del file %s\n", filename);
        exit(1);
    }

    uint8_t header[44];
    fread(header, sizeof(uint8_t), 44, file);
    int dataSize = header[40] | (header[41] << 8) | (header[42] << 16) | (header[43] << 24);
    fclose(file);
    return dataSize / sizeof(int16_t);
}

// Funzione per leggere i campioni audio da un file .wav
void readWavFile(const char *filename, double *x, int N) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Errore nell'apertura del file %s\n", filename);
        exit(1);
    }

    fseek(file, 44, SEEK_SET);
    int16_t *buffer = (int16_t *)malloc(N * sizeof(int16_t));
    fread(buffer, sizeof(int16_t), N, file);
    for (int i = 0; i < N; i++) {
        x[i] = (double)buffer[i];
    }

    free(buffer);
    fclose(file);
}

// Funzione per scrivere un file .wav con l'intestazione
void writeWavFile(const char *filename, double *x, int N) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Errore nell'apertura del file %s\n", filename);
        exit(1);
    }

    uint8_t header[44] = {
        'R', 'I', 'F', 'F',
        0, 0, 0, 0,
        'W', 'A', 'V', 'E',
        'f', 'm', 't', ' ',
        16, 0, 0, 0,
        1, 0,
        1, 0,
        0x44, 0xAC, 0x00, 0x00,
        0x88, 0x58, 0x01, 0x00,
        2, 0,
        16, 0,
        'd', 'a', 't', 'a',
        0, 0, 0, 0
    };

    int dataSize = N * sizeof(int16_t);
    int fileSize = 44 + dataSize - 8;
    header[4] = (fileSize & 0xFF);
    header[5] = ((fileSize >> 8) & 0xFF);
    header[6] = ((fileSize >> 16) & 0xFF);
    header[7] = ((fileSize >> 24) & 0xFF);
    header[40] = (dataSize & 0xFF);
    header[41] = ((dataSize >> 8) & 0xFF);
    header[42] = ((dataSize >> 16) & 0xFF);
    header[43] = ((dataSize >> 24) & 0xFF);

    fwrite(header, sizeof(uint8_t), 44, file);

    int16_t *buffer = (int16_t *)malloc(N * sizeof(int16_t));
    for (int i = 0; i < N; i++) {
        buffer[i] = (int16_t)x[i];
    }
    fwrite(buffer, sizeof(int16_t), N, file);

    free(buffer);
    fclose(file);
}

__global__ void dftKernel(const double *x, Complesso *X, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        X[i].real = 0;
        X[i].imag = 0;
        for(int j = 0; j < N; j++){
            double angle = 2 * PI * i * j / N;
            X[i].real += x[j] * cos(angle);
            X[i].imag -= x[j] * sin(angle);
        }
    }

}

void filtro(Complesso *X, int N, int fc, int fs) {
    // Calcolo dell'indice di taglio corrispondente alla frequenza fc
    int cutoffIndex = (int)((fc * N) / fs);
    
    for (int k = 0; k < N; k++) {
        // Applica il filtro passa-basso
        if (k > cutoffIndex && k < N - cutoffIndex) {
            X[k].real = 0;
            X[k].imag = 0;
        }
    }
}

__global__ void idftKernel(const Complesso *X, double *x, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        x[i] = 0;
        for(int j = 0; j < N; j++){
            double angle = 2 * PI * i * j / N;
            x[i] += X[j].real * cos(angle) - X[j].imag * sin(angle);
        }
        x[i] /= N;
    }

}

// Funzione per scrivere un report dei tempi di esecuzione
void writeReport(const char *filename, double dftTime, double filterTime, double idftTime, double totalTime) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Errore nell'apertura del file %s\n", filename);
        exit(1);
    }

    fprintf(file, "Report tempi di esecuzione:\n");
    fprintf(file, "------------------------------------\n");
    fprintf(file, "DFT  : %f secondi\n", dftTime);
    fprintf(file, "Filtro: %f secondi\n", filterTime);
    fprintf(file, "IDFT : %f secondi\n", idftTime);
    fprintf(file, "Totale: %f secondi\n", totalTime);
    fprintf(file, "------------------------------------\n");
    fclose(file);
}

//main
int main(int argc, char *argv[]){
    double *x, *y;
    Complesso *X;
    int N;
    clock_t start, stop;
    double dftTime, filterTime, idftTime, totalTime;
    char *filename;

    if(argc != 2){
        printf("Utilizzo: %s <file audio.wav>\n", argv[0]);
        exit(1);
    }

    filename = argv[1];

    // Creazione delle directory ./output e ./reports se non esistono
    mkdir("./output", 0777);
    mkdir("./reports", 0777);

    N = getWavFileLength(filename);

    x = (double *)malloc(N * sizeof(double));
    y = (double *)malloc(N * sizeof(double));
    X = (Complesso *)malloc(N * sizeof(Complesso));

    char timestamp[20];
    createTimestamp(timestamp, sizeof(timestamp));

    char outputFile[256], reportFile[256];

    snprintf(outputFile, sizeof(outputFile), "./output/output_Parallelo_v1.0C_%s.wav", timestamp);
    snprintf(reportFile, sizeof(reportFile), "./reports/report_Parallelo_v1.0C_%s.txt", timestamp);

    readWavFile(filename, x, N);

    double *d_x, *d_y;
    Complesso *d_X;
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));
    cudaMalloc(&d_X, N * sizeof(Complesso));

    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    start = clock();
    dftKernel<<<gridSize, blockSize>>>(d_x, d_X, N);
    cudaDeviceSynchronize();
    stop = clock();
    dftTime = (double)(stop - start) / CLOCKS_PER_SEC;

    start = clock();
    cudaMemcpy(X, d_X, N * sizeof(Complesso), cudaMemcpyDeviceToHost);
    filtro(X, N, 1000, 44100);
    cudaMemcpy(d_X, X, N * sizeof(Complesso), cudaMemcpyHostToDevice);
    stop = clock();
    filterTime = (double)(stop - start) / CLOCKS_PER_SEC;

    start = clock();
    idftKernel<<<gridSize, blockSize>>>(d_X, d_y, N);
    cudaDeviceSynchronize();
    stop = clock();
    idftTime = (double)(stop - start) / CLOCKS_PER_SEC;

    cudaMemcpy(y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);

    writeWavFile(outputFile, y, N);

    totalTime = dftTime + filterTime + idftTime;

    writeReport(reportFile, dftTime, filterTime, idftTime, totalTime);

    free(x);
    free(y);
    free(X);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_X);

    return 0;
}