//filtro host con i float

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sys/stat.h>
#include <cuda.h>

#define PI 3.14159265358979323846

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
void readWavFile(const char *filename, float *x, int N) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Errore nell'apertura del file %s\n", filename);
        exit(1);
    }

    fseek(file, 44, SEEK_SET);
    int16_t *buffer = (int16_t *)malloc(N * sizeof(int16_t));
    fread(buffer, sizeof(int16_t), N, file);
    for (int i = 0; i < N; i++) {
        x[i] = (float)buffer[i];
    }

    free(buffer);
    fclose(file);
}

// Funzione per scrivere un file .wav con l'intestazione
void writeWavFile(const char *filename, float *x, int N) {
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

// Kernel per calcolare la DFT (solo parte reale)
__global__ void dftKernel(const float *x, float *X_real, int N) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < N) {
        float sum_real = 0.0f;
        for (int n = 0; n < N; n++) {
            float angle = 2.0f * PI * k * n / N;
            sum_real += x[n] * cosf(angle);
        }
        X_real[k] = sum_real;
    }
}

// Funzione host per applicare un filtro passa-basso al segnale audio
void filtroHost(float *X_real, int N, int fc) {
    for (int k = 0; k < N; k++) {
        if (k > fc && k < N - fc) {
            X_real[k] = 0.0f;
        }
    }
}

// Kernel per calcolare la IDFT (solo parte reale)
__global__ void idftKernel(const float *X_real, float *x, int N) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            float angle = 2.0f * PI * k * n / N;
            sum += X_real[k] * cosf(angle);
        }
        x[n] = sum / N;
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

// Funzione principale
int main(int argc, char *argv[]) {
    float *x, *X_real, *y;
    int N;
    clock_t start, end;
    double dftTime, filterTime, idftTime;
    char *filename;

    // Verifica che il numero di argomenti sia corretto
    if (argc != 2) {
        printf("Utilizzo: %s <file_audio.wav>\n", argv[0]);
        exit(1);
    }

    filename = argv[1];

    // Creazione delle directory ./output e ./reports se non esistono
    mkdir("./output", 0777);
    mkdir("./reports", 0777);

    // Determina la lunghezza del file audio
    N = getWavFileLength(filename);

    // Allocazione dinamica della memoria
    x = (float *)malloc(N * sizeof(float));
    X_real = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    // Creazione di un timestamp
    char timestamp[20];
    createTimestamp(timestamp, sizeof(timestamp));

    // Percorsi per i file di output
    char outputFile[256], reportFile[256];
    snprintf(outputFile, sizeof(outputFile), "./output/cuda_v1.5_output_%s.wav", timestamp);
    snprintf(reportFile, sizeof(reportFile), "./reports/cuda_v1.5_report_%s.txt", timestamp);

    // Leggi il file audio
    readWavFile(filename, x, N);

    // Allocazione memoria device
    float *d_x, *d_X_real, *d_y;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_X_real, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));

    // Copia dati host -> device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configurazione kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Calcolo DFT
    start = clock();
    dftKernel<<<gridSize, blockSize>>>(d_x, d_X_real, N);
    cudaDeviceSynchronize();
    end = clock();
    dftTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Applicazione filtro passa-basso (host)
    start = clock();
    cudaMemcpy(X_real, d_X_real, N * sizeof(float), cudaMemcpyDeviceToHost);
    filtroHost(X_real, N, 1000);
    cudaMemcpy(d_X_real, X_real, N * sizeof(float), cudaMemcpyHostToDevice);
    end = clock();
    filterTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Calcolo IDFT
    start = clock();
    idftKernel<<<gridSize, blockSize>>>(d_X_real, d_y, N);
    cudaDeviceSynchronize();
    end = clock();
    idftTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Copia risultati device -> host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Scrivi il file audio modificato
    writeWavFile(outputFile, y, N);

    // Scrivi il report dei tempi
    writeReport(reportFile, dftTime, filterTime, idftTime, dftTime + filterTime + idftTime);

    // Libera la memoria
    free(x);
    free(X_real);
    free(y);
    cudaFree(d_x);
    cudaFree(d_X_real);
    cudaFree(d_y);

    return 0;
}
