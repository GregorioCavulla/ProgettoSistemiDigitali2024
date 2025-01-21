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

// Kernel per calcolare la DFT (solo parte reale) con FMA
__global__ void dftKernel(const float *x, float *X_real, int N) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < N) {
        float sum_real = 0.0f;
        float angle, cos_val;


        for(int n = 0; n < N; n += 10){
            angle = 2.0f * PI * k * n / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+1) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+1], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+2) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+2], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+3) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+3], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+4) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+4], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+5) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+5], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+6) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+6], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+7) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+7], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+8) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+8], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+9) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+9], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

        }
        X_real[k] = sum_real;
    }
}

// Funzione che applica un filtro passa-basso al segnale audio
__global__ void filtroKernel(float *X_real, int N, int fc) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < N && (k > fc && k < N - fc)) {
        X_real[k] = 0.0;
    }
}

// Kernel per calcolare la IDFT (solo parte reale) con FMA
__global__ void idftKernel(const float *X_real, float *x, int N) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N) {
        float sum = 0.0f;
        float angle, cos_val;

        for (int k = 0; k < N; k += 10) {
            angle = 2.0f * PI * k * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+1) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+1], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+2) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+2], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+3) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+3], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+4) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+4], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+5) * n / N;
            cos_val = cosf(angle);    
            sum = fmaf(X_real[k+5], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+6) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+6], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+7) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+7], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+8) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+8], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+9) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+9], cos_val, sum); // FMA: sum += X_real[k] * cos_val
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
    snprintf(outputFile, sizeof(outputFile), "./output/cuda_v3.4_output_%s.wav", timestamp);
    snprintf(reportFile, sizeof(reportFile), "./reports/cuda_v3.4_report_%s.txt", timestamp);

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

    // Applicazione filtro passa-basso
    start = clock();
    filtroKernel<<<gridSize, blockSize>>>(d_X_real, N, 1000);
    cudaDeviceSynchronize();
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

// Kernel per calcolare la DFT (solo parte reale) con FMA
__global__ void dftKernel(const float *x, float *X_real, int N) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < N) {
        float sum_real = 0.0f;
        float angle, cos_val;


        for(int n = 0; n < N; n += 10){
            angle = 2.0f * PI * k * n / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+1) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+1], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+2) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+2], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+3) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+3], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+4) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+4], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+5) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+5], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+6) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+6], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+7) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+7], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+8) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+8], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

            angle = 2.0f * PI * k * (n+9) / N;
            cos_val = cosf(angle);
            sum_real = fmaf(x[n+9], cos_val, sum_real); // FMA: sum_real += x[n] * cos_val

        }
        X_real[k] = sum_real;
    }
}

// Funzione che applica un filtro passa-basso al segnale audio
__global__ void filtroKernel(float *X_real, int N, int fc) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < N && (k > fc && k < N - fc)) {
        X_real[k] = 0.0;
    }
}

// Kernel per calcolare la IDFT (solo parte reale) con FMA
__global__ void idftKernel(const float *X_real, float *x, int N) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N) {
        float sum = 0.0f;
        float angle, cos_val;

        for (int k = 0; k < N; k += 10) {
            angle = 2.0f * PI * k * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+1) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+1], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+2) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+2], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+3) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+3], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+4) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+4], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+5) * n / N;
            cos_val = cosf(angle);    
            sum = fmaf(X_real[k+5], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+6) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+6], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+7) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+7], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+8) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+8], cos_val, sum); // FMA: sum += X_real[k] * cos_val

            angle = 2.0f * PI * (k+9) * n / N;
            cos_val = cosf(angle);
            sum = fmaf(X_real[k+9], cos_val, sum); // FMA: sum += X_real[k] * cos_val
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
    snprintf(outputFile, sizeof(outputFile), "./output/cuda_v3.4_output_%s.wav", timestamp);
    snprintf(reportFile, sizeof(reportFile), "./reports/cuda_v3.4_report_%s.txt", timestamp);

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

    // Applicazione filtro passa-basso
    start = clock();
    filtroKernel<<<gridSize, blockSize>>>(d_X_real, N, 1000);
    cudaDeviceSynchronize();
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



/*

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sys/stat.h>
#include <cuda.h>

define PI 3.14159265358979323846

// Struttura per rappresentare numeri complessi
typedef struct {
    float real;
    float imag;
} Complex;

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

// Kernel per calcolare la DFT (parte reale e immaginaria) con FMA
__global__ void dftKernel(const float *x, Complex *X, int N) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < N) {
        Complex sum = {0.0f, 0.0f};

        for (int n = 0; n < N; ++n) {
            float angle = 2.0f * PI * k * n / N;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            sum.real = fmaf(x[n], cos_val, sum.real);
            sum.imag = fmaf(-x[n], sin_val, sum.imag);
        }

        X[k] = sum;
    }
}

// Kernel per applicare un filtro passa-basso
__global__ void filtroKernel(Complex *X, int N, int fc) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < N && (k > fc && k < N - fc)) {
        X[k].real = 0.0f;
        X[k].imag = 0.0f;
    }
}

// Kernel per calcolare la IDFT (parte reale) con FMA
__global__ void idftKernel(const Complex *X, float *x, int N) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; ++k) {
            float angle = 2.0f * PI * k * n / N;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            sum = fmaf(X[k].real, cos_val, sum);
            sum = fmaf(X[k].imag, sin_val, sum);
        }

        x[n] = sum / N;
    }
}

// Funzione principale
int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Utilizzo: %s <file_audio.wav>\n", argv[0]);
        exit(1);
    }

    const char *filename = argv[1];

    // Creazione delle directory output e reports
    mkdir("./output", 0777);
    mkdir("./reports", 0777);

    // Determina la lunghezza del file audio
    int N = getWavFileLength(filename);

    // Allocazione memoria
    float *x = (float *)malloc(N * sizeof(float));
    Complex *X = (Complex *)malloc(N * sizeof(Complex));
    float *y = (float *)malloc(N * sizeof(float));

    readWavFile(filename, x, N);

    // Allocazione memoria device
    float *d_x;
    Complex *d_X;
    float *d_y;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_X, N * sizeof(Complex));
    cudaMalloc((void **)&d_y, N * sizeof(float));

    // Copia dati host -> device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configurazione kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Esecuzione DFT
    dftKernel<<<gridSize, blockSize>>>(d_x, d_X, N);
    cudaDeviceSynchronize();

    // Applicazione filtro
    filtroKernel<<<gridSize, blockSize>>>(d_X, N, 1000);
    cudaDeviceSynchronize();

    // Esecuzione IDFT
    idftKernel<<<gridSize, blockSize>>>(d_X, d_y, N);
    cudaDeviceSynchronize();

    // Copia dati device -> host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Libera memoria
    free(x);
    free(X);
    free(y);
    cudaFree(d_x);
    cudaFree(d_X);
    cudaFree(d_y);

    return 0;
}

*/