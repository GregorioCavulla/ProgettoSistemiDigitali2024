//Shared memory

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <sys/stat.h>
#include <cuda.h>

#define PI 3.14159265358979323846

// Struttura per rappresentare un numero complesso con allineamento
typedef struct __align__(8) {
    float real;
    float imag;
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

// Kernel per la trasformata discreta di Fourier (DFT)
__global__ void dftKernel(const float *x, Complesso *X, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float angle, cosAngle, sinAngle;
    float angleFactor = 2.0f * PI * i / N;
    float real = 0;
    float imag = 0;
    if (i < N) {
        X[i].real = 0;
        X[i].imag = 0;
        for (int j = 0; j < N; j+=10) {

            angle = angleFactor * j;
            __sincosf(angle, &sinAngle, &cosAngle);
            real = fmaf(x[j], cosAngle, real);
            imag = fmaf(-x[j], sinAngle, imag);

            angle = angleFactor * (j+1);
            __sincosf(angle, &sinAngle, &cosAngle);
            real = fmaf(x[j+1], cosAngle, real);
            imag = fmaf(-x[j+1], sinAngle, imag);

            angle = angleFactor * (j+2);
            __sincosf(angle, &sinAngle, &cosAngle);
            real = fmaf(x[j+2], cosAngle, real);
            imag = fmaf(-x[j+2], sinAngle, imag);

            angle = angleFactor * (j+3);
            __sincosf(angle, &sinAngle, &cosAngle);
            real = fmaf(x[j+3], cosAngle, real);
            imag = fmaf(-x[j+3], sinAngle, imag);

            angle = angleFactor * (j+4);
            __sincosf(angle, &sinAngle, &cosAngle);
            real = fmaf(x[j+4], cosAngle, real);
            imag = fmaf(-x[j+4], sinAngle, imag);

            angle = angleFactor * (j+5);
            __sincosf(angle, &sinAngle, &cosAngle);
            real = fmaf(x[j+5], cosAngle, real);
            imag = fmaf(-x[j+5], sinAngle, imag);

            angle = angleFactor * (j+6);
            __sincosf(angle, &sinAngle, &cosAngle);
            real = fmaf(x[j+6], cosAngle, real);
            imag = fmaf(-x[j+6], sinAngle, imag);

            angle = angleFactor * (j+7);
            __sincosf(angle, &sinAngle, &cosAngle);
            real = fmaf(x[j+7], cosAngle, real);
            imag = fmaf(-x[j+7], sinAngle, imag);

            angle = angleFactor * (j+8);
            __sincosf(angle, &sinAngle, &cosAngle);
            real = fmaf(x[j+8], cosAngle, real);
            imag = fmaf(-x[j+8], sinAngle, imag);

            angle = angleFactor * (j+9);
            __sincosf(angle, &sinAngle, &cosAngle);
            real = fmaf(x[j+9], cosAngle, real);
            imag = fmaf(-x[j+9], sinAngle, imag);
        }
        // Salva il risultato nei valori complessi di output
        X[i].real = real;
        X[i].imag = imag;
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
    float angle;
    if (i < N) {
        x[i] = 0;

        float angleFactor = 2.0f * PI * i / N;
        float cosAngle, sinAngle;

        for (int j = 0; j < N; j+=10) {

            angle = angleFactor * j;
            __sincosf(angle, &sinAngle, &cosAngle);
            x[i] = fmaf(X[j].real, cosAngle, x[i]);
            x[i] = fmaf(-X[j].imag, sinAngle, x[i]);
            
            angle = angleFactor * (j+1);
            __sincosf(angle, &sinAngle, &cosAngle);
            x[i] = fmaf(X[j+1].real, cosAngle, x[i]);
            x[i] = fmaf(-X[j+1].imag, sinAngle, x[i]);
            
            angle = angleFactor * (j+2);
            __sincosf(angle, &sinAngle, &cosAngle);
            x[i] = fmaf(X[j+2].real, cosAngle, x[i]);
            x[i] = fmaf(-X[j+2].imag, sinAngle, x[i]);
            
            angle = angleFactor * (j+3);
            __sincosf(angle, &sinAngle, &cosAngle);
            x[i] = fmaf(X[j+3].real, cosAngle, x[i]);
            x[i] = fmaf(-X[j+3].imag, sinAngle, x[i]);
            
            angle = angleFactor * (j+4);
            __sincosf(angle, &sinAngle, &cosAngle);
            x[i] = fmaf(X[j+4].real, cosAngle, x[i]);
            x[i] = fmaf(-X[j+4].imag, sinAngle, x[i]);
            
            angle = angleFactor * (j+5);
            __sincosf(angle, &sinAngle, &cosAngle);
            x[i] = fmaf(X[j+5].real, cosAngle, x[i]);
            x[i] = fmaf(-X[j+5].imag, sinAngle, x[i]);
            
            angle = angleFactor * (j+6);
            __sincosf(angle, &sinAngle, &cosAngle);
            x[i] = fmaf(X[j+6].real, cosAngle, x[i]);
            x[i] = fmaf(-X[j+6].imag, sinAngle, x[i]);
            
            angle = angleFactor * (j+7);
            __sincosf(angle, &sinAngle, &cosAngle);
            x[i] = fmaf(X[j+7].real, cosAngle, x[i]);
            x[i] = fmaf(-X[j+7].imag, sinAngle, x[i]);
            
            angle = angleFactor * (j+8);
            __sincosf(angle, &sinAngle, &cosAngle);
            x[i] = fmaf(X[j+8].real, cosAngle, x[i]);
            x[i] = fmaf(-X[j+8].imag, sinAngle, x[i]);
            
            angle = angleFactor * (j+9);
            __sincosf(angle, &sinAngle, &cosAngle);
            x[i] = fmaf(X[j+9].real, cosAngle, x[i]);
            x[i] = fmaf(-X[j+9].imag, sinAngle, x[i]);
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

/* ------------- OLD MAIN -------------
//main
int main(int argc, char *argv[]){
    float *x, *y;
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

    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));
    X = (Complesso *)malloc(N * sizeof(Complesso));

    char timestamp[20];
    createTimestamp(timestamp, sizeof(timestamp));

    char outputFile[256], reportFile[256];

    snprintf(outputFile, sizeof(outputFile), "./output/output_Parallelo_v4.0C_%s.wav", timestamp);
    snprintf(reportFile, sizeof(reportFile), "./reports/report_Parallelo_v4.0C_%s.txt", timestamp);

    readWavFile(filename, x, N);

    float *d_x, *d_y;
    Complesso *d_X;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_X, N * sizeof(Complesso));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    start = clock();
    dftKernel<<<gridSize, blockSize>>>(d_x, d_X, N);
    cudaDeviceSynchronize();
    stop = clock();
    dftTime = (double)(stop - start) / CLOCKS_PER_SEC;

    start = clock();
    filtro<<<gridSize, blockSize>>>(d_X, N, 1000, 44100);
    cudaDeviceSynchronize();
    stop = clock();
    filterTime = (double)(stop - start) / CLOCKS_PER_SEC;

    start = clock();
    idftKernel<<<gridSize, blockSize>>>(d_X, d_y, N);
    cudaDeviceSynchronize();
    stop = clock();
    idftTime = (double)(stop - start) / CLOCKS_PER_SEC;

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

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
*/


// nuovo main
int main(int argc, char *argv[]) {
    // Dichiarazione delle variabili principali
    float *x, *y;        // Array per i dati audio in ingresso (x) e in uscita (y)
    Complesso *X;        // Array per memorizzare i risultati della DFT (numeri complessi)
    int N;               // Numero di campioni audio
    clock_t start, stop; // Variabili per la misurazione dei tempi
    double dftTime, filterTime, idftTime, totalTime; // Tempi di esecuzione per ciascuna fase
    char *filename;      // Nome del file audio in ingresso

    // Controllo che il programma sia avviato con un parametro (il file audio)
    if (argc != 2) {
        printf("Utilizzo: %s <file audio.wav>\n", argv[0]);
        exit(1); // Esce con errore se non c'è il parametro richiesto
    }

    filename = argv[1]; // Assegna il nome del file passato come parametro

    // Creazione delle directory ./output e ./reports se non esistono già
    mkdir("./output", 0777);
    mkdir("./reports", 0777);

    // Determina la lunghezza del file .wav in campioni
    N = getWavFileLength(filename);

    // Alloca memoria per i dati audio in ingresso, in uscita e per la trasformata
    x = (float *)malloc(N * sizeof(float));   // Array per i campioni originali
    y = (float *)malloc(N * sizeof(float));   // Array per i campioni processati
    X = (Complesso *)malloc(N * sizeof(Complesso)); // Array complesso per la DFT

    // Genera un timestamp per identificare i file di output
    char timestamp[20];
    createTimestamp(timestamp, sizeof(timestamp));

    // Costruisce i nomi dei file di output e del report usando il timestamp
    char outputFile[256], reportFile[256];
    snprintf(outputFile, sizeof(outputFile), "./output/output_Parallelo_v4.0C_%s.wav", timestamp);
    snprintf(reportFile, sizeof(reportFile), "./reports/report_Parallelo_v4.0C_%s.txt", timestamp);

    // Legge i dati dal file .wav di input e li memorizza nell'array x
    readWavFile(filename, x, N);

    // Dichiarazione e allocazione di memoria per i dati sul dispositivo GPU
    float *d_x, *d_y;         // Array per i campioni audio sulla GPU
    Complesso *d_X;           // Array per i risultati della DFT sulla GPU
    cudaMalloc(&d_x, N * sizeof(float)); // Allocazione per i campioni originali
    cudaMalloc(&d_y, N * sizeof(float)); // Allocazione per i campioni filtrati
    cudaMalloc(&d_X, N * sizeof(Complesso)); // Allocazione per i numeri complessi della DFT

    // Copia i dati dal host (CPU) al device (GPU)
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Definizione dei parametri di esecuzione per i kernel CUDA
    int blockSize = 256; // Numero di thread per blocco
    int gridSize = (N + blockSize - 1) / blockSize; // Numero di blocchi
    size_t sharedMemorySize = blockSize * sizeof(float); // Allocazione per thread DFT e IDFT
    size_t sharedMemorySizeFilter = blockSize * sizeof(Complesso); // Allocazione per thread filtro

    // Esegue la DFT sui dati usando la GPU
    start = clock(); // Inizia la misurazione del tempo
    dftKernel<<<gridSize, blockSize, sharedMemorySize>>>(d_x, d_X, N);    
    cudaDeviceSynchronize(); // Aspetta che la GPU completi il calcolo
    stop = clock(); // Termina la misurazione del tempo
    dftTime = (double)(stop - start) / CLOCKS_PER_SEC; // Calcola il tempo impiegato

    // Applica un filtro passa-basso ai dati trasformati
    start = clock(); // Inizia la misurazione del tempo
    filtro<<<gridSize, blockSize, sharedMemorySizeFilter>>>(d_X, N, 1000, 44100);
    cudaDeviceSynchronize(); // Aspetta che la GPU completi il calcolo
    stop = clock(); // Termina la misurazione del tempo
    filterTime = (double)(stop - start) / CLOCKS_PER_SEC; // Calcola il tempo impiegato

    // Esegue la IDFT sui dati filtrati per riportarli nel dominio del tempo
    start = clock(); // Inizia la misurazione del tempo
    idftKernel<<<gridSize, blockSize, sharedMemorySize>>>(d_X, d_y, N);
    cudaDeviceSynchronize(); // Aspetta che la GPU completi il calcolo
    stop = clock(); // Termina la misurazione del tempo
    idftTime = (double)(stop - start) / CLOCKS_PER_SEC; // Calcola il tempo impiegato

    // Copia i risultati dal device (GPU) all'host (CPU)
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Scrive i dati processati in un nuovo file .wav
    writeWavFile(outputFile, y, N);

    // Calcola il tempo totale del processo
    totalTime = dftTime + filterTime + idftTime;

    // Scrive un report dei tempi di esecuzione
    writeReport(reportFile, dftTime, filterTime, idftTime, totalTime);

    // Libera la memoria allocata su CPU
    free(x);
    free(y);
    free(X);

    // Libera la memoria allocata su GPU
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_X);

    return 0; // Fine del programma
}
