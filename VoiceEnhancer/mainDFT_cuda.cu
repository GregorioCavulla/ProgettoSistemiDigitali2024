#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h> // Per misurare il tempo
#include <cuda.h>

#define TWO_PI 6.28318530718

// Struttura per l'header del file WAV
typedef struct {
    char chunkID[4];
    int chunkSize;
    char format[4];
    char subchunk1ID[4];
    int subchunk1Size;
    short audioFormat;
    short numChannels;
    int sampleRate;
    int byteRate;
    short blockAlign;
    short bitsPerSample;
    char subchunk2ID[4];
    int subchunk2Size;
} WAVHeader;

// Funzione per leggere l'header WAV
void readWAVHeader(FILE *file, WAVHeader *header) {
    fread(header, sizeof(WAVHeader), 1, file);
    // Verifica la presenza di blocchi aggiuntivi come LIST
    while (strncmp(header->subchunk2ID, "data", 4) != 0) {
        // Salta il blocco corrente
        fseek(file, header->subchunk2Size, SEEK_CUR);
        // Rileggi il successivo subchunk
        fread(&header->subchunk2ID, sizeof(header->subchunk2ID), 1, file);
        fread(&header->subchunk2Size, sizeof(header->subchunk2Size), 1, file);
    }

    printf("Chunk ID: %.4s\n", header->chunkID);
}

// Kernel CUDA per calcolare la DFT
__global__ void computeDFTKernel(double *input, double *real, double *imag, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        real[k] = 0;
        imag[k] = 0;
        for (int n = 0; n < N; n++) {
            double angle = TWO_PI * k * n / N;
            real[k] += input[n] * cos(angle);
            imag[k] -= input[n] * sin(angle);
        }
    }
}

// Kernel CUDA per calcolare la IDFT
__global__ void computeIDFTKernel(double *real, double *imag, double *output, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        output[n] = 0;
        for (int k = 0; k < N; k++) {
            double angle = TWO_PI * k * n / N;
            output[n] += real[k] * cos(angle) - imag[k] * sin(angle);
        }
        output[n] /= N; // Normalizzazione
    }
}

// Funzione host per calcolare la DFT usando CUDA
void computeDFT(double *input, double *real, double *imag, int N) {
    double *d_input, *d_real, *d_imag;

    // Allocazione memoria sulla GPU
    cudaMalloc(&d_input, N * sizeof(double));
    cudaMalloc(&d_real, N * sizeof(double));
    cudaMalloc(&d_imag, N * sizeof(double));

    // Copia dei dati sulla GPU
    cudaMemcpy(d_input, input, N * sizeof(double), cudaMemcpyHostToDevice);

    // Lancio del kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeDFTKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_real, d_imag, N);

    // Copia dei risultati dalla GPU alla CPU
    cudaMemcpy(real, d_real, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(imag, d_imag, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Libera memoria sulla GPU
    cudaFree(d_input);
    cudaFree(d_real);
    cudaFree(d_imag);
}

// Funzione host per calcolare la IDFT usando CUDA
void computeIDFT(double *real, double *imag, double *output, int N) {
    double *d_real, *d_imag, *d_output;

    // Allocazione memoria sulla GPU
    cudaMalloc(&d_real, N * sizeof(double));
    cudaMalloc(&d_imag, N * sizeof(double));
    cudaMalloc(&d_output, N * sizeof(double));

    // Copia dei dati sulla GPU
    cudaMemcpy(d_real, real, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imag, imag, N * sizeof(double), cudaMemcpyHostToDevice);

    // Lancio del kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeIDFTKernel<<<blocksPerGrid, threadsPerBlock>>>(d_real, d_imag, d_output, N);

    // Copia dei risultati dalla GPU alla CPU
    cudaMemcpy(output, d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Libera memoria sulla GPU
    cudaFree(d_real);
    cudaFree(d_imag);
    cudaFree(d_output);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Uso: %s input.wav output.wav report.txt\n", argv[0]);
        return 1;
    }

    // Inizio misurazione del tempo
    printf("Elaborazione in corso...\n");
    clock_t startTotal = clock();

    // Apri file WAV
    FILE *inputFile = fopen(argv[1], "rb");
    if (!inputFile) {
        perror("Errore nell'apertura del file WAV");
        return 1;
    }

    // Leggi header
    WAVHeader header;
    readWAVHeader(inputFile, &header);

    // Controlla che sia un file PCM a 16-bit
    if (header.audioFormat != 1 || header.bitsPerSample != 16 || header.numChannels != 1) {
        printf("Supporto solo per file WAV mono, 16-bit PCM.\n");
        fclose(inputFile);
        return 1;
    }

    int numSamples = header.subchunk2Size / sizeof(short);
    short *audioData = (short *)malloc(numSamples * sizeof(short));
    fread(audioData, sizeof(short), numSamples, inputFile);
    fclose(inputFile);

    // Converto campioni in double
    double *input = (double *)malloc(numSamples * sizeof(double));
    for (int i = 0; i < numSamples; i++) {
        input[i] = (double)audioData[i];
    }

    // Alloco memoria per DFT
    double *real = (double *)malloc(numSamples * sizeof(double));
    double *imag = (double *)malloc(numSamples * sizeof(double));
    double *output = (double *)malloc(numSamples * sizeof(double));

    // Misura tempo DFT
    printf("Calcolo DFT...\n");
    clock_t startDFT = clock();
    computeDFT(input, real, imag, numSamples);
    clock_t endDFT = clock();
    double timeDFT = (double)(endDFT - startDFT) / CLOCKS_PER_SEC;
    printf("Tempo DFT: %.5f secondi\n", timeDFT);

    // Applico filtro: es. enfatizzo voce umana (85 Hz - 300 Hz)
    double lowCut = 85.0 / header.sampleRate;
    double highCut = 300.0 / header.sampleRate;

    for (int k = 0; k < numSamples; k++) {
        double frequency = (double)k / numSamples; // Normalizzazione
        if (frequency < lowCut || frequency > highCut) {
            real[k] = 0;
            imag[k] = 0;
        }
    }

    // Misura tempo IDFT
    printf("Calcolo IDFT...\n");
    clock_t startIDFT = clock();
    computeIDFT(real, imag, output, numSamples);
    clock_t endIDFT = clock();
    double timeIDFT = (double)(endIDFT - startIDFT) / CLOCKS_PER_SEC;
    printf("Tempo IDFT: %.5f secondi\n", timeIDFT);

    // Scrivo il file WAV risultante
    printf("Scrittura file WAV...\n");
    FILE *outputFile = fopen(argv[2], "wb");
    fwrite(&header, sizeof(WAVHeader), 1, outputFile);
    for (int i = 0; i < numSamples; i++) {
        short sample = (short)output[i];
        fwrite(&sample, sizeof(short), 1, outputFile);
    }
    fclose(outputFile);

    // Fine misurazione del tempo totale
    clock_t endTotal = clock();
    double timeTotal = (double)(endTotal - startTotal) / CLOCKS_PER_SEC;
    printf("Tempo totale: %.5f secondi\n", timeTotal);

    // Scrittura del report
    FILE *reportFile = fopen(argv[3], "w");
    if (reportFile) {
        fprintf(reportFile, "Tempo DFT: %.5f secondi\n", timeDFT);
        fprintf(reportFile, "Tempo IDFT: %.5f secondi\n", timeIDFT);
        fprintf(reportFile, "Tempo totale: %.5f secondi\n", timeTotal);
        fclose(reportFile);
    } else {
        perror("Errore nella creazione del file di report");
    }

    // Libero memoria
    free(audioData);
    free(input);
    free(real);
    free(imag);
    free(output);

    printf("File processato e salvato in: %s\n", argv[2]);
    return 0;
}