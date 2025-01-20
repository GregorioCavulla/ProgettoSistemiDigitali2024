#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h> // Per misurare il tempo
#include <stdbool.h>

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

// Funzione per calcolare la DFT
void computeDFT(double *input, double *real, double *imag, int N) {
    for (int k = 0; k < N; k++) {
        real[k] = 0;
        imag[k] = 0;
        for (int n = 0; n < N; n++) {
            double angle = TWO_PI * k * n / N;
            real[k] += input[n] * cos(angle);
            imag[k] -= input[n] * sin(angle);
        }
    }
}

// Funzione per applicare un filtro
void applyFilter(double *real, double *imag, int N, double lowCut, double highCut) {
    for (int k = 0; k < N; k++) {
        double frequency = (double)k / N; // Normalizzazione

        if (frequency < lowCut || frequency > highCut) {
            real[k] = 0;
            imag[k] = 0;
        }
    }
}

// Funzione per calcolare la IDFT
void computeIDFT(double *real, double *imag, double *output, int N) {
    for (int n = 0; n < N; n++) {
        output[n] = 0;
        for (int k = 0; k < N; k++) {
            double angle = TWO_PI * k * n / N;
            output[n] += real[k] * cos(angle) - imag[k] * sin(angle);
        }
        output[n] /= N; // Normalizzazione
    }
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
    computeDFT(input, real, imag, numSamples);   //FUNZIONE DA KERNELLIZZARE
    clock_t endDFT = clock();
    double timeDFT = (double)(endDFT - startDFT) / CLOCKS_PER_SEC;
    printf("Tempo DFT: %.5f secondi\n", timeDFT);

    // Applico filtro: es. enfatizzo voce umana (85 Hz - 300 Hz)
    double lowCut = 250.0 / header.sampleRate;
    double highCut = 350.0 / header.sampleRate;

    // Misura tempo filtro
    printf("Applicazione filtro...\n");
    clock_t startFilter = clock();
    applyFilter(real, imag, numSamples, lowCut, highCut);
    clock_t endFilter = clock();
    double timeFilter = (double)(endFilter - startFilter) / CLOCKS_PER_SEC;
    printf("Tempo filtro: %.5f secondi\n", timeFilter);

    // Misura tempo IDFT
    printf("Calcolo IDFT...\n");
    clock_t startIDFT = clock();
    computeIDFT(real, imag, output, numSamples);   //FUNZIONE DA KERNELLIZZARE
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
        fprintf(reportFile, "Tempo filtro: %.5f secondi\n", timeFilter);
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
