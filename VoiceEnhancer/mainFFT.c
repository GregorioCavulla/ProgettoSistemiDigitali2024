#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h> // Per misurare il tempo
#include <fftw3.h> // Per FFTW
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

// Funzione per applicare un filtro equalizzatore, che non azzera le frequenze fuori dal range specificato ma le riduce di ampiezza
void applyPeriodicFilter(fftw_complex *fft, int N, double lowCut, double highCut, int sampleRate) {
    for (int k = 0; k < N; k++) {
        double frequency = (double)k * sampleRate / N; // Calcolo della frequenza

        if (frequency > 1800.0) {
            // Elimina tutte le frequenze sopra i 2000 Hz
            fft[k][0] = 0.0;
            fft[k][1] = 0.0;
        } else {
            bool inBand = false;
            for (int i = 0; i * lowCut <= 2000.0; i++) {
                double lowerBound = lowCut + i*100;
                double upperBound = highCut + i*100;

                if (frequency >= lowerBound && frequency <= upperBound) {
                    inBand = true;
                    break;
                }
            }

            if (!inBand) {
                // Elimina frequenze fuori dalla banda
                fft[k][0] = 0.0;
                fft[k][1] = 0.0;
            } else {
                // amplifica le frequenze nella banda
                fft[k][0] *= 1.5;
                fft[k][1] *= 1.5;
            }
        }
    }
}

void applyFilter(fftw_complex *fft, int N, double lowCut, double highCut, int sampleRate) {
    for (int k = 0; k < N; k++) {
        double frequency = (double)k * sampleRate / N; // Calcolo della frequenza

        if (frequency < lowCut || frequency > highCut) {
            // Elimina le frequenze fuori dalla banda
            fft[k][0] = 0.0;
            fft[k][1] = 0.0;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s input.wav output.wav\n", argv[0]);
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

    // Alloco memoria per FFTW
    fftw_complex *fft = fftw_malloc(sizeof(fftw_complex) * numSamples);
    fftw_complex *ifft = fftw_malloc(sizeof(fftw_complex) * numSamples);
    double *output = (double *)malloc(numSamples * sizeof(double));

    fftw_plan forwardPlan = fftw_plan_dft_r2c_1d(numSamples, input, fft, FFTW_ESTIMATE);
    fftw_plan backwardPlan = fftw_plan_dft_c2r_1d(numSamples, fft, output, FFTW_ESTIMATE);

    // Misura tempo FFT
    printf("Calcolo FFT...\n");
    clock_t startFFT = clock();
    fftw_execute(forwardPlan);
    clock_t endFFT = clock();
    printf("Tempo FFT: %.2f secondi\n", (double)(endFFT - startFFT) / CLOCKS_PER_SEC);

    // Applico filtro: mantengo solo frequenze tra 85 Hz - 300 Hz
    double lowCut = 250.0;
    double highCut = 350.0;

    // Misura tempo filtro
    printf("Applicazione filtro...\n");
    clock_t startFilter = clock();
    // applyPeriodicFilter(fft, numSamples, lowCut, highCut, header.sampleRate);
    applyFilter(fft, numSamples, lowCut, highCut, header.sampleRate);
    clock_t endFilter = clock();
    printf("Tempo filtro: %.2f secondi\n", (double)(endFilter - startFilter) / CLOCKS_PER_SEC);

    // Misura tempo IFFT
    printf("Calcolo IFFT...\n");
    clock_t startIFFT = clock();
    fftw_execute(backwardPlan);
    clock_t endIFFT = clock();
    printf("Tempo IFFT: %.2f secondi\n", (double)(endIFFT - startIFFT) / CLOCKS_PER_SEC);

    // Normalizzo l'output
    for (int i = 0; i < numSamples; i++) {
        output[i] /= numSamples;
    }

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
    printf("Tempo totale: %.2f secondi\n", (double)(endTotal - startTotal) / CLOCKS_PER_SEC);

    // Libero memoria
    free(audioData);
    free(input);
    fftw_free(fft);
    fftw_free(ifft);
    free(output);
    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(backwardPlan);

    printf("File processato e salvato in: %s\n", argv[2]);
    return 0;
}
