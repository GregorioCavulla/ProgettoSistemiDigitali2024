#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define PI 3.14159265358979323846

// Funzione per leggere l'intestazione di un file .wav
void readWavFile(const char *filename, double *x, int N) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Errore nell'apertura del file %s\n", filename);
        exit(1);
    }

    // Leggi l'intestazione del file .wav (44 byte standard)
    uint8_t header[44];
    fread(header, sizeof(uint8_t), 44, file);

    // Leggi i campioni audio come int16_t e convertili in double
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

    // Scrivi un'intestazione standard per un file .wav a 16 bit, mono, 44.1 kHz
    uint8_t header[44] = {
        'R', 'I', 'F', 'F',
        0, 0, 0, 0, // Placeholder per la dimensione del file
        'W', 'A', 'V', 'E',
        'f', 'm', 't', ' ',
        16, 0, 0, 0, // Dimensione del blocco fmt
        1, 0, // PCM
        1, 0, // Canali (mono)
        0x44, 0xAC, 0x00, 0x00, // Frequenza di campionamento: 44100 Hz
        0x88, 0x58, 0x01, 0x00, // Byte rate: 44100 * 2
        2, 0, // Block align: 2 byte
        16, 0, // Bit depth: 16 bit
        'd', 'a', 't', 'a',
        0, 0, 0, 0 // Placeholder per la dimensione dei dati
    };

    // Calcola la dimensione totale del file e dei dati
    int dataSize = N * sizeof(int16_t);
    int fileSize = 44 + dataSize - 8;

    // Aggiorna i campi dell'intestazione
    header[4] = (fileSize & 0xFF);
    header[5] = ((fileSize >> 8) & 0xFF);
    header[6] = ((fileSize >> 16) & 0xFF);
    header[7] = ((fileSize >> 24) & 0xFF);

    header[40] = (dataSize & 0xFF);
    header[41] = ((dataSize >> 8) & 0xFF);
    header[42] = ((dataSize >> 16) & 0xFF);
    header[43] = ((dataSize >> 24) & 0xFF);

    // Scrivi l'intestazione
    fwrite(header, sizeof(uint8_t), 44, file);

    // Scrivi i campioni audio convertiti in int16_t
    int16_t *buffer = (int16_t *)malloc(N * sizeof(int16_t));
    for (int i = 0; i < N; i++) {
        buffer[i] = (int16_t)x[i];
    }
    fwrite(buffer, sizeof(int16_t), N, file);

    free(buffer);
    fclose(file);
}

// Funzione che calcola la Discrete Fourier Transform di un segnale audio
void dft(double *x, double *X, int N) {
    for (int k = 0; k < N; k++) {
        X[k] = 0;
        for (int n = 0; n < N; n++) {
            X[k] += x[n] * cos(2 * PI * k * n / N);
        }
    }
}

// Funzione che applica un filtro passa-basso al segnale audio
void filtro(double *X, int N, int fc) {
    for (int k = 0; k < N; k++) {
        if (k > fc && k < N - fc) {
            X[k] = 0;
        }
    }
}

// Funzione che calcola l'Inverse Discrete Fourier Transform del segnale audio filtrato
void idft(double *X, double *x, int N) {
    for (int n = 0; n < N; n++) {
        x[n] = 0;
        for (int k = 0; k < N; k++) {
            x[n] += X[k] * cos(2 * PI * k * n / N);
        }
        x[n] /= N;
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

// Funzione main, prende in input il nome del file audio e restituisce il file audio filtrato
int main(int argc, char *argv[] ) {
    int N = 44100;
    double *x = (double *)malloc(N * sizeof(double)); // Allocazione dinamica di memoria per il segnale audio
    double *X = (double *)malloc(N * sizeof(double)); // Allocazione dinamica di memoria per la DFT del segnale audio
    double *y = (double *)malloc(N * sizeof(double)); // Allocazione dinamica di memoria per il segnale audio filtrato

    clock_t start, end;
    double dftTime, filterTime, idftTime;
    char *filename;

    // Verifica che il numero di argomenti sia corretto
    if (argc != 2) {
        printf("Utilizzo: %s <file_audio.wav>\n", argv[0]);
        exit(1);
    }

    filename = argv[1];

    // Leggi il file audio
    readWavFile(filename, x, N);

    // Calcola la DFT
    start = clock();
    dft(x, X, N);
    end = clock();
    dftTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Applica il filtro passa-basso
    start = clock();
    filtro(X, N, 1000);
    end = clock();
    filterTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Calcola la IDFT
    start = clock();
    idft(X, y, N);
    end = clock();
    idftTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Scrivi il file audio filtrato
    writeWavFile("output.wav", y, N);

    // Scrivi il report dei tempi
    writeReport("reportDFT.txt", dftTime, filterTime, idftTime, dftTime + filterTime + idftTime);

    // Libera la memoria
    free(x);
    free(X);
    free(y);

    return 0;
}
