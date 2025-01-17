#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846

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

// Funzione che legge un file .wav e determina se è mono o stereo
int readWavFile(char *filename, double **xL, double **xR, int *N, int *numChannels) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Errore nell'apertura del file\n");
        return -1;
    }

    // Lettura dell'intestazione WAV
    fseek(file, 22, SEEK_SET);
    fread(numChannels, sizeof(short), 1, file);
    
    fseek(file, 24, SEEK_SET);
    int sampleRate;
    fread(&sampleRate, sizeof(int), 1, file);

    fseek(file, 40, SEEK_SET);
    fread(N, sizeof(int), 1, file);
    *N /= (*numChannels * sizeof(short)); // Numero di campioni per canale

    *xL = (double *)malloc(*N * sizeof(double));
    *xR = *numChannels == 2 ? (double *)malloc(*N * sizeof(double)) : NULL;

    fseek(file, 44, SEEK_SET);
    for (int i = 0; i < *N; i++) {
        short sampleL, sampleR;
        fread(&sampleL, sizeof(short), 1, file);
        (*xL)[i] = sampleL / 32768.0; // Normalizzazione

        if (*numChannels == 2) {
            fread(&sampleR, sizeof(short), 1, file);
            (*xR)[i] = sampleR / 32768.0; // Normalizzazione
        }
    }

    fclose(file);
    return sampleRate;
}

// Funzione che scrive un file .wav mono o stereo
void writeWavFile(char *filename, double *xL, double *xR, int N, int numChannels, int sampleRate) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Errore nell'apertura del file\n");
        exit(1);
    }

    short bitsPerSample = 16;
    short blockAlign = numChannels * bitsPerSample / 8;
    int byteRate = sampleRate * blockAlign;
    int subchunk2Size = N * blockAlign;
    int chunkSize = 36 + subchunk2Size;

    // Intestazione WAV
    fwrite("RIFF", 1, 4, file);
    fwrite(&chunkSize, sizeof(int), 1, file);
    fwrite("WAVE", 1, 4, file);
    fwrite("fmt ", 1, 4, file);

    int subchunk1Size = 16;
    short audioFormat = 1;
    fwrite(&subchunk1Size, sizeof(int), 1, file);
    fwrite(&audioFormat, sizeof(short), 1, file);
    fwrite(&numChannels, sizeof(short), 1, file);
    fwrite(&sampleRate, sizeof(int), 1, file);
    fwrite(&byteRate, sizeof(int), 1, file);
    fwrite(&blockAlign, sizeof(short), 1, file);
    fwrite(&bitsPerSample, sizeof(short), 1, file);

    fwrite("data", 1, 4, file);
    fwrite(&subchunk2Size, sizeof(int), 1, file);

    // Scrittura dei campioni
    for (int i = 0; i < N; i++) {
        short sampleL = xL[i] * 32767;
        fwrite(&sampleL, sizeof(short), 1, file);

        if (numChannels == 2) {
            short sampleR = xR[i] * 32767;
            fwrite(&sampleR, sizeof(short), 1, file);
        }
    }

    fclose(file);
}

// Funzione main
int main() {
    int N;
    int numChannels;
    double *xL, *xR = NULL;
    double *XL, *XR = NULL;
    double *yL, *yR = NULL;

    clock_t start, end;

    int sampleRate = readWavFile("./test_wav/short_stereo.wav", &xL, &xR, &N, &numChannels);

    XL = (double *)malloc(N * sizeof(double));
    yL = (double *)malloc(N * sizeof(double));
    if (numChannels == 2) {
        XR = (double *)malloc(N * sizeof(double));
        yR = (double *)malloc(N * sizeof(double));
    }

    // Elaborazione canale sinistro
    start = clock();
    dft(xL, XL, N);
    end = clock();
    printf("Tempo di esecuzione per la DFT (canale sinistro): %f secondi\n", (double)(end - start) / CLOCKS_PER_SEC);

    filtro(XL, N, 1000);

    start = clock();
    idft(XL, yL, N);
    end = clock();
    printf("Tempo di esecuzione per la IDFT (canale sinistro): %f secondi\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Elaborazione canale destro (se stereo)
    if (numChannels == 2) {
        start = clock();
        dft(xR, XR, N);
        end = clock();
        printf("Tempo di esecuzione per la DFT (canale destro): %f secondi\n", (double)(end - start) / CLOCKS_PER_SEC);

        filtro(XR, N, 1000);

        start = clock();
        idft(XR, yR, N);
        end = clock();
        printf("Tempo di esecuzione per la IDFT (canale destro): %f secondi\n", (double)(end - start) / CLOCKS_PER_SEC);
    }

    writeWavFile("output.wav", yL, yR, N, numChannels, sampleRate);

    free(xL);
    free(XL);
    free(yL);
    if (numChannels == 2) {
        free(xR);
        free(XR);
        free(yR);
    }

    return 0;
}
