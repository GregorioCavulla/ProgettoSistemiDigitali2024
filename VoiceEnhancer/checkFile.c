#include <stdio.h>
#include <stdlib.h>

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

// Funzione per stampare l'header WAV
void printWAVHeader(WAVHeader *header) {
    printf("Chunk ID: %.4s\n", header->chunkID);
    printf("Chunk Size: %d\n", header->chunkSize);
    printf("Format: %.4s\n", header->format);
    printf("Subchunk1 ID: %.4s\n", header->subchunk1ID);
    printf("Subchunk1 Size: %d\n", header->subchunk1Size);
    printf("Audio Format: %d\n", header->audioFormat);
    printf("Number of Channels: %d\n", header->numChannels);
    printf("Sample Rate: %d\n", header->sampleRate);
    printf("Byte Rate: %d\n", header->byteRate);
    printf("Block Align: %d\n", header->blockAlign);
    printf("Bits per Sample: %d\n", header->bitsPerSample);
    printf("Subchunk2 ID: %.4s\n", header->subchunk2ID);
    printf("Subchunk2 Size: %d\n", header->subchunk2Size);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s input.wav\n", argv[0]);
        return 1;
    }

    // Apri file WAV
    FILE *inputFile = fopen(argv[1], "rb");
    if (!inputFile) {
        perror("Errore nell'apertura del file WAV");
        return 1;
    }

    // Leggi header
    WAVHeader header;
    fread(&header, sizeof(WAVHeader), 1, inputFile);

    // Stampa informazioni sull'header
    printWAVHeader(&header);

    // Chiudi il file
    fclose(inputFile);

    return 0;
}
