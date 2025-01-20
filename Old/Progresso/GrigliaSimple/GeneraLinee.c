#include <stdio.h>

#define GRID_SIZE 1000      // Dimensione della griglia (1000x1000)
#define STEP 50             // Distanza tra le righe orizzontali

int main() {
    FILE *file = fopen("griglia.dat", "w");  // Apri il file per scrivere

    if (file == NULL) {
        perror("Errore nell'aprire il file");
        return 1;
    }

    // Genera le righe orizzontali
    for (int y = 0; y < GRID_SIZE; y += STEP) {
        // Scrive una riga orizzontale: due punti per ogni riga
        for (int x = 0; x < GRID_SIZE; x++) {
            fprintf(file, "%d %d\n", x, y);  // Coordinate x, y
        }
        fprintf(file, "\n");  // Aggiunge una riga vuota per separare le righe
    }

    fclose(file);  // Chiudi il file
    printf("File 'griglia.dat' generato con successo!\n");
    return 0;
}
