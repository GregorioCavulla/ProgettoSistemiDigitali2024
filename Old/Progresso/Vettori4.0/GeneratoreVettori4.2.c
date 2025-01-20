#include <stdio.h>
#include <unistd.h>  // Per la funzione sleep()

#define NUM_VETTORI 20      // Numero di vettori
#define VELOCITÀ 6          // Velocità di spostamento dei vettori (5 unità per secondo)
#define INTENSITÀ 5         // Intensità dei vettori
#define GRIGLIA_DIM 100     // Dimensione della griglia (100x100)
#define DURATA 100          // Durata del movimento in secondi

#define VENTILATORE_INIZIO 49  // Posizione X iniziale del ventilatore
#define VENTILATORE_FINE 51   // Posizione X finale del ventilatore
#define INCREMENTO_V 1   // Incremento della velocità in Y per il ventilatore

typedef struct {
    double x;   // Posizione lungo l'asse X
    double y;   // Posizione lungo l'asse Y
    double u;   // Componente della velocità lungo X
    double v;   // Componente della velocità lungo Y
} Vettore;

void applicaVentilatore(Vettore vettori[], int num_vettori) {
    // Funzione per applicare l'effetto del ventilatore sui vettori nella zona specificata
    for (int i = 0; i < num_vettori; i++) {
        if (vettori[i].x >= VENTILATORE_INIZIO && vettori[i].x <= VENTILATORE_FINE) {
            // Il ventilatore aumenta la velocità in Y di 0.5 ogni secondo
            vettori[i].v += INCREMENTO_V;
        }
    }
}

int main() {
    // Array per memorizzare le posizioni e velocità dei vettori
    Vettore vettori[NUM_VETTORI];

    // Inizializzazione delle posizioni e velocità dei vettori
    for (int i = 0; i < NUM_VETTORI; i++) {
        vettori[i].x = 0;  // Posizione iniziale X di tutti i vettori
        vettori[i].y = i * (GRIGLIA_DIM / NUM_VETTORI);  // Posizione iniziale Y (disposti uniformemente)
        vettori[i].u = 2.0; // Componente della velocità lungo X (inizialmente 1)
        vettori[i].v = 0.0; // Componente della velocità lungo Y (inizialmente 0)
    }

    // Ciclo che simula il movimento dei punti per DURATA secondi
    for (int t = 0; t < DURATA; t++) {
        printf("Generazione del file traiettoria_%d.dat\n", t);
        // Creare una griglia di dimensioni GRIGLIA_DIM x GRIGLIA_DIM inizializzata a zero
        double griglia[GRIGLIA_DIM][GRIGLIA_DIM] = {0};

        // Applica l'effetto del ventilatore ai vettori
        applicaVentilatore(vettori, NUM_VETTORI);

        // Apri il file per scrivere i dati della posizione dei punti al secondo t
        char filename[20];
        sprintf(filename, "./Data/traiettoria_%d.dat", t);  // Nome del file: traiettoria_0.dat, traiettoria_1.dat, ...
        FILE *file = fopen(filename, "w");
        if (file == NULL) {
            perror("Errore nell'aprire il file");
            return 1;
        }

        // Calcolare la posizione di ogni punto
        for (int i = 0; i < NUM_VETTORI; i++) {
            // Ogni vettore influenza la posizione del punto
            vettori[i].x += vettori[i].u;  // Aggiorna la posizione lungo l'asse X
            vettori[i].y += vettori[i].v;  // Aggiorna la posizione lungo l'asse Y

            // Assicurati che i punti restino all'interno dei limiti della griglia
            if (vettori[i].x >= GRIGLIA_DIM) {
                vettori[i].x = GRIGLIA_DIM - 1;  // Limita il movimento alla dimensione massima della griglia
            }
            if (vettori[i].y >= GRIGLIA_DIM) {
                vettori[i].y = GRIGLIA_DIM - 1;  // Limita il movimento alla dimensione massima della griglia
            }

            // Scrivi la posizione dei punti nel file
            fprintf(file, "%f %f\n", vettori[i].x, vettori[i].y);
        }

        fclose(file);  // Chiudi il file

    }

    printf("I file .dat con la traiettoria dei punti sono stati generati con successo!\n");
    return 0;
}
