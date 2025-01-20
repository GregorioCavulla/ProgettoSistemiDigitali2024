#include <stdio.h>
#include <unistd.h>  // Per la funzione sleep()
#include <stdlib.h>  // Per la funzione rand()
#include <time.h>    // Per la funzione time()

#define NUM_VETTORI 20      // Numero di vettori
#define INTENSITÀ 5         // Intensità dei vettori
#define GRIGLIA_DIM 100     // Dimensione della griglia (100x100)
#define DURATA 10           // Durata del movimento in secondi
#define VENTILATORE_INIZIO 50  // Posizione X iniziale del ventilatore
#define VENTILATORE_FINE 60   // Posizione X finale del ventilatore
#define INCREMENTO_V 10   // Incremento della velocità in Y per il ventilatore

// Definizione della struttura per ogni vettore
typedef struct {
    double x;     // Posizione lungo l'asse X
    double y;     // Posizione lungo l'asse Y
    double u;     // Velocità lungo l'asse X
    double v;     // Velocità lungo l'asse Y
    double intensità; // Intensità del vettore
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
    // Inizializzazione dell'array di vettori
    Vettore vettori[NUM_VETTORI];

    // Inizializzazione delle posizioni e velocità dei vettori
    for (int i = 0; i < NUM_VETTORI; i++) {
        // Posizione iniziale lungo l'asse Y (disposti uniformemente)
        vettori[i].y = i * (GRIGLIA_DIM / NUM_VETTORI);
        
        // Posizione iniziale lungo l'asse X (tutti partono da x = 0)
        vettori[i].x = 0;  

        // Velocità iniziale
        vettori[i].u = 10.0;  // Velocità iniziale lungo l'asse X
        vettori[i].v = 0.0;  // Velocità iniziale lungo l'asse Y

        // Intensità fissa per tutti i vettori
        vettori[i].intensità = 1.0;
    }

    // Ciclo che simula il movimento dei vettori per DURATA secondi
    for (int t = 0; t < DURATA; t++) {
        // Creare una griglia di dimensioni GRIGLIA_DIM x GRIGLIA_DIM inizializzata a zero
        double griglia[GRIGLIA_DIM][GRIGLIA_DIM] = {0};

        // Applica l'effetto del ventilatore ai vettori
        applicaVentilatore(vettori, NUM_VETTORI);

        // Apri il file per scrivere i dati della posizione dei vettori al secondo t
        char filename[20];
        sprintf(filename, "./Data/vettori_%d.dat", t);  // Nome del file: vettori_0.dat, vettori_1.dat, ...
        FILE *file = fopen(filename, "w");
        if (file == NULL) {
            perror("Errore nell'aprire il file");
            return 1;
        }

        // Scrivi i dati di tutti i vettori al secondo t
        for (int i = 0; i < NUM_VETTORI; i++) {
            // Ogni vettore si sposta secondo la sua velocità lungo X e Y
            vettori[i].x += vettori[i].u;  // Usa la velocità lungo X
            vettori[i].y += vettori[i].v;  // Usa la velocità lungo Y

            // Assicurati che i vettori restino all'interno dei limiti della griglia (100x100)
            if (vettori[i].x >= GRIGLIA_DIM) {
                vettori[i].x = GRIGLIA_DIM - 1;  // Limita il movimento alla dimensione massima della griglia
            }
            if (vettori[i].y >= GRIGLIA_DIM) {
                vettori[i].y = GRIGLIA_DIM - 1;  // Limita il movimento alla dimensione massima della griglia
            }

            // Posiziona il vettore sulla griglia (segna la sua presenza)
            // Griglia 2D: scriviamo la posizione del vettore sulla griglia
            griglia[(int)vettori[i].x][(int)vettori[i].y] = vettori[i].intensità;

            // Scrivi i dati dei vettori nel file
            // La posizione del vettore è (x_pos[i], y_pos[i]), la direzione è (u, v) e l'intensità è 'intensità'
            fprintf(file, "%f %f %f %f %f\n", vettori[i].x, vettori[i].y, vettori[i].u * vettori[i].intensità, vettori[i].v * vettori[i].intensità, vettori[i].intensità);
        }

        fclose(file);  // Chiudi il file

        // Aspetta un secondo prima di generare il file successivo
        sleep(1);
    }

    printf("I file .dat sono stati generati con successo!\n");
    return 0;
}
