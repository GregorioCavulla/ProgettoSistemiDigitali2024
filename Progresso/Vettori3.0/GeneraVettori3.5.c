#include <stdio.h>
#include <unistd.h>  // Per la funzione sleep()
#include <stdlib.h>  // Per la funzione rand()
#include <time.h>    // Per la funzione time()

#define NUM_VETTORI 20      // Numero di vettori
#define INTENSITÀ 5         // Intensità dei vettori
#define GRIGLIA_DIM 100     // Dimensione della griglia (100x100)
#define DURATA 10           // Durata del movimento in secondi

int main() {
    // Array per memorizzare le posizioni iniziali dei vettori lungo l'asse Y
    double y_pos[NUM_VETTORI];
    
    // Inizializzazione delle posizioni lungo l'asse Y (disposti uniformemente)
    for (int i = 0; i < NUM_VETTORI; i++) {
        y_pos[i] = i * (GRIGLIA_DIM / NUM_VETTORI);  // Spazio tra i vettori lungo l'asse Y
    }

    // Posizione iniziale degli oggetti lungo l'asse X (tutti partono da x = 0)
    double x_pos[NUM_VETTORI];
    for (int i = 0; i < NUM_VETTORI; i++) {
        x_pos[i] = 0;  // Posizione iniziale di tutti i vettori
    }

    // Velocità variabile per ciascun vettore lungo gli assi X e Y
    double u[NUM_VETTORI], v[NUM_VETTORI];
    for (int i = 0; i < NUM_VETTORI; i++) {
        // Assegna velocità casuale tra 1 e 6 per u (x) e v (y)
        u[i] = (rand() % 6) + 1;  // Velocità casuale lungo l'asse X
        v[i] = (rand() % 6) + 1;  // Velocità casuale lungo l'asse Y
    }

    // Intensità fissa per tutti i vettori
    double intensità = 1.0;

    // Ciclo che simula il movimento dei vettori per DURATA secondi
    for (int t = 0; t < DURATA; t++) {
        // Creare una griglia di dimensioni GRIGLIA_DIM x GRIGLIA_DIM inizializzata a zero
        double griglia[GRIGLIA_DIM][GRIGLIA_DIM] = {0};

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
            // Ogni vettore si sposta ogni secondo secondo la sua velocità lungo X e Y
            x_pos[i] += u[i];  // Usa la velocità lungo X
            y_pos[i] += v[i];  // Usa la velocità lungo Y

            // Assicurati che i vettori restino all'interno dei limiti della griglia (100x100)
            if (x_pos[i] >= GRIGLIA_DIM) {
                x_pos[i] = GRIGLIA_DIM - 1;  // Limita il movimento alla dimensione massima della griglia
            }
            if (y_pos[i] >= GRIGLIA_DIM) {
                y_pos[i] = GRIGLIA_DIM - 1;  // Limita il movimento alla dimensione massima della griglia
            }

            // Posiziona il vettore sulla griglia (segna la sua presenza)
            // Griglia 2D: scriviamo la posizione del vettore sulla griglia
            griglia[(int)x_pos[i]][(int)y_pos[i]] = intensità;

            // Scrivi i dati dei vettori nel file
            // La posizione del vettore è (x_pos[i], y_pos[i]), la direzione è (u, v) e l'intensità è 'intensità'
            fprintf(file, "%f %f %f %f %f\n", x_pos[i], y_pos[i], u[i] * intensità, v[i] * intensità, intensità);
        }

        fclose(file);  // Chiudi il file

        // Aspetta un secondo prima di generare il file successivo
        sleep(1);
    }

    printf("I file .dat sono stati generati con successo!\n");
    return 0;
}