#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N 100        // Dimensioni della griglia
#define M 100
#define MAX_STEPS 500 // Numero massimo di passi per una streamline
#define DT 0.1       // Passo temporale per il calcolo delle streamline

#define NUM_STREAMLINES 5

// Griglia di velocità (u, v)
double velocity[N][M][2]; 
int obstacle[N][M]; // 1 se è un ostacolo, 0 altrimenti

// Inizializza la griglia con un flusso uniforme e un ostacolo
void initialize() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            velocity[i][j][0] = 1.0; // Velocità lungo x
            velocity[i][j][1] = 0.0; // Velocità lungo y
            obstacle[i][j] = 0;      // Nessun ostacolo
        }
    }

    // Aggiungi un ostacolo rettangolare
    for (int i = 40; i < 60; i++) {
        for (int j = 40; j < 60; j++) {
            obstacle[i][j] = 1;
        }
    }
}

// Aggiorna la griglia considerando l'ostacolo
void update_grid() {
    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < M-1; j++) {
            if (obstacle[i][j] == 1) {
                velocity[i][j][0] = 0.0; // L'ostacolo blocca il flusso
                velocity[i][j][1] = 0.0;
            } else {
                // Regola semplice: media delle celle vicine
                velocity[i][j][0] = (velocity[i+1][j][0] + velocity[i-1][j][0] +
                                     velocity[i][j+1][0] + velocity[i][j-1][0]) / 4.0;
                velocity[i][j][1] = (velocity[i+1][j][1] + velocity[i-1][j][1] +
                                     velocity[i][j+1][1] + velocity[i][j-1][1]) / 4.0;
            }
        }
    }
}

// Calcola una streamline data una posizione iniziale (x, y)
void calculate_streamline(double x, double y, double streamline_x[], double streamline_y[]) {
    int step = 0;

    while (step < MAX_STEPS) {
        int i = (int)x;
        int j = (int)y;

        if (i < 0 || i >= N || j < 0 || j >= M || obstacle[i][j] == 1) break; // Esci se fuori griglia o su un ostacolo

        // Salva la posizione corrente
        streamline_x[step] = x;
        streamline_y[step] = y;

        // Velocità nella cella corrente
        double u = velocity[i][j][0];
        double v = velocity[i][j][1];

        // Aggiorna la posizione usando il metodo di Euler
        x += u * DT;
        y += v * DT;

        step++;
    }

    // Segnala la fine della streamline
    streamline_x[step] = -1;
    streamline_y[step] = -1;
}

// Salva una streamline in un file (includendo la velocità per il colore)
void save_streamline_to_file(const char* filename, double streamline_x[], double streamline_y[]) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Errore nell'aprire il file %s\n", filename);
        return;
    }

    int step = 0;
    while (streamline_x[step] != -1 && streamline_y[step] != -1) {
        int i = (int)streamline_x[step];
        int j = (int)streamline_y[step];

        double speed = sqrt(velocity[i][j][0] * velocity[i][j][0] + velocity[i][j][1] * velocity[i][j][1]);
        fprintf(file, "%lf %lf %lf\n", streamline_x[step], streamline_y[step], speed);

        step++;
    }

    fclose(file);
}

// Salva l'ostacolo in un file per Gnuplot
void save_obstacle_to_file(const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Errore nell'aprire il file %s\n", filename);
        return;
    }

    // Scrive le coordinate dell'ostacolo
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (obstacle[i][j] == 1) { // Se è un ostacolo
                fprintf(file, "%d %d\n", i, j); // Salva la posizione dell'ostacolo
            }
        }
    }

    fclose(file);
}

int main() {
    // Inizializza la griglia
    initialize();
    update_grid();

    // Array per memorizzare le streamline
    double streamline_x[MAX_STEPS], streamline_y[MAX_STEPS];

    // Calcola e salva le streamline da diversi punti di partenza
    double start_points[NUM_STREAMLINES][2] = {{10, 50}, {20, 50}, {30, 50}, {35, 45}, {35, 55}};

    for (int i = 0; i < NUM_STREAMLINES; i++) {
        calculate_streamline(start_points[i][0], start_points[i][1], streamline_x, streamline_y);

        char filename[50];
        sprintf(filename, "streamline_%d.dat", i);
        save_streamline_to_file(filename, streamline_x, streamline_y);

        printf("Streamline %d salvata in %s\n", i, filename);
    }

    // Salva l'ostacolo in un file
    save_obstacle_to_file("obstacle.dat");

    printf("Streamline calcolate e salvate. Usa Gnuplot per visualizzarle.\n");
    return 0;
}