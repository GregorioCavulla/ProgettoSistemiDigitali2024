#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Dimensioni della griglia
#define N 100  // asse x
#define M 100  // asse y

// Numero massimo di passi per una streamline
#define MAX_STEPS 500

// Passo temporale per il calcolo delle streamline
#define DT 0.1

// Numero di linee da generare (default 20)
#define NUM_LINES 20  // Numero di linee da generare (modifica qui per cambiarlo)

// Griglia di velocità (u, v)
double velocity[N][M][2];

// Griglia di ostacoli
int obstacle[N][M];

// Funzione di inizializzazione della griglia
void initialize(){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            velocity[i][j][0] = 1.0; // Velocità lungo x
            velocity[i][j][1] = 0.0; // Velocità lungo y
            obstacle[i][j] = 0;      // Nessun ostacolo
        }
    }

    // Aggiungi un ostacolo rettangolare al tempo 50 alto 20 largo 20 appoggiato all'asse x
    for (int i = 50; i < 70; i++){
        for (int j = 0; j<20; j++){
            obstacle[i][j] = 1;
        }
    }
}

// Funzione di aggiornamento della griglia
void update_grid() {
    for(int i=1; i<N-1; i++){
        for(int j=1; j<M-1; j++){
            if(obstacle[i][j] == 1){
                velocity[i][j][0] = 0.0;
                velocity[i][j][1] = 0.0;
            } else {
                velocity[i][j][0] = (velocity[i+1][j][0] + velocity[i-1][j][0] + velocity[i][j+1][0] + velocity[i][j-1][0]) / 4.0;
                velocity[i][j][1] = (velocity[i+1][j][1] + velocity[i-1][j][1] + velocity[i][j+1][1] + velocity[i][j-1][1]) / 4.0;
            }
        }
    }
}

// Calcola la streamline
void calculate_streamline(double x, double y, double streamline_x[], double streamline_y[]){
    int step = 0;

    while(step < MAX_STEPS){
        int i = (int)x;
        int j = (int)y;

        if(i < 0 || i >= N || j < 0 || j >= M || obstacle[i][j] == 1) break;

        streamline_x[step] = x;
        streamline_y[step] = y;

        double u = velocity[i][j][0];
        double v = velocity[i][j][1];

        x += u * DT;
        y += v * DT;

        step++;
    }

    // Segnala la fine della streamline
    streamline_x[step] = -1;
    streamline_y[step] = -1;
}

// Salva una streamline in un file
void save_streamline_to_file(const char* filename, double streamline_x[], double streamline_y[]){
    FILE* file = fopen(filename, "w");
    if(!file){
        printf("Errore nell'aprire il file %s\n", filename);
        return;
    }

    int step = 0;
    while(streamline_x[step] != -1 && streamline_y[step] != -1){
        int i = (int)streamline_x[step];
        int j = (int)streamline_y[step];

        double speed = sqrt(velocity[i][j][0] * velocity[i][j][0] + velocity[i][j][1] * velocity[i][j][1]);
        fprintf(file, "%lf %lf %lf\n", streamline_x[step], streamline_y[step], speed);

        step++;
    }

    fclose(file);
}

// Salva l'ostacolo in un file per Gnuplot
void save_obstacle_to_file(const char* filename){
    FILE* file = fopen(filename, "w");
    if(!file){
        printf("Errore nell'aprire il file %s\n", filename);
        return;
    }

    // Scrive le coordinate dell'ostacolo
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            if(obstacle[i][j] == 1){
                fprintf(file, "%d %d\n", i, j);
            }
        }
    }

    fclose(file);
}

// Funzione main
int main(){
    initialize();
    update_grid();

    double streamline_x[MAX_STEPS];
    double streamline_y[MAX_STEPS];

    // Calcola il passo tra le linee
    int step_between_lines = M / NUM_LINES;

    // Genera le streamline
    for(int i = 0; i < NUM_LINES; i++){
        int start_y = i * step_between_lines;  // Posizione di partenza per la y

        for(int j = 0; j < N; j += 5){  // Posizione lungo l'asse x separata di 5 unità
            calculate_streamline(j, start_y, streamline_x, streamline_y);
            char filename[100];
            sprintf(filename, "./StreamData/streamline_%d_%d.dat", j, start_y);
            save_streamline_to_file(filename, streamline_x, streamline_y);
        }
    }

    save_obstacle_to_file("ObjectData/object.dat");

    return 0;
}
