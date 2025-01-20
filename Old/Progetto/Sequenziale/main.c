#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NX 100
#define NY 100
#define Q 9
#define STEPS 1000

typedef struct {
    double f[Q];
    double rho;
    double ux, uy;
    int is_solid;
} Cell;

Cell grid[NX][NY];

void initialize_grid(const char *solid_file, double velocity) {
    FILE *file = fopen(solid_file, "r");
    if (!file) {
        perror("Errore nell'apertura del file");
        exit(EXIT_FAILURE);
    }

    int x, y, value;
    while (fscanf(file, "%d %d %d", &x, &y, &value) == 3) {
        grid[x][y].is_solid = value;
        grid[x][y].rho = 1.0;
        grid[x][y].ux = velocity;
        grid[x][y].uy = 0.0;
        for (int i = 0; i < Q; i++) {
            grid[x][y].f[i] = 1.0 / Q;
        }
    }

    fclose(file);
}

void save_grid_state(int timestep) {
    char filename[256];
    snprintf(filename, sizeof(filename), "./data/grid_step_%05d.dat", timestep);
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Errore nell'apertura del file per il salvataggio dello stato");
        exit(EXIT_FAILURE);
    }

    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            fprintf(file, "%d %d %.6f %.6f %.6f %d\n", x, y, grid[x][y].rho, grid[x][y].ux, grid[x][y].uy, grid[x][y].is_solid);
        }
    }

    fclose(file);
}

void collide_and_stream() {
    double w[Q] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
    int ex[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    int ey[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            if (grid[x][y].is_solid) {
                for (int i = 0; i < Q; i++) {
                    int opp = (i < 5) ? i + 4 : i - 4;
                    grid[x][y].f[i] = grid[x][y].f[opp];
                }
            } else {
                double rho = 0.0;
                double ux = 0.0;
                double uy = 0.0;

                for (int i = 0; i < Q; i++) {
                    rho += grid[x][y].f[i];
                    ux += grid[x][y].f[i] * ex[i];
                    uy += grid[x][y].f[i] * ey[i];
                }

                ux /= rho;
                uy /= rho;
                grid[x][y].rho = rho;
                grid[x][y].ux = ux;
                grid[x][y].uy = uy;

                for (int i = 0; i < Q; i++) {
                    double eu = ex[i] * ux + ey[i] * uy;
                    double uu = ux * ux + uy * uy;
                    double feq = w[i] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uu);
                    grid[x][y].f[i] += -1.0 * (grid[x][y].f[i] - feq);
                }
            }
        }
    }
}

int main() {
    printf("Seleziona il tipo di solido:\n");
    printf("1. Cerchio\n");
    printf("2. Quadrato\n");
    printf("3. Profilo alare\n");
    int choice;
    scanf("%d", &choice);

    printf("Inserischi velocità iniziale\n");
    double velocity;
    scanf("%d", &velocity);

    const char *solid_file;
    switch (choice) {
        case 1:
            solid_file = "../Solids/circle.dat";
            break;
        case 2:
            solid_file = "../Solids/square.dat";
            break;
        case 3:
            solid_file = "../Solids/wing.dat";
            break;
        default:
            printf("Scelta non valida. Uscita.\n");
            return EXIT_FAILURE;
    }

    initialize_grid(solid_file, velocity);

    for (int t = 0; t < STEPS; t++) {
        collide_and_stream();
        save_grid_state(t);
        printf("Step %05d completato\n", t);
    }

    return 0;
}
