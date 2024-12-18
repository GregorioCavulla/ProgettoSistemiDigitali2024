#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define WIDTH 200      // Larghezza griglia
#define HEIGHT 200     // Altezza griglia
#define SPECIES 3      // Numero di specie (1, 2, 3)
#define STATES 3       // Stati della cella (1, 2, 3)
#define ITERATIONS 200 // Iterazioni totali
#define FILENAME_SIZE 50

// Definizione della cella
typedef struct {
    int species;  // Specie della cella (0 = vuota)
    int state;    // Stato della cella (0 = inattiva, >0 = stato attivo)
    double energy; // Energia della cella
} Cell;

// Relazioni di predazione tra le specie
const int predators[SPECIES + 1] = {0, 2, 3, 1}; // 1→2, 2→3, 3→1

// Funzione per contare i vicini di una specie
int countNeighbors(Cell grid[HEIGHT][WIDTH], int x, int y, int species) {
    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;

            int nx = (x + dx + WIDTH) % WIDTH; // Toroidale
            int ny = (y + dy + HEIGHT) % HEIGHT;

            if (grid[ny][nx].species == species) {
                count++;
            }
        }
    }
    return count;
}

// Funzione per calcolare l'energia pesata dei vicini
double weightedNeighborEnergy(Cell grid[HEIGHT][WIDTH], int x, int y, int species) {
    double energySum = 0.0;
    for (int dx = -2; dx <= 2; dx++) {
        for (int dy = -2; dy <= 2; dy++) {
            if (dx == 0 && dy == 0) continue;

            int nx = (x + dx + WIDTH) % WIDTH;   // Toroidale
            int ny = (y + dy + HEIGHT) % HEIGHT;

            double distance = sqrt(dx * dx + dy * dy);
            double weight = 1.0 / (1.0 + distance);

            if (grid[ny][nx].species == species) {
                energySum += weight * grid[ny][nx].energy;
            }
        }
    }
    return energySum;
}

// Funzione per inizializzare la griglia da file
void initializeGrid(Cell grid[HEIGHT][WIDTH], const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Errore: impossibile aprire il file %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int species, state;
            double energy;
            if (fscanf(file, "%d:%d:%lf ", &species, &state, &energy) != 3) {
                printf("Errore nella lettura del file alla posizione (%d, %d).\n", x, y);
                fclose(file);
                exit(EXIT_FAILURE);
            }
            grid[y][x].species = species;
            grid[y][x].state = state;
            grid[y][x].energy = energy;
        }
    }
    fclose(file);
    printf("Griglia inizializzata da %s\n", filename);
}

// Funzione per gestire la predazione
void handlePredation(Cell grid[HEIGHT][WIDTH], int x, int y, int predator, int prey) {
    int radius = (grid[y][x].energy > 1.5) ? 2 : 1;

    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            if (dx == 0 && dy == 0) continue;

            int nx = (x + dx + WIDTH) % WIDTH;
            int ny = (y + dy + HEIGHT) % HEIGHT;

            if (grid[ny][nx].species == prey) {
                grid[ny][nx].species = predator;
                grid[ny][nx].state = 3;
                grid[ny][nx].energy = 1.0;
            }
        }
    }
}

// Funzione per generare la prossima generazione
void nextGeneration(Cell grid[HEIGHT][WIDTH], Cell newGrid[HEIGHT][WIDTH]) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            Cell currentCell = grid[y][x];
            Cell newCell = {0, 0, 0.0};

            if (currentCell.species > 0) {
                int predator = predators[currentCell.species];
                int prey = currentCell.species;
                handlePredation(grid, x, y, predator, prey);

                double neighborEnergy = weightedNeighborEnergy(grid, x, y, currentCell.species);
                newCell.energy = currentCell.energy * 0.9 + 0.1 * neighborEnergy;

                if (newCell.energy > 1.0) {
                    newCell.species = currentCell.species;
                    newCell.state = currentCell.state - 1;
                    if (newCell.state < 1) {
                        newCell.species = 0;
                    }
                } else {
                    newCell.species = 0;
                }
            } else {
                int bestSpecies = 0;
                double maxEnergy = 0.0;

                for (int species = 1; species <= SPECIES; species++) {
                    double energy = weightedNeighborEnergy(grid, x, y, species);
                    if (energy > maxEnergy && energy > 1.5) {
                        maxEnergy = energy;
                        bestSpecies = species;
                    }
                }

                if (bestSpecies > 0) {
                    newCell.species = bestSpecies;
                    newCell.state = 3;
                    newCell.energy = maxEnergy / 2.0;
                }
            }
            newGrid[y][x] = newCell;
        }
    }
}

// Funzione per salvare la griglia su file
void saveGridToFile(Cell grid[HEIGHT][WIDTH], int iteration) {
    char filename[50];
    sprintf(filename, "./Grids/grid_iteration_%05d.txt", iteration);
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        printf("Errore nell'aprire il file %s\n", filename);
        return;
    }

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (grid[y][x].species == 0) {
                fprintf(file, "0:0:0 ");
            } else {
                fprintf(file, "%d:%d:%.2f ", grid[y][x].species, grid[y][x].state, grid[y][x].energy);
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Stato salvato in %s\n", filename);
}

// Funzione principale
int main() {
    srand(time(NULL));

    Cell grid[HEIGHT][WIDTH];
    Cell newGrid[HEIGHT][WIDTH];
    char inputFile[FILENAME_SIZE];

    printf("Inserisci il nome del file di input (es. grid0.txt): ");
    scanf("%49s", inputFile);

    initializeGrid(grid, inputFile);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        nextGeneration(grid, newGrid);
        saveGridToFile(newGrid, iter);

        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                grid[y][x] = newGrid[y][x];
            }
        }
    }

    return 0;
}
