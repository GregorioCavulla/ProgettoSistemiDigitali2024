#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 1000       // Larghezza griglia
#define HEIGHT 1000      // Altezza griglia
#define SPECIES 3        // Numero di specie (1, 2, 3)
#define STATES 3         // Stati della cella (1, 2, 3)
#define ITERATIONS 10    // Iterazioni totali

typedef struct {
    int species;  // Specie della cella (0 = vuota)
    int state;    // Stato della cella (0 = inattiva, >0 = stato attivo)
} Cell;

// Funzione per contare i vicini di una specie
int countNeighbors(Cell grid[HEIGHT][WIDTH], int x, int y, int species) {
    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue; // Ignora la cella corrente

            int nx = (x + dx + WIDTH) % WIDTH; // Toroidale
            int ny = (y + dy + HEIGHT) % HEIGHT;

            if (grid[ny][nx].species == species) {
                count++;
            }
        }
    }
    return count;
}

// Inizializza la griglia con specie casuali e stati iniziali
void initializeGrid(Cell grid[HEIGHT][WIDTH]) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            grid[y][x].species = rand() % (SPECIES + 1);  // Specie casuale (0 = vuota)
            grid[y][x].state = (grid[y][x].species == 0) ? 0 : (rand() % STATES + 1); // Stato casuale
        }
    }
}

// Salva lo stato della griglia in un file
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
                fprintf(file, "0:0 "); // Cella vuota
            } else {
                fprintf(file, "%d:%d ", grid[y][x].species, grid[y][x].state);
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Salvato lo stato della griglia in %s\n", filename);
}

// Aggiorna la griglia per la prossima generazione
void nextGeneration(Cell grid[HEIGHT][WIDTH], Cell newGrid[HEIGHT][WIDTH]) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            Cell currentCell = grid[y][x];
            Cell newCell = {0, 0}; // Di default vuota

            int bestSpecies = 0;
            int bestNeighbors = 0;

            for (int species = 1; species <= SPECIES; species++) {
                int neighbors = countNeighbors(grid, x, y, species);

                if (currentCell.species == species) {
                    // Regole di sopravvivenza
                    if (neighbors == 2 || neighbors == 3) {
                        newCell.species = species;
                        newCell.state = currentCell.state - 1; // Consuma stato
                        if (newCell.state < 1) newCell.species = 0; // Muore se stato = 0
                    }
                } else if (currentCell.species == 0) {
                    // Riproduzione
                    if (neighbors == 3 && neighbors > bestNeighbors) {
                        bestSpecies = species;
                        bestNeighbors = neighbors;
                    }
                }
            }

            // Assegna nuova specie se la cella nasce
            if (currentCell.species == 0 && bestSpecies > 0) {
                newCell.species = bestSpecies;
                newCell.state = 3; // Stato iniziale
            }

            newGrid[y][x] = newCell;
        }
    }
}

// Calcola statistiche della griglia
void calculateStats(Cell grid[HEIGHT][WIDTH]) {
    int speciesCount[SPECIES + 1] = {0};
    int totalState[SPECIES + 1] = {0};

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int species = grid[y][x].species;
            speciesCount[species]++;
            totalState[species] += grid[y][x].state;
        }
    }

    printf("Statistiche della griglia:\n");
    for (int i = 1; i <= SPECIES; i++) {
        printf("  Specie %d: %d celle, Stato medio: %.2f\n", i, speciesCount[i],
               speciesCount[i] ? (double)totalState[i] / speciesCount[i] : 0.0);
    }
    printf("  Celle vuote: %d\n\n", speciesCount[0]);
}

int main() {
    srand(time(NULL));

    Cell (*grid)[WIDTH] = malloc(sizeof(Cell) * HEIGHT * WIDTH);
    Cell (*newGrid)[WIDTH] = malloc(sizeof(Cell) * HEIGHT * WIDTH);

    if (grid == NULL || newGrid == NULL) {
        printf("Errore: memoria insufficiente per allocare la griglia.\n");
        return 1;
    }

    initializeGrid(grid);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        printf("Iterazione %d:\n", iter + 1);
        calculateStats(grid);
        saveGridToFile(grid, iter + 1);

        nextGeneration(grid, newGrid);

        // Scambia grid e newGrid
        Cell (*temp)[WIDTH] = grid;
        grid = newGrid;
        newGrid = temp;
    }

    free(grid);
    free(newGrid);
    return 0;
}
