#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 150     // Larghezza griglia
#define HEIGHT 150     // Altezza griglia
#define SPECIES 3        // Numero di specie (1, 2, 3)
#define STATES 3         // Stati della cella (1, 2, 3)
#define ITERATIONS 150   // Iterazioni totali
#define DENSITY 0.5     // Percentuale di celle vive iniziali

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

// Funzione per inizializzare i glider in diverse posizioni
// Funzione per inizializzare i glider in diverse posizioni in funzione della dimensione della griglia
void initializeGliders(Cell grid[HEIGHT][WIDTH]) {
    // Pattern Glider per specie 1
    int x1 = WIDTH / 5;  // Posizione del glider 1
    int y1 = HEIGHT / 5;
    grid[y1][x1].species = 1; grid[y1][x1].state = 3;
    grid[y1 + 1][x1].species = 1; grid[y1 + 1][x1].state = 3;
    grid[y1 + 2][x1].species = 1; grid[y1 + 2][x1].state = 3;
    grid[y1 + 1][x1 + 1].species = 1; grid[y1 + 1][x1 + 1].state = 3;
    grid[y1][x1 + 2].species = 1; grid[y1][x1 + 2].state = 3;

    // Pattern Glider per specie 2 (posizione angolo diverso)
    int x2 = WIDTH / 2;  // Posizione del glider 2
    int y2 = HEIGHT / 4;
    grid[y2][x2].species = 2; grid[y2][x2].state = 3;
    grid[y2 + 1][x2].species = 2; grid[y2 + 1][x2].state = 3;
    grid[y2 + 2][x2].species = 2; grid[y2 + 2][x2].state = 3;
    grid[y2 + 1][x2 + 1].species = 2; grid[y2 + 1][x2 + 1].state = 3;
    grid[y2][x2 + 2].species = 2; grid[y2][x2 + 2].state = 3;

    // Pattern Glider per specie 3 (posizione angolo diverso)
    int x3 = WIDTH - WIDTH / 5;  // Posizione del glider 3
    int y3 = HEIGHT - HEIGHT / 5;
    grid[y3][x3].species = 3; grid[y3][x3].state = 3;
    grid[y3 + 1][x3].species = 3; grid[y3 + 1][x3].state = 3;
    grid[y3 + 2][x3].species = 3; grid[y3 + 2][x3].state = 3;
    grid[y3 + 1][x3 + 1].species = 3; grid[y3 + 1][x3 + 1].state = 3;
    grid[y3][x3 + 2].species = 3; grid[y3][x3 + 2].state = 3;
}


// Inizializza la griglia con specie casuali e stati iniziali
void initializeGrid(Cell grid[HEIGHT][WIDTH]) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (rand() / (double)RAND_MAX < DENSITY) {
                grid[y][x].species = rand() % SPECIES + 1;  // Una specie casuale tra 1, 2, 3
                grid[y][x].state = rand() % STATES + 1;     // Stato casuale tra 1 e 3
            } else {
                grid[y][x].species = 0;  // Cella vuota
                grid[y][x].state = 0;
            }
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

    // Inizializza la griglia con più celle vive
    initializeGrid(grid);

    // Aggiungi i glider
    initializeGliders(grid);

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
