#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define WIDTH 200     // Larghezza griglia
#define HEIGHT 200     // Altezza griglia
#define SPECIES 3      // Numero di specie (1, 2, 3)
#define STATES 3       // Stati della cella (1, 2, 3)
#define ITERATIONS 200 // Iterazioni totali
#define DENSITY 0.1    // Percentuale di celle vive iniziali

typedef struct {
    int species;  // Specie della cella (0 = vuota)
    int state;    // Stato della cella (0 = inattiva, >0 = stato attivo)
    double energy;  // Energia della cella
} Cell;

const int predators[SPECIES + 1] = {0, 2, 3, 1}; // 1→2 (rosso mangia verde), 2→3 (verde mangia blu), 3→1 (blu mangia rosso)

// Funzione per inizializzare i glider in diverse posizioni
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

double weightedNeighborEnergy(Cell grid[HEIGHT][WIDTH], int x, int y, int species) {
    double energySum = 0.0;
    for (int dx = -2; dx <= 2; dx++) {
        for (int dy = -2; dy <= 2; dy++) {
            if (dx == 0 && dy == 0) continue; // Ignora la cella stessa

            int nx = (x + dx + WIDTH) % WIDTH;   // Toroidale
            int ny = (y + dy + HEIGHT) % HEIGHT;

            double distance = sqrt(dx * dx + dy * dy);
            double weight = 1.0 / (1.0 + distance); // Peso decrescente con la distanza

            if (grid[ny][nx].species == species) {
                energySum += weight * grid[ny][nx].energy; // Energia pesata
            }
        }
    }
    return energySum;
}

// Eventi casuali che alterano la griglia
void randomEvents(Cell grid[HEIGHT][WIDTH]) {
    // Evento di mutazione casuale
    if (rand() % 10 == 0) {
        int x = rand() % WIDTH;
        int y = rand() % HEIGHT;
        if (grid[y][x].species > 0) {
            grid[y][x].species = rand() % SPECIES + 1;
            grid[y][x].energy += 0.5; // Aumenta l'energia per la mutazione
        }
    }

    // Evento di siccità: riduce l'energia in un raggio
    if (rand() % 20 == 0) {
        int x = rand() % WIDTH;
        int y = rand() % HEIGHT;
        for (int dx = -2; dx <= 2; dx++) {
            for (int dy = -2; dy <= 2; dy++) {
                if (dx == 0 && dy == 0) continue;

                int nx = (x + dx + WIDTH) % WIDTH;
                int ny = (y + dy + HEIGHT) % HEIGHT;
                grid[ny][nx].energy *= 0.7; // Riduce l'energia
            }
        }
    }

    // Evento di cambiamento ambientale (esempio di evoluzione)
    if (rand() % 15 == 0) {
        int x = rand() % WIDTH;
        int y = rand() % HEIGHT;
        if (grid[y][x].species > 0) {
            grid[y][x].energy *= 1.2; // Aumenta l'energia della cella casuale
        }
    }
}

// Inizializza la griglia con specie casuali e stati iniziali
void initializeGrid(Cell grid[HEIGHT][WIDTH]) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (rand() / (double)RAND_MAX < DENSITY) {
                grid[y][x].species = rand() % SPECIES + 1;
                grid[y][x].state = rand() % STATES + 1;
                grid[y][x].energy = rand() / (double)RAND_MAX * 2.0; // Energia casuale tra 0 e 2
            } else {
                grid[y][x].species = 0;
                grid[y][x].state = 0;
                grid[y][x].energy = 0.0;
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
                fprintf(file, "0:0:0 "); // Vuota
            } else {
                fprintf(file, "%d:%d:%.2f ", grid[y][x].species, grid[y][x].state, grid[y][x].energy);
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Salvato lo stato della griglia in %s\n", filename);
}

void nextGeneration(Cell grid[HEIGHT][WIDTH], Cell newGrid[HEIGHT][WIDTH]) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            Cell currentCell = grid[y][x];
            Cell newCell = {0, 0, 0.0}; // Inizializza la cella vuota

            if (currentCell.species > 0) {
                // Controllo predazione: verifica se c'è un predatore nei vicini
                int predator = predators[currentCell.species];
                int predatorCount = 0;
                int radius = (currentCell.energy > 1.5) ? 2 : 1; // Determina il raggio della predazione

                // Cerca i predatori nei vicini entro il raggio determinato
                for (int dx = -radius; dx <= radius; dx++) {
                    for (int dy = -radius; dy <= radius; dy++) {
                        if (dx == 0 && dy == 0) continue; // Ignora la cella stessa

                        int nx = (x + dx + WIDTH) % WIDTH;  // Toroidale
                        int ny = (y + dy + HEIGHT) % HEIGHT;

                        if (grid[ny][nx].species == predator) {
                            predatorCount++;
                        }
                    }
                }

                // Se c'è almeno un predatore, la cella viene mangiata
                if (predatorCount > 0) {
                    newCell.species = 0; // La cella è mangiata
                    newCell.state = 0;
                    newCell.energy = 0.0;
                } else {
                    // Calcola l'energia dei vicini della stessa specie
                    double neighborEnergy = weightedNeighborEnergy(grid, x, y, currentCell.species);
                    newCell.energy = currentCell.energy * 0.9 + 0.1 * neighborEnergy;

                    // Sopravvivenza: controlla l'energia e lo stato
                    if (newCell.energy > 1.0) {
                        newCell.species = currentCell.species;
                        newCell.state = currentCell.state - 1;
                        if (newCell.state < 1) {
                            newCell.species = 0; // Muore per esaurimento stato
                        }
                    } else {
                        newCell.species = 0; // Muore per energia insufficiente
                    }
                }
            } else {
                // Riproduzione competitiva: cerca la specie con energia maggiore nei vicini
                int bestSpecies = 0;
                double maxEnergy = 0.0;

                for (int species = 1; species <= SPECIES; species++) {
                    double energy = weightedNeighborEnergy(grid, x, y, species);
                    if (energy > maxEnergy && energy > 1.5) { // Soglia energetica
                        maxEnergy = energy;
                        bestSpecies = species;
                    }
                }

                if (bestSpecies > 0) {
                    newCell.species = bestSpecies;
                    newCell.state = 3;   // Stato iniziale
                    newCell.energy = maxEnergy / 2.0;
                }
            }

            newGrid[y][x] = newCell;
        }
    }
}

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
        printf("  Specie %d: %d celle, Stato medio: %.2f\n", i, speciesCount[i], speciesCount[i] ? (double)totalState[i] / speciesCount[i] : 0.0);
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

        // Evento casuale che cambia la griglia
        randomEvents(grid);

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
