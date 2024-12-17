#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define WIDTH 200      // Larghezza griglia
#define HEIGHT 200     // Altezza griglia
#define SPECIES 3      // Numero di specie (1, 2, 3)
#define STATES 3       // Stati della cella (1, 2, 3)
#define ITERATIONS 200 // Iterazioni totali
#define DENSITY 0.2    // Percentuale di celle vive iniziali
#define RANDOM_INTERVAL 10 // Intervallo per generare gruppi casuali

// Definizione della cella
typedef struct {
    int species;  // Specie della cella (0 = vuota)
    int state;    // Stato della cella (0 = inattiva, >0 = stato attivo)
    double energy; // Energia della cella
} Cell;

// Relazioni di predazione tra le specie
// 1→2 (rosso mangia verde), 2→3 (verde mangia blu), 3→1 (blu mangia rosso)
const int predators[SPECIES + 1] = {0, 2, 3, 1};

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

// Funzione per calcolare l'energia pesata dei vicini
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

// Funzione per inizializzare le specie
void initializeSpecies(Cell grid[HEIGHT][WIDTH]) {
    // Inizializzazione della griglia
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            grid[y][x].species = 0;  // Celle vuote
            grid[y][x].state = 0;    // Stato inattivo
            grid[y][x].energy = 0.0; // Energia nulla
        }
    }

    // Distribuzione casuale delle celle vive con densità specificata
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if ((rand() / (double)RAND_MAX) < DENSITY) {
                grid[y][x].species = rand() % SPECIES + 1;
                grid[y][x].state = STATES;
                grid[y][x].energy = rand() / (double)RAND_MAX * 2.0;
            }
        }
    }
}

// Funzione per generare gruppi di specie casuali
void generateRandomGroups(Cell grid[HEIGHT][WIDTH]) {
    int numGroups = rand() % 5 + 1; // Numero di gruppi da generare (da 1 a 5)
    for (int i = 0; i < numGroups; i++) {
        int groupSpecies = rand() % SPECIES + 1; // Scegli una specie per il gruppo
        int groupSize = rand() % 100 + 50; // Numero di celle nel gruppo (da 50 a 150)
        for (int j = 0; j < groupSize; j++) {
            int x = rand() % WIDTH;
            int y = rand() % HEIGHT;
            grid[y][x].species = groupSpecies;
            grid[y][x].state = STATES;
            grid[y][x].energy = rand() / (double)RAND_MAX * 2.0;
        }
    }
}

// Funzione per simulare il lancio di una moneta
int flipCoin() {
    return rand() % 2; // 0 = croce, 1 = testa
}

// Funzione che gestisce la predazione
void handlePredation(Cell grid[HEIGHT][WIDTH], int x, int y, int predator, int prey) {
    int radius = (grid[y][x].energy > 1.5) ? 2 : 1; // Raggio di predazione

    // Cerca predatori e prede nei vicini
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            if (dx == 0 && dy == 0) continue; // Ignora la cella corrente

            int nx = (x + dx + WIDTH) % WIDTH;  // Toroidale
            int ny = (y + dy + HEIGHT) % HEIGHT;

            // Se una preda viene trovata
            if (grid[ny][nx].species == prey) {
                // Simula il lancio della moneta
                if (flipCoin() == 1) {
                    // Testa: la preda si trasforma in predatore
                    grid[ny][nx].species = predator;
                    grid[ny][nx].state = 3; // Stato attivo
                    grid[ny][nx].energy = rand() / (double)RAND_MAX * 2.0; // Energia casuale
                } else {
                    // Croce: la preda muore
                    grid[ny][nx].species = 0;
                    grid[ny][nx].state = 0;
                    grid[ny][nx].energy = 0.0;
                }
            }
        }
    }
}

// Funzione per generare la prossima generazione
void nextGeneration(Cell grid[HEIGHT][WIDTH], Cell newGrid[HEIGHT][WIDTH]) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            Cell currentCell = grid[y][x];
            Cell newCell = {0, 0, 0.0}; // Cella vuota

            if (currentCell.species > 0) {
                // Predazione
                int predator = predators[currentCell.species];
                int prey = currentCell.species;
                handlePredation(grid, x, y, predator, prey);

                // Sopravvivenza e riproduzione
                double neighborEnergy = weightedNeighborEnergy(grid, x, y, currentCell.species);
                newCell.energy = currentCell.energy * 0.9 + 0.1 * neighborEnergy;

                if (newCell.energy > 1.0) {
                    newCell.species = currentCell.species;
                    newCell.state = currentCell.state - 1;
                    if (newCell.state < 1) {
                        newCell.species = 0; // Muore per esaurimento stato
                    }
                } else {
                    newCell.species = 0; // Muore per energia insufficiente
                }
            } else {
                // Riproduzione competitiva
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

// Funzione per calcolare le statistiche della griglia
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

// Funzione per salvare lo stato della griglia in un file
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
    printf("Stato salvato in %s\n", filename);
}

// Funzione principale
int main() {
    srand(time(NULL));

    // Creazione griglie
    Cell grid[HEIGHT][WIDTH];
    Cell newGrid[HEIGHT][WIDTH];

    // Inizializzazione della griglia
    initializeSpecies(grid);

    // Simulazione
    for (int iter = 0; iter < ITERATIONS; iter++) {
        if (iter % RANDOM_INTERVAL == 0) {
            generateRandomGroups(grid); // Genera gruppi casuali ogni X iterazioni
        }

        nextGeneration(grid, newGrid);
        calculateStats(newGrid);
        saveGridToFile(newGrid, iter);

        // Copia nuova griglia nella griglia principale per la prossima iterazione
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                grid[y][x] = newGrid[y][x];
            }
        }
    }

    return 0;
}
