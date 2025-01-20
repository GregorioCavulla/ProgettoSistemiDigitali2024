import random
import os
import math

# Dimensioni della griglia
WIDTH = 200
HEIGHT = 200
SPECIES = 3
STATE = 3  # Stato iniziale massimo
ENERGY = 1.5  # Energia iniziale predefinita
RADIUS = min(WIDTH, HEIGHT) // 2  # Raggio del cerchio

# Crea una directory "Grids" se non esiste
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

def write_grid_to_file(grid, filename):
    """Scrive una griglia su file."""
    with open(filename, "w") as f:
        for row in grid:
            for cell in row:
                f.write(f"{cell[0]}:{cell[1]}:{cell[2]:.2f} ")
            f.write("\n")

def is_within_circle(x, y, center_x, center_y, radius):
    """Verifica se un punto (x, y) è all'interno di un cerchio dato."""
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

def get_sector(x, y, center_x, center_y):
    """Restituisce il settore in cui si trova il punto (x, y) rispetto al centro del cerchio."""
    angle = math.atan2(y - center_y, x - center_x)  # Calcola l'angolo del punto rispetto al centro
    if angle < 0:
        angle += 2 * math.pi  # Assicurati che l'angolo sia positivo
    if angle < 2 * math.pi / SPECIES:
        return 0  # Primo settore (0°-120°)
    elif angle < 2 * math.pi * 2 / SPECIES:
        return 1  # Secondo settore (120°-240°)
    else:
        return 2  # Terzo settore (240°-360°)

def place_glider(grid, species, x_start, y_start):
    """Posiziona un glider (configurazione Game of Life) in una certa posizione."""
    glider_pattern = [
        (0, 1), (1, 2), (2, 0), (2, 1), (2, 2)  # Coordinate relative di un glider
    ]
    for dx, dy in glider_pattern:
        x, y = x_start + dx, y_start + dy
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            grid[y][x] = (species, random.randint(1, STATE), ENERGY)

def place_species_in_sector(grid, species, center_x, center_y, radius, sector_id):
    """Posiziona una specie all'interno di uno dei 3 settori del cerchio."""
    # Posiziona le cellule nel settore corrispondente
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if is_within_circle(x, y, center_x, center_y, radius):
                sector = get_sector(x, y, center_x, center_y)
                if sector == sector_id:
                    grid[y][x] = (species, random.randint(1, STATE), ENERGY)

def create_grid():
    """Crea una griglia con specie distribuite in 3 settori circolari separati."""
    grid = [[(0, 0, 0.0) for _ in range(WIDTH)] for _ in range(HEIGHT)]  # Griglia vuota
    center_x, center_y = WIDTH // 2, HEIGHT // 2

    # Distribuisci le specie nei 3 settori del cerchio
    for species in range(1, SPECIES + 1):
        place_species_in_sector(grid, species, center_x, center_y, RADIUS, species - 1)

    return grid

# Creazione e salvataggio della griglia
grid = create_grid()
filename = os.path.join(output_dir, "grid_start.txt")
write_grid_to_file(grid, filename)
print(f"Griglia di partenza salvata in '{filename}'")
