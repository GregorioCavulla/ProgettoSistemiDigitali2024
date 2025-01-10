import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Colori base per le specie (RGB)
COLORS = {
    0: (1.0, 1.0, 1.0),  # Bianco per celle vuote
    1: (1.0, 0.0, 0.0),  # Rosso per specie 1
    2: (0.0, 1.0, 0.0),  # Verde per specie 2
    3: (0.0, 0.0, 1.0),  # Blu per specie 3
    4: (1.0, 1.0, 0.0),  # Giallo per specie 4
}

# Funzione per leggere la griglia da file
def load_grid(filename):
    """
    Carica la griglia da un file specificato.
    """
    grid = []
    with open(filename, "r") as file:
        for line in file:
            row = []
            cells = line.strip().split(" ")
            for cell in cells:
                if cell:
                    try:
                        species, state, intensity = cell.split(":")
                        row.append((int(species), int(state), float(intensity)))
                    except ValueError:
                        row.append((0, 0, 0.0))  # Valori di default in caso di errore
            grid.append(row)
    return np.array(grid, dtype=object)

# Funzione per convertire la griglia in una mappa di colori
def grid_to_colors(grid, cell_size):
    """
    Converte la griglia in una rappresentazione a colori RGB.
    """
    height, width = grid.shape[:2]
    color_grid = np.zeros((height * cell_size, width * cell_size, 3))  # Matrice RGB

    for y in range(height):
        for x in range(width):
            species, state, intensity = grid[y, x]
            base_color = COLORS.get(species, (1.0, 1.0, 1.0))  # Colore base per la specie
            if species == 0 or state == 0:  # Celle vuote o morte
                color_grid[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size] = (1.0, 1.0, 1.0)  # Bianco
            else:
                # Correggi intensità fuori intervallo
                intensity = max(0.0, min(1.0, intensity))
                # Modula il colore in base all'intensità
                adjusted_color = tuple(c * intensity for c in base_color)
                color_grid[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size] = adjusted_color

    return color_grid

# Funzione per generare un'immagine da una griglia
def grid_to_image(grid, iteration, resolution=2048): # RISOLUZIONE QUI
    """
    Genera un'immagine PIL da una griglia, con un numero di iterazione.
    """
    cell_size = resolution // max(grid.shape)  # Calcola la dimensione delle celle
    color_grid = grid_to_colors(grid, cell_size)

    # Crea l'immagine da array numpy
    img = Image.fromarray((color_grid * 255).astype(np.uint8))

    # Aggiungi testo con il numero di iterazione
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"Generation: {iteration}"
    draw.text((10, 10), text, font=font, fill=(0, 0, 0))  # Testo nero

    return img

# Funzione per generare una GIF da una lista di immagini
def generate_gif(image_list, output_path, duration=200):
    """
    Genera una GIF animata da una lista di immagini.
    """
    print(f"Generazione della GIF in: {output_path}")
    image_list[0].save(
        output_path,
        save_all=True,
        append_images=image_list[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF salvata in: {output_path}")

# Funzione per caricare tutte le griglie in parallelo
def load_grids_parallel(grid_dir):
    files = sorted(
        [f for f in os.listdir(grid_dir) if f.startswith("grid_iteration_") and f.endswith(".txt")]
    )

    if not files:
        print(f"Nessun file trovato in '{grid_dir}'.")
        return []

    print(f"Trovati {len(files)} file: {files}")

    grids = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_grid, os.path.join(grid_dir, file)): file for file in files}
        for future in futures:
            grids.append(future.result())

    return grids

# Funzione per generare immagini in parallelo
def generate_images_parallel(grids):
    images = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(grid_to_image, grid, idx): idx for idx, grid in enumerate(grids)}
        for future in futures:
            images.append(future.result())
    return images

# Script principale
def main():
    grid_dir = "./Grids"
    output_gif = "game_of_life.gif"

    # Carica tutte le griglie in parallelo
    grids = load_grids_parallel(grid_dir)

    if not grids:
        return

    # Genera tutte le immagini in parallelo
    images = generate_images_parallel(grids)

    # Genera la GIF
    generate_gif(images, output_gif)

if __name__ == "__main__":
    main()
