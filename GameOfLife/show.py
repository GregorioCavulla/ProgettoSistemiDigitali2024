import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont

# Colori base per le specie (RGB)
COLORS = {
    0: (1.0, 1.0, 1.0),  # Bianco per celle vuote
    1: (1.0, 0.0, 0.0),  # Rosso per specie 1
    2: (0.0, 1.0, 0.0),  # Verde per specie 2
    3: (0.0, 0.0, 1.0),  # Blu per specie 3
}

# Funzione per leggere la griglia da file
def load_grid(filename):
    """
    Carica la griglia da un file specificato.

    Args:
        filename (str): Percorso del file.

    Returns:
        np.array: Griglia delle celle con specie, stato e intensità.
    """
    grid = []
    with open(filename, "r") as file:
        for line in file:
            row = []
            cells = line.strip().split(" ")
            for cell in cells:
                if cell:  # Evita spazi vuoti
                    try:
                        # Dividi la cella nei tre valori: species, state e intensity
                        species, state, intensity = cell.split(":")
                        row.append((int(species), int(state), float(intensity)))
                    except ValueError as e:
                        print(f"Errore nel parsing della cella '{cell}' nel file {filename}: {e}")
                        row.append((0, 0, 0.0))  # Valori di default in caso di errore
            grid.append(row)
    return np.array(grid, dtype=object)


# Funzione per convertire la griglia in una mappa di colori
def grid_to_colors(grid, cell_size):
    """
    Converte la griglia in una rappresentazione a colori RGB.

    Args:
        grid (np.array): Griglia con specie, stato e intensità.
        cell_size (int): Dimensione della singola cella in pixel.

    Returns:
        np.array: Griglia RGB pronta per la visualizzazione.
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

    # Correggi valori NaN o fuori intervallo
    color_grid = np.nan_to_num(color_grid, nan=1.0)  # Sostituisci NaN con bianco
    color_grid = np.clip(color_grid, 0.0, 1.0)  # Forza i valori nell'intervallo [0, 1]

    return color_grid


# Funzione per generare un'immagine da una griglia
def grid_to_image(grid, iteration, resolution=2048): # Risoluzione di default
    """
    Genera un'immagine PIL da una griglia, con un numero di iterazione.

    Args:
        grid (np.array): Griglia delle celle.
        iteration (int): Numero dell'iterazione corrente.
        resolution (int): Risoluzione totale dell'immagine.

    Returns:
        Image: Immagine PIL generata.
    """
    cell_size = resolution // max(grid.shape)  # Calcola la dimensione delle celle
    color_grid = grid_to_colors(grid, cell_size)

    # Crea l'immagine da array numpy
    img = Image.fromarray((color_grid * 255).astype(np.uint8))

    # Aggiunge testo con il numero di iterazione
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"Generation: {iteration}"
    draw.text((10, 10), text, font=font, fill=(0, 0, 0))  # Testo nero
    return img

# Funzione per generare una GIF da una lista di immagini
def generate_gif(image_list, output_path, duration=700):
    """
    Genera una GIF animata da una lista di immagini.

    Args:
        image_list (list): Lista di immagini PIL.
        output_path (str): Percorso di output per la GIF.
        duration (int): Durata di ciascun frame in millisecondi.
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

# Script principale
def main():
    grid_dir = "./Grids"
    output_gif = "game_of_life.gif"

    # Verifica la presenza della directory
    if not os.path.exists(grid_dir):
        print(f"Errore: la directory '{grid_dir}' non esiste.")
        return

    # Ordina i file di iterazione
    files = sorted(
        [f for f in os.listdir(grid_dir) if f.startswith("grid_iteration_") and f.endswith(".txt")]
    )

    if not files:
        print(f"Nessun file trovato in '{grid_dir}'.")
        return

    print(f"Trovati {len(files)} file: {files}")

    images = []  # Lista per salvare le immagini

    # Carica ogni file e genera un'immagine
    for file in files:
        iteration = int(file.split("_")[-1].split(".")[0])
        file_path = os.path.join(grid_dir, file)
        print(f"Caricamento: {file_path} (Iterazione {iteration})")
        
        grid = load_grid(file_path)
        img = grid_to_image(grid, iteration)
        images.append(img)

    # Genera la GIF
    generate_gif(images, output_gif)

if __name__ == "__main__":
    main()
