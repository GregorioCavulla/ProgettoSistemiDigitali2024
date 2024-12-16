import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Colori per le specie: ogni specie ha un colore base (RGB)
COLORS = {
    0: (1.0, 1.0, 1.0),  # Bianco per celle vuote
    1: (1.0, 0.0, 0.0),  # Rosso per specie 1
    2: (0.0, 1.0, 0.0),  # Verde per specie 2
    3: (0.0, 0.0, 1.0),  # Blu per specie 3
}

# Funzione per leggere la griglia da file
def load_grid(filename):
    grid = []
    with open(filename, "r") as file:
        for line in file:
            row = []
            cells = line.strip().split(" ")
            for cell in cells:
                if cell == "":  # Salta celle vuote (eventuali spazi extra)
                    continue
                species, state = map(int, cell.split(":"))
                row.append((species, state))
            grid.append(row)
    return np.array(grid, dtype=object)

# Funzione per mappare la griglia su colori
def grid_to_colors(grid):
    height, width = grid.shape[:2]
    color_grid = np.zeros((height, width, 3))  # Matrice RGB

    for y in range(height):
        for x in range(width):
            species, state = grid[y, x]
            base_color = COLORS[species]
            if species == 0 or state == 0:  # Celle vuote o morte
                color_grid[y, x] = (1.0, 1.0, 1.0)  # Bianco
            else:
                # Regola la saturazione in base allo stato
                saturation = state / 3.0  # Stato massimo = 3
                color_grid[y, x] = tuple(c * saturation for c in base_color)
    return color_grid

# Funzione per creare un'immagine da una griglia
def grid_to_image(grid):
    color_grid = grid_to_colors(grid)
    img = (color_grid * 255).astype(np.uint8)  # Scala RGB a [0-255]
    return Image.fromarray(img)

# Funzione per visualizzare una griglia
def visualize_grid(grid, iteration):
    color_grid = grid_to_colors(grid)
    plt.figure(figsize=(10, 10))
    plt.imshow(color_grid, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Game of Life - Iteration {iteration}", fontsize=16)
    plt.show()

# Funzione per generare una GIF animata
def generate_gif(image_list, output_path, duration=200):
    print(f"Generazione della GIF: {output_path}")
    image_list[0].save(
        output_path,
        save_all=True,
        append_images=image_list[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF salvata con successo in {output_path}")

# Main script per caricare, visualizzare e creare la GIF
def main():
    # Directory dei file generati
    grid_dir = "./Grids"
    output_gif = "game_of_life.gif"

    if not os.path.exists(grid_dir):
        print(f"La directory {grid_dir} non esiste. Assicurati di aver salvato i file.")
        return

    # Trova i file generati
    files = sorted(
        [f for f in os.listdir(grid_dir) if f.startswith("grid_iteration_") and f.endswith(".txt")]
    )
    if not files:
        print(f"Nessun file di iterazione trovato nella directory {grid_dir}.")
        return

    print(f"Trovati {len(files)} file: {files}")

    images = []  # Lista per salvare le immagini per la GIF

    # Carica ogni file e genera un'immagine
    for file in files:
        iteration = int(file.split("_")[-1].split(".")[0])  # Estrai il numero di iterazione
        file_path = os.path.join(grid_dir, file)
        print(f"Caricamento file: {file_path} (Iterazione {iteration})")
        grid = load_grid(file_path)

        # Visualizza la griglia (facoltativo)
        visualize_grid(grid, iteration)

        # Aggiungi immagine alla lista per la GIF
        img = grid_to_image(grid)
        images.append(img)

    # Genera la GIF animata
    generate_gif(images, output_gif, duration=200)  # 200 ms per frame

if __name__ == "__main__":
    main()
