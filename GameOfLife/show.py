import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont

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
def grid_to_colors(grid, cell_size):
    height, width = grid.shape[:2]
    color_grid = np.zeros((height * cell_size, width * cell_size, 3))  # Matrice RGB

    for y in range(height):
        for x in range(width):
            species, state = grid[y, x]
            base_color = COLORS[species]
            if species == 0 or state == 0:  # Celle vuote o morte
                color_grid[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size] = (1.0, 1.0, 1.0)  # Bianco
            else:
                # Regola la saturazione in base allo stato
                saturation = state / 3.0  # Stato massimo = 3
                color_grid[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size] = tuple(c * saturation for c in base_color)
    return color_grid

# Funzione per creare un'immagine da una griglia e aggiungere il numero di iterazione
def grid_to_image(grid, iteration, resolution=2048):
    # Calcola la dimensione di ciascuna cella in pixel
    cell_size = resolution // max(grid.shape)
    color_grid = grid_to_colors(grid, cell_size)

    # Crea l'immagine usando PIL
    img = Image.fromarray((color_grid * 255).astype(np.uint8))

    # Aggiungi il numero di iterazione come testo sull'immagine
    draw = ImageDraw.Draw(img)
    
    # Usare un font di base se non è specificato un altro
    font = ImageFont.load_default()
    text = f"Generation: {iteration}"
    
    # Posizione del testo (in alto a sinistra)
    text_position = (10, 10)
    text_color = (0, 0, 0)  # Colore del testo (bianco)
    
    # Disegna il testo sull'immagine
    draw.text(text_position, text, font=font, fill=text_color)

    return img

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

        # Aggiungi immagine alla lista per la GIF
        img = grid_to_image(grid, iteration)
        images.append(img)

    # Genera la GIF animata
    generate_gif(images, output_gif, duration=200)  # 200 ms per frame

if __name__ == "__main__":
    main()
