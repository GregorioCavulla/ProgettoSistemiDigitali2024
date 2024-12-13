import numpy as np
import matplotlib.pyplot as plt

def read_dat_file(file_path):
    # Crea una griglia 100x100 di valori bianchi (0)
    grid = np.zeros((100, 100), dtype=int)
    
    with open(file_path, 'r') as file:
        for line in file:
            x, y, bool_value = line.split()
            x, y, bool_value = int(x), int(y), int(bool_value)
            
            # Imposta il valore corrispondente alla posizione (x, y) nella griglia
            grid[y, x] = bool_value
    
    return grid

def display_grid(grid):
    # Mostra la griglia come immagine
    plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Nasconde gli assi
    plt.show()


# Chiedo a utente quale solido vuole visualizzare
print("Quale solido vuoi visualizzare?")
print("1. Cerchio")
print("2. Quadrato")
print("3. Ala")

choice = input("Inserisci il numero del solido: ")

if choice == "1":
    file_path = './Solids/circle.dat'
elif choice == "2":
    file_path = './Solids/square.dat'
elif choice == "3":
    file_path = './Solids/wing.dat'
else:
    print("Scelta non valida. Uscita.")
    exit(1)

# Legge il file e crea la griglia
grid = read_dat_file(file_path)

# Visualizza la griglia come immagine in bianco e nero
display_grid(grid)
