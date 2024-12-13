import os
import numpy as np
import matplotlib.pyplot as plt

def select_data_directory():
    """Chiede all'utente di selezionare la modalità (Sequenziale o CUDA) e restituisce la directory corrispondente."""
    print("Seleziona la modalità:")
    print("1. Sequenziale")
    print("2. CUDA")
    choice = input("Inserisci il numero della modalità: ")

    if choice == "1":
        return "./Sequenziale/data"
    elif choice == "2":
        return "./CUDA/data"
    else:
        print("Scelta non valida. Uscita.")
        exit(1)

def plot_grid_live(file_path):
    """Legge i dati dal file e aggiorna un plot in tempo reale."""
    data = np.loadtxt(file_path)

    # Estrae le dimensioni della griglia
    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    rho = data[:, 2]
    ux = data[:, 3]
    uy = data[:, 4]

    # Converte i dati in matrici 2D
    NX, NY = x.max() + 1, y.max() + 1
    rho_grid = rho.reshape((NX, NY))

    # Plot della densità
    plt.clf()
    plt.imshow(rho_grid.T, origin="lower", cmap="viridis")
    plt.colorbar(label="Densità (rho)")
    plt.title("Densità della griglia (step temporale)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.pause(0.001)

# Seleziona la directory dei dati
data_dir = select_data_directory()

# Verifica che la directory esista
if not os.path.exists(data_dir):
    print(f"La directory {data_dir} non esiste. Uscita.")
    exit(1)

# Trova tutti i file .dat nella directory selezionata
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".dat")])

if not files:
    print(f"Nessun file .dat trovato nella directory {data_dir}. Uscita.")
    exit(1)

# Mostra una sequenza di plot
plt.figure(figsize=(10, 8))
for i, file_name in enumerate(files):
    file_path = os.path.join(data_dir, file_name)
    print(f"Visualizzazione immagine per {file_name}")
    plot_grid_live(file_path)

print("Visualizzazione completata.")
