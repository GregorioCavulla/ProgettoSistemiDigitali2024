import numpy as np
import os

def create_circle_dat(filename, NX, NY, radius):
    """Crea un file .dat con un cerchio al centro della griglia."""
    with open(filename, 'w') as f:
        for x in range(NX):
            for y in range(NY):
                # Determina se il punto (x, y) è all'interno del cerchio
                xc, yc = NX // 2, NY // 2
                value = 1 if (x - xc)**2 + (y - yc)**2 <= radius**2 else 0
                f.write(f"{x} {y} {value}\n")

def create_square_dat(filename, NX, NY, size):
    """Crea un file .dat con un quadrato al centro della griglia."""
    with open(filename, 'w') as f:
        for x in range(NX):
            for y in range(NY):
                # Determina se il punto (x, y) è all'interno del quadrato
                xc, yc = NX // 2, NY // 2
                value = 1 if abs(x - xc) <= size // 2 and abs(y - yc) <= size // 2 else 0
                f.write(f"{x} {y} {value}\n")

def create_wing_dat(filename, NX, NY):
    """Crea un file .dat con un profilo alare al centro della griglia."""
    with open(filename, 'w') as f:
        for x in range(NX):
            for y in range(NY):
                # Profili alari semplici (ad esempio un NACA simmetrico semplificato)
                xc = NX // 2
                chord = NX // 3
                yc = NY // 2
                y_upper = yc + int(0.1 * chord * np.sin(np.pi * (x - xc + chord / 2) / chord))
                y_lower = yc - int(0.1 * chord * np.sin(np.pi * (x - xc + chord / 2) / chord))

                value = 1 if xc - chord // 2 <= x <= xc + chord // 2 and y_lower <= y <= y_upper else 0
                f.write(f"{x} {y} {value}\n")

# Parametri della griglia
NX, NY = 100, 100
output_dir = "./Solids"
os.makedirs(output_dir, exist_ok=True)

# Generazione dei file
create_circle_dat(os.path.join(output_dir, "circle.dat"), NX, NY, radius=20)
create_square_dat(os.path.join(output_dir, "square.dat"), NX, NY, size=30)
create_wing_dat(os.path.join(output_dir, "wing.dat"), NX, NY)

print(f"File generati nella directory {output_dir}.")
