import os
import subprocess
import re
import time
from pathlib import Path

# Directory e configurazioni
program_dir = "./"  # Directory contenente i programmi eseguibili
input_file = "./input/short_noise.wav"  # File di input da passare ai programmi
report_dir = "./reports"  # Directory dei report generati
dest_dir = "./reports_minimi"  # Directory per i report con tempi minimi
sleep_time = 1  # Tempo in secondi da aspettare tra una esecuzione e l'altra

# Crea la directory dei report minimi se non esiste
os.makedirs(dest_dir, exist_ok=True)

# Fase 1: Esecuzione sequenziale dei programmi
def execute_programs():
    # Trova i file che corrispondono al pattern
    programs = [f for f in os.listdir(program_dir) if re.match(r"DFTparallelo_v\d+\.\d+C\.o", f)]
    
    for program in programs:
        program_path = os.path.join(program_dir, program)
        if os.access(program_path, os.X_OK):  # Controlla se il file è eseguibile
            print(f"Esecuzione del programma: {program}")
            for i in range(5):
                print(f"  Esecuzione {i + 1}/5...")
                try:
                    # Esegue il programma con l'input specificato
                    subprocess.run([f"./{program_path}", input_file], check=True)
                    print(f"  Esecuzione {i + 1} completata.")
                    time.sleep(sleep_time)  # Pausa tra le esecuzioni
                except subprocess.CalledProcessError as e:
                    print(f"Errore durante l'esecuzione di {program}: {e}")
        else:
            print(f"Il file {program} non è eseguibile. Verifica i permessi.")


# Funzione principale
if __name__ == "__main__":
    print("Fase 1: Esecuzione dei programmi...")
    execute_programs()
    
    print("\nOperazione completata.")
