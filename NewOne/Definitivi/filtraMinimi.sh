#!/bin/bash

# Directory dei report
REPORT_DIR="./reports"
DEST_DIR="./reports_minimi"

# Crea la directory di destinazione, se non esiste
mkdir -p "$DEST_DIR"

# Trova tutte le versioni uniche X.X dai file report
versions=$(ls "$REPORT_DIR" | grep -oP "DFTparallelo_v\d+\.\d+" | sort -u)

# Per ogni versione
for version in $versions; do
    echo "Elaborazione per la versione: $version"
    min_file=""
    min_time=9999999

    # Trova tutti i file della stessa versione
    for file in "$REPORT_DIR"/${version}_*; do
        if [[ -f "$file" ]]; then
            # Estrai il tempo totale dal file
            total_time=$(grep "Totale:" "$file" | grep -oP "[0-9]+\.[0-9]+")
            
            # Confronta il tempo totale
            if (( $(echo "$total_time < $min_time" | bc -l) )); then
                min_time=$total_time
                min_file=$file
            fi
        fi
    done

    # Sposta il file con il tempo minore nella directory di destinazione
    if [[ -n "$min_file" ]]; then
        echo "File con tempo minore: $min_file (Tempo totale: $min_time secondi)"
        cp "$min_file" "$DEST_DIR"
    else
        echo "Nessun file trovato per la versione $version."
    fi
done

echo "Operazione completata. I file con i tempi minori sono stati copiati in $DEST_DIR."
