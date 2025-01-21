#!/bin/bash

# Itera su tutti i file che corrispondono al pattern
for file in DFTparallelo_v*.o; do
    # Controlla se il file è eseguibile
    if [[ -x "$file" ]]; then
        echo "Esecuzione del file: $file"
        for i in {1..5}; do
            echo "Esecuzione numero $i di $file"
            ./"$file" # Lancia il file
        done
    else
        echo "Il file $file non è eseguibile. Verifica i permessi."
    fi
done
