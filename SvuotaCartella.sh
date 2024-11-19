#!/bin/bash

# Controlla se è stato fornito un argomento
if [ -z "$1" ]; then
  echo "Errore: specifica il percorso della cartella da svuotare."
  exit 1
fi

# Verifica se la cartella esiste
if [ ! -d "$1" ]; then
  echo "Errore: la cartella '$1' non esiste."
  exit 1
fi

# Svuota la cartella
rm -rf "$1"/*

echo "La cartella '$1' è stata svuotata."
