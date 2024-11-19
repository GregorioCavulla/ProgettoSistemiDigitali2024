# Script gnuplot per visualizzare la griglia con righe orizzontali
set terminal pngcairo size 800, 800  # Salva l'immagine come PNG
set output 'griglia.png'             # Nome del file immagine di output
set title 'Griglia con righe orizzontali'
set xlabel 'X'
set ylabel 'Y'

# Imposta gli assi per la visualizzazione della griglia
set xrange [0:1000]  # Imposta il range dell'asse x
set yrange [0:1000]  # Imposta il range dell'asse y

# Imposta le linee della griglia
set grid

# Plotta il file 'griglia.dat' con linee
plot 'griglia.dat' with lines notitle

# Chiudi l'output
unset output
