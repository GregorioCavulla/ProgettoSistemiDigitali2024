set terminal pngcairo size 800, 800
set output 'traiettoria.png'

# Imposta il nome del grafico
set title "Traiettoria dei Punti"

# Imposta le etichette degli assi
set xlabel "X"
set ylabel "Y"

# Imposta la dimensione della griglia
set xrange [0:100]
set yrange [0:100]

# Abilita la griglia
set grid

# Disabilita la legenda
set key off

# Carica i dati da tutti i file .dat generati dal programma C
# La struttura del file .dat è: x y (posizione dei punti)
# Disegnamo i punti nei rispettivi file

# Carica i dati da più file per ogni istante di tempo
plot for [i=0:99] './Data/traiettoria_'.i.'.dat' using 1:2 with points pt 7 ps 1 lc rgb "blue" title "Traiettoria"
