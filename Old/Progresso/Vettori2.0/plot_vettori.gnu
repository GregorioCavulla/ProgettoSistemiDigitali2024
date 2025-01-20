set xrange [0:100]
set yrange [0:100]
set terminal pngcairo size 800, 800
set output 'vettori.png'
set title 'Vettori in movimento'
set xlabel 'X'
set ylabel 'Y'
set grid

# Aggiungi tutti i file .dat generati dal programma
plot for [i=0:9] sprintf('./Data/vettori_%d.dat', i) using 1:2:3:4 with vectors head filled lt 1 notitle
