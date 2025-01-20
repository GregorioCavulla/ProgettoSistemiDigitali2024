set terminal pngcairo size 1024, 1024
set output 'vettori.png'
set title 'Vettori in movimento'
set xlabel 'X'
set ylabel 'Y'
set grid

# Aggiungi tutti i file .dat generati dal programma
# In questo caso, gnuplot carica tutti i file vettori_0.dat, vettori_1.dat, ..., vettori_9.dat

# Loop per plottare tutti i file .dat
plot for [i=0:9] sprintf('./Data/vettori_%d.dat', i) using 1:2:3:4 with vectors head filled lt 1 notitle
