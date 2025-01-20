# Impostazioni generali
set terminal pngcairo size 800,600 enhanced
set output 'streamlines_plot.png'

# Definisci i limiti della griglia (assumiamo che N = 100 e M = 100)
N = 100
M = 100

# Impostazioni per la visualizzazione
set xlabel "Asse X"
set ylabel "Asse Y"
set title "Streamlines and Obstacle"
set grid

# Limiti degli assi
set xrange [0:N]
set yrange [0:M]

# Colore per l'ostacolo (nero)
set style line 1 lc rgb "black" lw 2

# Colori per le streamline (puoi aggiungere una mappa di colori o usare un solo colore)
set style line 2 lc rgb "blue" lw 1

# Percorso della cartella StreamData
stream_folder = 'StreamData/'
object_folder = 'ObjectData/'

# Usa il ciclo do per tracciare le streamline
# Itera sulle posizioni iniziali delle streamline

# Plot delle streamline separate ogni 5 unità lungo l'asse y
plot for [i=0:N-1:5] for [j=0:M-1:5] \
    stream_folder . 'streamline_' . i . '_' . j . '.dat' using 1:2 with lines linestyle 2 notitle, \
    object_folder . 'object.dat' using 1:2 with points linestyle 1 title "object"
