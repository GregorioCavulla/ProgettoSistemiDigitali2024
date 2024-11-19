set term pngcairo size 800,600
set output 'streamlines.png'
set title "Streamline del flusso d'aria"
set xlabel "X"
set ylabel "Y"
set palette defined (0 "blue", 1 "green", 2 "yellow", 3 "red")
set cblabel "Velocità"
unset key

plot for [i=0:4] sprintf("streamline_%d.dat", i) using 1:2:3 with lines lc palette