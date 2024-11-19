// Con questo codice voglio generare delle linee continue nei valori di x ma ogni 50 y parallele all'asse x.
// In una griglia 1000 x 1000
// Le linee le voglio rappresentare in un file .dat che sarà poi usato per generare un grafico con gnuplot.

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Dimensioni della griglia
#define N 1000  // asse x
#define M 1000  // asse y

// Numero massimo di passi per una streamline
#define MAX_STEPS 500

// Passo temporale per il calcolo delle streamline
#define DT 0.1

// Numero di linee da generare (default 20)
#define NUM_LINES 20  // Numero di linee da generare (modifica qui per cambiarlo)

// 