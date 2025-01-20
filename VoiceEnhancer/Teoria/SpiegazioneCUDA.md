Questo codice implementa un filtro audio che utilizza la Trasformata di Fourier Discreta (DFT) e la sua inversa (IDFT) per modificare un file WAV mono a 16-bit. Inoltre, è parallelizzato utilizzando CUDA per sfruttare la GPU.

Ecco una spiegazione dettagliata del codice, con focus sui principali blocchi e kernel CUDA.

---

### **1. Struttura e Lettura dell'Header WAV**
- La struttura `WAVHeader` rappresenta i metadati del file WAV, inclusi informazioni sul formato, frequenza di campionamento, numero di canali e dimensione dei dati audio.
- La funzione `readWAVHeader` legge l'header dal file e si assicura che il blocco `data` sia raggiunto.

```c
fread(header, sizeof(WAVHeader), 1, file);
while (strncmp(header->subchunk2ID, "data", 4) != 0) {
    fseek(file, header->subchunk2Size, SEEK_CUR);
    fread(&header->subchunk2ID, sizeof(header->subchunk2ID), 1, file);
    fread(&header->subchunk2Size, sizeof(header->subchunk2Size), 1, file);
}
```

### **2. Kernel CUDA per DFT**
Il kernel `computeDFTKernel` calcola la DFT di un segnale audio:
- Ogni thread calcola i valori reali (`real[k]`) e immaginari (`imag[k]`) per un indice \( k \).
- La somma dei contributi viene calcolata iterando su \( n \) e moltiplicando i campioni del segnale con le funzioni coseno e seno.

```c
__global__ void computeDFTKernel(double *input, double *real, double *imag, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        real[k] = 0;
        imag[k] = 0;
        for (int n = 0; n < N; n++) {
            double angle = TWO_PI * k * n / N;
            real[k] += input[n] * cos(angle);
            imag[k] -= input[n] * sin(angle);
        }
    }
}
```

### **3. Kernel CUDA per IDFT**
Il kernel `computeIDFTKernel` calcola la IDFT:
- Ogni thread calcola \( output[n] \), somma di tutte le componenti frequenziali.
- I valori vengono normalizzati dividendo per \( N \).

```c
__global__ void computeIDFTKernel(double *real, double *imag, double *output, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        output[n] = 0;
        for (int k = 0; k < N; k++) {
            double angle = TWO_PI * k * n / N;
            output[n] += real[k] * cos(angle) - imag[k] * sin(angle);
        }
        output[n] /= N;
    }
}
```

### **4. Calcolo della DFT e IDFT**
Le funzioni `computeDFT` e `computeIDFT` gestiscono il trasferimento dei dati tra CPU e GPU, e lanciano i kernel CUDA:
1. **Allocazione memoria sulla GPU** usando `cudaMalloc`.
2. **Copia dei dati** tra CPU e GPU con `cudaMemcpy`.
3. **Lancio del kernel** con configurazione di thread e blocchi.
4. **Copia dei risultati** dalla GPU alla CPU e rilascio della memoria GPU.

```c
cudaMalloc(&d_input, N * sizeof(double));
cudaMemcpy(d_input, input, N * sizeof(double), cudaMemcpyHostToDevice);
computeDFTKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_real, d_imag, N);
cudaMemcpy(real, d_real, N * sizeof(double), cudaMemcpyDeviceToHost);
```

### **5. Filtro Frequenziale**
Dopo aver calcolato la DFT, il codice applica un filtro passa-banda:
- Azzera le componenti frequenziali al di fuori dell'intervallo 85 Hz - 300 Hz.

```c
for (int k = 0; k < numSamples; k++) {
    double frequency = (double)k / numSamples;
    if (frequency < lowCut || frequency > highCut) {
        real[k] = 0;
        imag[k] = 0;
    }
}
```

### **6. Scrittura del File WAV Modificato**
- I dati filtrati vengono trasformati tramite la IDFT.
- I campioni risultanti vengono convertiti in interi a 16-bit e salvati nel file di output.

```c
for (int i = 0; i < numSamples; i++) {
    short sample = (short)output[i];
    fwrite(&sample, sizeof(short), 1, outputFile);
}
```

### **7. Ottimizzazione con CUDA**
Il codice sfrutta CUDA per accelerare i calcoli della DFT e della IDFT:
- Utilizza un numero ottimale di thread per blocco (256).
- Parallelizza i calcoli su più blocchi per gestire grandi quantità di campioni.

```c
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
computeDFTKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_real, d_imag, N);
```

---

### **Esecuzione**
1. **Compilazione con CUDA:**
   ```bash
   nvcc -o filtroWAV filtroWAV.cu -lm
   ```

2. **Esecuzione:**
   ```bash
   ./filtroWAV input.wav output.wav report.txt
   ```

### **Risultati**
- Il file audio filtrato viene salvato come `output.wav`.
- I tempi di elaborazione (DFT, IDFT, totale) vengono riportati in `report.txt`.

Se hai altre domande, posso approfondire! 😊