import numpy as np
import matplotlib.pyplot as plt
import wave
import os

def read_wav(filename):
    """
    Legge un file WAV e restituisce i dati audio e la frequenza di campionamento.

    Parameters:
    - filename (str): Nome del file WAV da leggere.

    Returns:
    - sample_rate (int): Frequenza di campionamento.
    - audio_data (numpy array): Dati audio normalizzati (-1 a 1).
    - channels (int): Numero di canali (1 per mono, 2 per stereo).
    """
    with wave.open(filename, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        raw_data = wav_file.readframes(num_frames)

        # Converte i dati in array numpy
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        # Se stereo, separa i canali
        if num_channels == 2:
            audio_data = audio_data.reshape(-1, 2)

        # Normalizza i dati (-1 a 1)
        audio_data = audio_data / 32768.0

    return sample_rate, audio_data, num_channels

def plot_audio(sample_rate, audio_data, title, channels):
    """
    Traccia un grafico del segnale audio.

    Parameters:
    - sample_rate (int): Frequenza di campionamento.
    - audio_data (numpy array): Dati audio.
    - title (str): Titolo del grafico.
    - channels (int): Numero di canali (1 per mono, 2 per stereo).
    """
    time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))

    plt.figure(figsize=(12, 6))
    if channels == 1:
        plt.plot(time, audio_data, label='Mono')
    else:
        plt.plot(time, audio_data[:, 0], label='Left Channel', alpha=0.7)
        plt.plot(time, audio_data[:, 1], label='Right Channel', alpha=0.7)

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    """
    Legge file audio generati prima e traccia i grafici per evidenziare l'effetto del filtro.
    """
    input_files = ["test_wav/short_stereo.wav", "test_wav/long_stereo.wav"]
    output_files = ["output_short_stereo.wav", "output_long_stereo.wav"]

    for i, input_file in enumerate(input_files):
        if os.path.exists(input_file) and os.path.exists(output_files[i]):
            # Legge i file audio
            sample_rate_in, audio_data_in, channels_in = read_wav(input_file)
            sample_rate_out, audio_data_out, channels_out = read_wav(output_files[i])

            # Grafico per il file originale
            plot_audio(
                sample_rate_in, 
                audio_data_in, 
                f"Original Audio ({os.path.basename(input_file)})", 
                channels_in
            )

            # Grafico per il file filtrato
            plot_audio(
                sample_rate_out, 
                audio_data_out, 
                f"Filtered Audio ({os.path.basename(output_files[i])})", 
                channels_out
            )

if __name__ == "__main__":
    main()
