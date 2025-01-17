import numpy as np
import wave
import os

def generate_wav_file(filename, duration, sample_rate=44100, mono=True):
    """
    Genera un file .wav mono o stereo.

    Parameters:
    - filename (str): Nome del file .wav di output.
    - duration (float): Durata del file audio in secondi.
    - sample_rate (int): Frequenza di campionamento (default: 44100 Hz).
    - mono (bool): Se True, genera un file mono; altrimenti stereo.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq_left = 440.0  # Frequenza del canale sinistro (A4)
    freq_right = 880.0  # Frequenza del canale destro (A5, solo per stereo)

    # Genera i campioni per il canale sinistro
    left_channel = 0.5 * np.sin(2 * np.pi * freq_left * t)

    if mono:
        # Per audio mono, usa solo il canale sinistro
        audio_data = left_channel
    else:
        # Genera i campioni per il canale destro (se stereo)
        right_channel = 0.5 * np.sin(2 * np.pi * freq_right * t)
        # Combina i due canali in un unico array interlacciato
        audio_data = np.empty((left_channel.size + right_channel.size,), dtype=np.float32)
        audio_data[0::2] = left_channel
        audio_data[1::2] = right_channel

    # Converte i dati in formato PCM a 16 bit
    audio_data = (audio_data * 32767).astype(np.int16)

    # Scrive il file .wav
    with wave.open(filename, 'w') as wav_file:
        num_channels = 1 if mono else 2
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

# Genera i file di test
os.makedirs("test_wav", exist_ok=True)

# File brevi (durata minima)
generate_wav_file("test_wav/short_mono.wav", duration=0.5, mono=True)  # Mono breve
generate_wav_file("test_wav/short_stereo.wav", duration=0.5, mono=False)  # Stereo breve

# File lunghi (durata apprezzabile)
generate_wav_file("test_wav/long_mono.wav", duration=5.0, mono=True)  # Mono lungo
generate_wav_file("test_wav/long_stereo.wav", duration=5.0, mono=False)  # Stereo lungo

print("File .wav generati nella cartella 'test_wav':")
print("- short_mono.wav\n- short_stereo.wav\n- long_mono.wav\n- long_stereo.wav")
