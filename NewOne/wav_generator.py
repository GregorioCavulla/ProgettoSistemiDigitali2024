import numpy as np
import wave
import os

def generate_wav_file(filename, duration, sample_rate=44100):
    """
    Genera un file .wav contenente rumore bianco (tutte le frequenze alla stessa intensità).

    Parameters:
    - filename (str): Nome del file .wav di output.
    - duration (float): Durata del file audio in secondi.
    - sample_rate (int): Frequenza di campionamento (default: 44100 Hz).
    """
    # Genera il rumore bianco
    num_samples = int(sample_rate * duration)
    audio_data = np.random.uniform(-1, 1, num_samples)

    # Converte i dati in formato PCM a 16 bit
    audio_data = (audio_data * 32767).astype(np.int16)

    # Scrive il file .wav
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

# Genera i file di test
os.makedirs("test_wav", exist_ok=True)

# File brevi (durata minima)
# generate_wav_file("test_wav/short_noise.wav", duration=0.5)  # Breve

# File lunghi (durata apprezzabile)
# generate_wav_file("test_wav/long_noise.wav", duration=5.0)  # Lungo

# File molto lunghi (durata apprezzabile)
generate_wav_file("test_wav/veryLong_noise.wav", duration=15.0)  # Molto lungo

print("File .wav generati nella cartella 'test_wav':")
print("- short_noise.wav\n- long_noise.wav")
