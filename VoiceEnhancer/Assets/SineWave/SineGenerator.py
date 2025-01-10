import numpy as np
from scipy.io.wavfile import write

# Parametri
sample_rate = 44100  # Hz
duration = 5  # secondi
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Sinusoidale a 300 Hz
freq_sine = 300  # Hz
sine_wave = 0.5 * np.sin(2 * np.pi * freq_sine * t)  # Amplitudine ridotta a 0.5

# Rumore nei range 400-3000 Hz e 0-200 Hz
fft_size = len(t)
frequencies = np.fft.rfftfreq(fft_size, d=1/sample_rate)

# Genera rumore bianco e filtra nelle bande richieste
noise = np.random.normal(0, 0.05, size=fft_size)  # Rumore bianco
noise_fft = np.fft.rfft(noise)

# Filtra il rumore per includere solo 400-3000 Hz e 0-200 Hz
mask = ((frequencies >= 400) & (frequencies <= 3000)) | (frequencies <= 200)
filtered_noise_fft = noise_fft * mask
filtered_noise = np.fft.irfft(filtered_noise_fft, n=fft_size)

# Combina sinusoidale e rumore
audio_signal = sine_wave + filtered_noise

# Normalizza per evitare clipping
audio_signal /= np.max(np.abs(audio_signal))

# Salva il file audio
output_path = "/mnt/data/sine_and_filtered_noise.wav"
write(output_path, sample_rate, (audio_signal * 32767).astype(np.int16))

output_path
