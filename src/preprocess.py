import librosa   # (library for audio processing)
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def load_and_melspec(filepath, target_sr=16000, n_mels=64):
    """
    Load a .wav file, convert it to mono, resample to 16kHz,
    normalize, and create a log-mel spectrogram.
    """

    # Load audio
    y, sr = librosa.load(filepath, sr=target_sr, mono=True)

    # Normalize (make loudness consistent)
    y = y / np.max(np.abs(y))

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512
    )

    # Convert to log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db, sr

def save_spectrogram_image(mel_db, out_file="spectrogram.png"):
    """Save mel spectrogram as an image for checking."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
