import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def load_and_melspec(filepath, sr=16000, n_fft=2048, hop_length=512, n_mels=128):
    y, sr = librosa.load(filepath, sr=sr, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db, sr

def save_spectrogram_image(mel_db, out_path):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
