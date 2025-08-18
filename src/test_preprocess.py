from preprocess import load_and_melspec, save_spectrogram_image

# Replace "example.wav" with any .wav file you have
mel_db, sr = load_and_melspec("example.wav")
print("Spectrogram shape:", mel_db.shape, "Sample rate:", sr)

# Save image
save_spectrogram_image(mel_db, "check_spectrogram.png")
print("Spectrogram image saved as check_spectrogram.png")
