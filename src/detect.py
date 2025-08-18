import numpy as np

def detect_events(mel_db, sr, hop_length=512, threshold_db=-40):
    """
    Detect ALL sound events in a mel spectrogram.
    No duration filter applied — every detected change is stored.

    mel_db: log-mel spectrogram (2D array from preprocess)
    sr: sample rate (e.g. 16000)
    hop_length: hop size used in spectrogram (default 512)
    threshold_db: level above which we call it 'sound present'

    Returns: list of (start_time, end_time, score)
    """

    # 1. Collapse across frequency (average loudness at each time step)
    energy = mel_db.mean(axis=0)

    # 2. Find frames above threshold
    active = energy > threshold_db

    events = []
    in_event = False
    start, end = 0, 0

    for i, is_active in enumerate(active):
        if is_active and not in_event:
            # Start of a new event
            in_event = True
            start = i
        elif not is_active and in_event:
            # End of an event
            in_event = False
            end = i
            start_time = start * hop_length / sr
            end_time = end * hop_length / sr
            score = float(energy[start:end].max() / np.max(energy))
            events.append((start_time, end_time, score))

    # If audio ends while still in an event
    if in_event:
        end = len(active)
        start_time = start * hop_length / sr
        end_time = end * hop_length / sr
        score = float(energy[start:end].max() / np.max(energy))
        events.append((start_time, end_time, score))

    return events
