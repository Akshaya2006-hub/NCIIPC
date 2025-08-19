import numpy as np

def detect_events(mel_db, sr, hop_length=512, threshold_db=-40):
    energy = mel_db.mean(axis=0)
    active = energy > threshold_db

    events = []
    in_event = False
    start, end = 0, 0

    for i, is_active in enumerate(active):
        if is_active and not in_event:
            in_event = True
            start = i
        elif not is_active and in_event:
            in_event = False
            end = i
            start_time = start * hop_length / sr
            end_time = end * hop_length / sr
            score = float(energy[start:end].max() / (np.max(energy) + 1e-9))
            events.append((start_time, end_time, score))

    if in_event:
        end = len(active)
        start_time = start * hop_length / sr
        end_time = end * hop_length / sr
        score = float(energy[start:end].max() / (np.max(energy) + 1e-9))
        events.append((start_time, end_time, score))

    return events
