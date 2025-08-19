import numpy as np
import librosa

CATEGORIES = {1:"vessel",2:"marine_animal",3:"natural_sound",4:"other_anthropogenic"}

def _mean(a): 
    return float(np.nanmean(a)) if np.size(a) else 0.0

def _segment(y, sr, s, e):
    i0, i1 = max(0, int(s*sr)), max(0, int(e*sr))
    if i1 <= i0: i1 = min(len(y), i0 + int(0.05*sr))
    return y[i0:i1]

def classify_event_from_file(filepath, start_s, end_s):
    y, sr = librosa.load(filepath, sr=16000, mono=True)
    seg = _segment(y, sr, start_s, end_s)
    if seg.size == 0: 
        return 4, CATEGORIES[4], 0.2

    seg = seg / (np.max(np.abs(seg)) + 1e-9)

    centroid = _mean(librosa.feature.spectral_centroid(y=seg, sr=sr))
    bandwidth = _mean(librosa.feature.spectral_bandwidth(y=seg, sr=sr))
    flatness = _mean(librosa.feature.spectral_flatness(y=seg))
    rolloff  = _mean(librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85))
    zcr      = _mean(librosa.feature.zero_crossing_rate(seg))
    dur      = len(seg)/sr

    low_c   = centroid < 500
    mid_c   = 500 <= centroid < 2000
    high_c  = centroid >= 2000
    narrow  = bandwidth < 800
    wide    = bandwidth >= 1200
    noisy   = flatness > 0.4
    tonal   = flatness < 0.2
    short   = dur < 0.6

    if low_c and tonal and not short:
        conf = min(1.0, 0.6 + (0.2 if rolloff < 1500 else 0.0) + (0.2 if zcr < 0.08 else 0.0))
        return 1, CATEGORIES[1], conf
    if (mid_c or high_c) and tonal and narrow:
        conf = min(1.0, 0.6 + (0.2 if rolloff < 5000 else 0.0) + (0.2 if zcr < 0.08 else 0.0))
        return 2, CATEGORIES[2], conf
    if noisy and wide and not short:
        conf = min(1.0, 0.6 + (0.2 if centroid < 1500 else 0.0) + (0.2 if zcr < 0.12 else 0.0))
        return 3, CATEGORIES[3], conf
    conf = min(1.0, 0.5 + (0.2 if high_c else 0.0) + (0.2 if short else 0.0))
    return 4, CATEGORIES[4], conf
