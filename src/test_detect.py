import glob, os, soundfile as sf
from preprocess import load_and_melspec, save_spectrogram_image
from detect import detect_events
from json_writer import load_results, save_results

# Load existing results
results = load_results("results.json")
processed_files = {audio["file_name"] for audio in results["audios"]}

audio_id = len(results["audios"]) + 1
ann_id = len(results["annotations"]) + 1

for f in glob.glob("*.wav"):
    if f in processed_files:
        print(f"Skipping {f} (already processed)")
        continue

    print(f"\nProcessing {f} ...")
    mel_db, sr = load_and_melspec(f)

    # ✅ Save spectrogram image also
    save_spectrogram_image(mel_db, f"{f}_spectrogram.png")

    events = detect_events(mel_db, sr)

    # Get audio duration
    data, sr2 = sf.read(f)
    dur = len(data) / sr2

    # Save audio entry
    results["audios"].append({
        "id": audio_id,
        "file_name": f,
        "file_path": os.path.abspath(f),
        "duration": float(dur)
    })

    # Save annotations
    for e in events:
        results["annotations"].append({
            "id": ann_id,
            "audio_id": audio_id,
            "category_id": 1,  # placeholder until classification
            "start_time": round(e[0], 2),
            "end_time": round(e[1], 2),
            "duration": round(e[1] - e[0], 2),
            "score": round(e[2], 2)
        })
        ann_id += 1

    audio_id += 1

# Save results back to JSON
save_results(results, "results.json")
print("\n✅ Updated results.json and saved spectrogram images")
