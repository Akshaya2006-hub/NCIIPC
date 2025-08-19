import glob, os, soundfile as sf
from preprocess import load_and_melspec, save_spectrogram_image
from detect import detect_events
from json_writer import load_results, save_results
from classify import classify_event_from_file

# Path to the dataset
AUDIO_ROOT = r"D:\uda_project\DeepShip"

# Load existing results.json
results = load_results("results.json")
processed_files = {audio["file_path"] for audio in results["audios"]}

audio_id = len(results["audios"]) + 1
ann_id = len(results["annotations"]) + 1

# Search all .wav files in all subfolders
for f in glob.glob(os.path.join(AUDIO_ROOT, "**", "*.wav"), recursive=True):
    if f in processed_files:
        print(f"Skipping {f} (already processed)")
        continue

    fname = os.path.basename(f)
    print(f"\nProcessing {fname} ...")
    mel_db, sr = load_and_melspec(f)

    os.makedirs("spectrograms", exist_ok=True)
    save_spectrogram_image(mel_db, os.path.join("spectrograms", f"{fname}_spectrogram.png"))

    events = detect_events(mel_db, sr)

    data, sr2 = sf.read(f)
    dur = len(data) / sr2

    results["audios"].append({
        "id": audio_id,
        "file_name": fname,
        "file_path": os.path.abspath(f),
        "duration": float(dur)
    })

    for e in events:
        start, end, score = e
        cat_id, cat_name, cat_conf = classify_event_from_file(f, start, end)

        results["annotations"].append({
            "id": ann_id,
            "audio_id": audio_id,
            "category_id": int(cat_id),
            "category_name": cat_name,
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(end - start, 2),
            "score": round(min(1.0, (score + cat_conf) / 2.0), 2)
        })
        ann_id += 1

    audio_id += 1

# Save results
save_results(results, "results.json")

print(f"\n✅ Processed audios: {len(results['audios'])}")
print(f"✅ Processed annotations: {len(results['annotations'])}")
