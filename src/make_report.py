import json, csv

with open("results.json","r") as f:
    data = json.load(f)

audio_lookup = {a["id"]: a for a in data["audios"]}
cat_lookup = {1:"vessel",2:"marine_animal",3:"natural_sound",4:"other_anthropogenic"}

with open("report.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["file_name","start_time(s)","end_time(s)","duration(s)","category","score"])
    for ann in data["annotations"]:
        a = audio_lookup.get(ann["audio_id"], {})
        w.writerow([
            a.get("file_name","unknown"),
            ann["start_time"],
            ann["end_time"],
            ann["duration"],
            cat_lookup.get(ann["category_id"], "unknown"),
            ann["score"]
        ])

print("✅ report.csv created")
