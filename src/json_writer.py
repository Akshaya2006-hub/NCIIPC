import json
import os

def load_results(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        return {
            "info": {
                "description": "Grand Challenge UDA",
                "version": "1.0",
                "year": 2025
            },
            "audios": [],
            "categories": [
                {"id": 1, "name": "vessel"},
                {"id": 2, "name": "marine_animal"},
                {"id": 3, "name": "natural_sound"},
                {"id": 4, "name": "other_anthropogenic"}
            ],
            "annotations": []
        }

def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to {path}")
