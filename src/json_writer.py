import os, json

def load_results(json_path="results.json"):
    """Load existing results file or return empty structure."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    else:
        return {
            "info": {"description": "Grand Challenge UDA", "version": "1.0", "year": 2025},
            "audios": [],
            "categories": [
                {"id": 1, "name": "vessel"},
                {"id": 2, "name": "marine_animal"},
                {"id": 3, "name": "natural_sound"},
                {"id": 4, "name": "other_anthropogenic"},
            ],
            "annotations": []
        }

def save_results(data, json_path="results.json"):
    """Save results to file."""
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
