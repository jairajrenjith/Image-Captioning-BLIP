from datasets import load_dataset
import json, os

os.makedirs("data", exist_ok=True)

dataset = load_dataset("conceptual_captions", split="train")
dataset = dataset.shuffle(seed=42).select(range(1500))

captions = [{"image_url": d["image_url"], "caption": d["caption"]} for d in dataset]

with open("data/captions.json", "w") as f:
    json.dump(captions, f)

print("Captions metadata saved.")
