import os, json, requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

os.makedirs("images", exist_ok=True)

with open("data/captions.json") as f:
    data = json.load(f)

valid = []

for idx, item in enumerate(tqdm(data)):
    try:
        r = requests.get(item["image_url"], timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        path = f"images/{idx}.jpg"
        img.save(path)
        valid.append({"image_path": path, "caption": item["caption"]})
    except:
        continue

with open("data/cleaned_data.json", "w") as f:
    json.dump(valid, f)

print("Images downloaded:", len(valid))
