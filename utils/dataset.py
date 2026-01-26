import torch, json
from PIL import Image
from torch.utils.data import Dataset

class CaptionDataset(Dataset):
    def __init__(self, json_path, transform, vocab):
        self.data = json.load(open(json_path))
        self.transform = transform
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.transform(Image.open(item["image_path"]).convert("RGB"))

        caption = [self.vocab.stoi["<SOS>"]]
        caption += self.vocab.numericalize(item["caption"])
        caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(caption)
