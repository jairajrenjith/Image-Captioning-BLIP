import torch, pickle
from PIL import Image
from utils.transforms import get_transforms
from models.encoder import Encoder
from models.decoder import Decoder
from models.caption_model import CaptionModel

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

encoder = Encoder(256)
decoder = Decoder(256, 512, len(vocab.itos))
model = CaptionModel(encoder, decoder).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = get_transforms()

def generate_caption(image_path, max_len=20):
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encoder(image)

    caption = [vocab.stoi["<SOS>"]]

    for _ in range(max_len):
        cap_tensor = torch.tensor(caption).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.decoder(features, cap_tensor)

        next_word = output.argmax(2)[:, -1].item()
        caption.append(next_word)

        if next_word == vocab.stoi["<EOS>"]:
            break

    words = [vocab.itos.get(idx, "<UNK>") for idx in caption[1:-1]]
    return " ".join(words)

print(generate_caption("images/0.jpg"))
