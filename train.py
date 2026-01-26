import torch, json, pickle
from torch.utils.data import DataLoader
from utils.vocabulary import Vocabulary
from utils.dataset import CaptionDataset
from utils.transforms import get_transforms
from utils.collate_fn import collate_fn
from models.encoder import Encoder
from models.decoder import Decoder
from models.caption_model import CaptionModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

data = json.load(open("data/cleaned_data.json"))
captions = [d["caption"] for d in data]

vocab = Vocabulary(freq_threshold=3)
vocab.build_vocab(captions)

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

dataset = CaptionDataset("data/cleaned_data.json", get_transforms(), vocab)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

encoder = Encoder(256)
decoder = Decoder(256, 512, len(vocab.itos))
model = CaptionModel(encoder, decoder).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(decoder.parameters(), lr=3e-4)

for epoch in range(3):
    model.train()
    loop = tqdm(loader)
    for imgs, caps in loop:
        imgs, caps = imgs.to(device), caps.to(device)

        outputs = model(imgs, caps[:, :-1])
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), caps[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "model.pth")
