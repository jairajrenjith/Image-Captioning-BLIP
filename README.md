# Image Captioning with BLIP Transformer

A transformer-based image captioning project that uses a pretrained **BLIP (Bootstrapped Language Image Pretraining)** model to generate natural language descriptions from images. This project demonstrates how vision-language models can understand visual content and produce meaningful captions without expensive model training.

---

## Features

- Uses a **state-of-the-art pretrained BLIP model**
- Generates captions for **any input image**
- Works on **CPU or GPU**
- No model training required
- Clean and simple notebook-based implementation

---

## Model Used

**BLIP Image Captioning Base**  
From Salesforce Research via Hugging Face Transformers.

The model combines:
- A **Vision Transformer (ViT)** image encoder  
- A **Language Transformer** decoder  

to generate descriptive captions.

---

## Project Structure

```
Image-Captioning-BLIP/
│
├── data/
│   ├── captions.json
│   ├── cleaned_data.json
│   ├── download_dataset.py
│   └── download_images.py
│
├── images/
│
├── models/
│   ├── __pycache__/
│   ├── caption_model.py
│   ├── decoder.py
│   └── encoder.py
│
├── utils/
│   ├── __pycache__/
│   ├── collate_fn.py
│   ├── dataset.py
│   ├── transforms.py
│   └── vocabulary.py
│
├── venv/
│
├── Image_Captioning_BLIP.ipynb
├── model.pth
├── predict.py
├── requirements.txt
├── train.py
├── vocab.pkl
└── README.md
```

---

## Installation

Make sure Python 3.8+ is installed.

Create a `requirements.txt` file containing:

- torch
- torchvision
- datasets
- nltk
- pillow
- tqdm
- requests


Install dependencies with:

  ```bash
  pip install -r requirements.txt
  ```

---

## How to Run

1. Clone the repository:

  ```bash
  git clone <repo-link>
  cd image-captioning-blip
  ```

2. Launch Jupyter Notebook:

  ```bash
  jupyter notebook
  ```

3. Open the notebook:

  ```bash
  Image_Captioning_BLIP.ipynb
  ```

4. Run all cells step by step.

---

## Using Your Own Image

Place your image in the project folder and update the image path in the notebook:

```
image_path = "your_image.jpg"
```

Then run the caption generation cell.

---

## Example Output

Input image → Dog running in a field  

Model Output:
```
a dog running through a grassy field
```

---

## Multiple Caption Generation

The notebook also supports generating multiple captions using beam search for more descriptive variations.

---

## Why Use Pretrained Models?

Training image captioning models from scratch requires:
- Large datasets
- Powerful GPUs
- Long training time

Using pretrained BLIP allows:
- Faster results
- High-quality captions
- Practical deployment


---

## References

- BLIP Paper: https://arxiv.org/abs/2201.12086  
- Hugging Face Model: https://huggingface.co/Salesforce/blip-image-captioning-base  

---

## License

This project uses pretrained models from Hugging Face and is intended for educational and research purposes.

---

By Jairaj R.
