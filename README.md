# Image Captioning using Multiple Vision-Language Models

An image captioning project that demonstrates three different approaches to generating natural language descriptions from images: a Hugging Face pretrained model (BLIP), a CNN–RNN encoder–decoder model, and a Vision Transformer (ViT) + RNN model. The project highlights how different vision-language architectures can be used for image captioning, including both pretrained and custom-trained models.

---

## Features

- Hugging Face pretrained BLIP model fine-tuning
- CNN–RNN image captioning model implementation
- Vision Transformer (ViT) + RNN image captioning model
- Uses the Conceptual Captions dataset
- Demonstrates training and inference pipelines
- Works on CPU or GPU

---

## Models Used

### 1. Hugging Face Pretrained Model (BLIP)
- Vision Transformer (ViT) image encoder
- Transformer-based language decoder
- Fine-tuned on a subset of the Conceptual Captions dataset

### 2. CNN–RNN Model
- ResNet-based CNN image encoder
- LSTM-based RNN decoder
- Trained using image–caption pairs

### 3. Vision Transformer (ViT) + RNN Model
- Vision Transformer image encoder
- GRU-based RNN decoder
- Demonstrates transformer-based visual encoding with sequential caption generation

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
├── 01_hf_finetuning.ipynb
├── 02_CNN_RNN_Image_Captioning.ipynb
├── 03_ViT_RNN_Image_Captioning.ipynb
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

3. Run the Notebooks:

  ```bash
  01_hf_finetuning.ipynb (Hugging Face BLIP fine-tuning)
  02_CNN_RNN_Image_Captioning.ipynb (CNN–RNN model)
  03_ViT_RNN_Image_Captioning.ipynb (ViT–RNN model)
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

## Example Outputs

- **BLIP (Fine-Tuned):** Generates fluent natural language captions for images.
- **CNN–RNN:** Produces token-level outputs demonstrating the encoder–decoder pipeline.
- **ViT–RNN:** Produces token-level outputs demonstrating transformer-based visual encoding with RNN decoding.

> Note: CNN–RNN and ViT–RNN models demonstrate the training and inference pipeline using minimal training and simplified vocabularies, hence outputs are shown at token level rather than fluent sentences.

---

## Multiple Caption Generation

The Hugging Face BLIP notebook supports generating multiple captions using beam search for more descriptive variations.

---

## Why Use Pretrained Models?

Training image captioning models from scratch requires:
- Large-scale datasets
- Powerful GPUs
- Significant training time

In this project, pretrained models are used where appropriate to demonstrate practical and efficient image captioning. The Hugging Face BLIP model leverages large-scale pretraining to produce fluent captions with minimal fine-tuning, while the CNN–RNN and ViT–RNN models are implemented to demonstrate custom encoder–decoder architectures and training pipelines.


---

## References

- BLIP: Bootstrapped Language-Image Pretraining  
  https://arxiv.org/abs/2201.12086  

- Hugging Face BLIP Model  
  https://huggingface.co/Salesforce/blip-image-captioning-base  

- Show and Tell: A Neural Image Caption Generator  
  https://arxiv.org/abs/1411.4555  

- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale  
  https://arxiv.org/abs/2010.11929

---

## License

This project uses pretrained models from Hugging Face and is intended for educational and research purposes.

---

By Jairaj R.
