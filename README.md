# 🖼️ Image Captioning with Xception + LSTM

> Automatically generate natural language descriptions for images using a CNN-LSTM deep learning architecture, trained on the Flickr8k dataset.

---

## 📌 Overview

This project implements an end-to-end **image captioning** system that:
1. Extracts visual features from images using a pre-trained **Xception** CNN
2. Encodes captions and trains a **LSTM**-based language model
3. Generates descriptive captions for unseen images using **beam search**

Built and trained on **Google Colab** with **Google Drive** for storage.

---

## 🧠 Model Architecture

```
Image ──► Xception (pretrained, frozen)
              │
              ▼
         Dense(256) ──────────────┐
                                  ▼
Caption ──► Embedding ──► LSTM(256) ──► add() ──► Dense(256) ──► Dense(vocab_size, softmax)
```

| Component | Details |
|---|---|
| Feature Extractor | Xception (ImageNet pretrained, no top, avg pooling → 2048-d) |
| Embedding | 256-d word embeddings |
| Sequence Model | Single-layer LSTM (256 units) |
| Fusion | Element-wise addition of image & text features |
| Output | Softmax over vocabulary |
| Loss | Categorical Cross-Entropy |
| Optimizer | Adam |

---

## 📂 Project Structure

```
ImageCaptioning/
│
├── ImageCaptioning_Clean.ipynb   # Main notebook (Colab)
├── README.md
│
└── (Google Drive)/
    ├── Flickr8k_Dataset/         # ~8,000 images
    ├── Flickr8k_text.txt         # Raw captions file
    ├── descriptions.txt          # Cleaned captions (generated)
    ├── features.pkl              # Extracted Xception features (generated)
    ├── tokenizer.pkl             # Fitted tokenizer (generated)
    ├── xception.weights.h5       # Cached Xception weights (generated)
    └── models/
        ├── model_epoch_1.h5
        ├── model_epoch_2.h5
        └── ...
```

---

## 🗂️ Dataset

**Flickr8k** — 8,092 images, each with 5 human-written captions (40,460 total).

| Split | Images |
|---|---|
| Train | ~8,091 |
| Vocabulary | 8,344 unique words |
| Max caption length | 34 tokens |

Download the dataset from:
- [Kaggle – Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

Place files in your Google Drive under `MyDrive/ImageCaptioning/`.

---

## ⚙️ Pipeline

```
Step 1  Mount Google Drive
Step 2  Imports & path configuration
Step 3  Load & clean captions (lowercase, strip punctuation)
Step 4  Extract Xception features (cached to Drive)
Step 5  Load descriptions & build tokenizer
Step 6  Data generator & model definition
Step 7  Train model (saves checkpoint each epoch)
Step 8  Load saved model for inference
Step 9  Generate & display captions (greedy or beam search)
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/image-captioning.git
cd image-captioning
```

### 2. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload `ImageCaptioning.ipynb` to Colab or open directly from your Drive.

### 3. Set up Google Drive

Organise your Drive as follows:

```
MyDrive/
└── ImageCaptioning/
    ├── Flickr8k_Dataset/    ← unzip images here
    └── Flickr8k_text.txt    ← captions file
```

### 4. Run the notebook

Execute all cells from top to bottom. The notebook handles:
- Automatic weight caching (no re-download on Colab restart)
- Feature extraction caching (skip if `features.pkl` exists)
- Model checkpointing after every epoch

---

## 📊 Training

```python
EPOCHS           = 10     # recommended: 15–20 for best results
VALIDATION_SPLIT = 0.2
steps_per_epoch  = len(train_desc)   # full dataset per epoch
```

Model checkpoints are saved after every epoch:
```
models/model_epoch_1.h5
models/model_epoch_2.h5
...
```

> **Tip:** Train for at least **15 epochs** for meaningful captions. Early epochs (1–3) tend to produce generic outputs like *"dog is playing in the grass"*.

---

## 🔍 Caption Generation

Two decoding strategies are supported:

### Greedy Search
Picks the highest-probability word at each step — fast but sometimes repetitive.

### Beam Search *(recommended)*
Maintains multiple candidate sequences and returns the globally best one.

```python
caption = generate_caption_beam(
    model     = caption_model,
    tokenizer = tokenizer,
    photo     = photo_feature,   # 2048-d Xception vector
    max_len   = max_len,
    beam_width= 3                # increase for better (but slower) results
)
```

---

## 🖼️ Sample Output

| Image | Generated Caption |
|---|---|
| Two dogs with a toy | *dog is playing in the grass* (epoch 2) |
| *(improves with more epochs)* | *two dogs are playing with green toy on carpet* (epoch 15+) |

---

## 🛠️ Requirements

All dependencies are pre-installed on Google Colab. For local setup:

```bash
pip install tensorflow pillow tqdm matplotlib numpy
```

| Package | Version |
|---|---|
| TensorFlow / Keras | ≥ 2.12 |
| NumPy | ≥ 1.23 |
| Pillow | ≥ 9.0 |
| tqdm | ≥ 4.0 |
| Matplotlib | ≥ 3.5 |

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [Xception – Chollet (2017)](https://arxiv.org/abs/1610.02357)
- [Show and Tell – Vinyals et al. (2015)](https://arxiv.org/abs/1411.4555)
