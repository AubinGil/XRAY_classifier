# Vision Transformer (ViT) for Multilabel Chest X-ray Classification

This notebook implements a Vision Transformer (ViT) model for multilabel classification on chest X-ray images using PyTorch and Hugging Face's Transformers library. The goal is to compare its performance with an InceptionV3 model already implemented separately.

---

## 🔍 Project Objective

The objective is to classify chest X-ray images into multiple possible conditions, including:
- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia

This is a **multilabel classification** task where each image can have multiple associated labels.

---

## 🧪 Model Architecture

The implemented model is based on `google/vit-base-patch16-224-in21k` and fine-tuned for multilabel classification. Key modifications:
- Replaced the classification head with a linear layer of output size = number of labels.
- Used `BCEWithLogitsLoss` to support multilabel output.

---

## ⚙️ Setup

Install the required packages:

```bash
pip install transformers torch torchvision torchaudio pandas scikit-learn
```


!<img width="580" height="490" alt="image" src="https://github.com/user-attachments/assets/9af957e5-e12a-496f-b394-72411e6d3cf9" />

