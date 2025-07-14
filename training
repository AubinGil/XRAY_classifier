# prompt: In this notebook i implemented a inception v3 can you implement a VIT transformer for multiclass classification so we can compare

!pip install transformers torch torchvision torchaudio pandas scikit-learn

import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
# Adjust based on your data
ALL_LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# --- Data Preparation ---

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, label_binarizer=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.label_binarizer = label_binarizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        # Correctly extract labels using ALL_LABELS
        labels = self.dataframe.iloc[idx][ALL_LABELS].values.astype(float)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)

# Load labels
train_df = pd.read_csv(os.path.join(TRAIN_LABELS_DIR, 'train_labels.csv'))
test_df = pd.read_csv(os.path.join(TEST_LABELS_DIR, 'test_labels.csv'))

# Binarize labels
mlb = MultiLabelBinarizer()
mlb.fit([ALL_LABELS]) # Fit on all possible labels to ensure consistent columns

# Split training data for validation
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Define transforms
transform = Compose([
    Resize(IMAGE_SIZE),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
train_dataset = ChestXrayDataset(train_df, IMAGES_DIR, transform=transform) # Removed label_binarizer
val_dataset = ChestXrayDataset(val_df, IMAGES_DIR, transform=transform) # Removed label_binarizer
test_dataset = ChestXrayDataset(test_df, IMAGES_DIR, transform=transform) # Removed label_binarizer

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# --- Model Definition ---

# Load pre-trained ViT model
# We will replace the classifier head for multi-label classification
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

# Modify the classifier head for multi-label classification
# The number of output classes should match the number of unique labels
num_labels = len(ALL_LABELS) # Use len(ALL_LABELS) instead of mlb.classes_
model.classifier = nn.Linear(model.config.hidden_size, num_labels)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss for multi-label classification
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---

print("Starting training...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images).logits # Get logits for multi-label
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {epoch_loss:.4f}")

    # --- Validation Loop ---
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)

    val_loss = running_val_loss / len(val_dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_loss:.4f}")

print("Training finished.")

# --- Evaluation ---
print("Evaluating on test set...")
model.eval()
running_test_loss = 0.0
predictions = []
true_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images).logits
        loss = criterion(outputs, labels)
        running_test_loss += loss.item() * images.size(0)

        # For multi-label, we typically threshold the sigmoid output
        # Here we get the raw logits, apply sigmoid later if needed for metrics
        predictions.extend(outputs.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

test_loss = running_test_loss / len(test_dataset)
print(f"Test Loss: {test_loss:.4f}")

# You would typically calculate metrics like AUC-ROC, F1-score for multi-label
# using the `predictions` and `true_labels` arrays.
# Example (requires sklearn):
from sklearn.metrics import roc_auc_score, f1_score

# Apply sigmoid to get probabilities for AUC/F1 (assuming outputs are logits)
import numpy as np
sigmoid_predictions = 1 / (1 + np.exp(-np.array(predictions)))
true_labels = np.array(true_labels) # Convert true_labels to a numpy array


# Calculate AUC for each label (One vs Rest)
# Handle cases where a label might be missing in the test set true_labels
aucs = []
for i in range(true_labels.shape[1]):
    try:
        auc = roc_auc_score(true_labels[:, i], sigmoid_predictions[:, i])
        aucs.append(auc)
    except ValueError:
        # Handle cases where a class might have only one sample in the test set
        # print(f"Could not calculate AUC for label {mlb.classes_[i]}")
        pass # Or handle as needed

mean_auc = np.mean(aucs) if aucs else 0
print(f"Mean AUC-ROC: {mean_auc:.4f}")

# For F1, you typically need to choose a threshold. A common approach is 0.5
# binary_predictions = (sigmoid_predictions > 0.5).astype(int)
# micro_f1 = f1_score(true_labels, binary_predictions, average='micro')
# macro_f1 = f1_score(true_labels, binary_predictions, average='macro')
# print(f"Micro F1 Score: {micro_f1:.4f}")
# print(f"Macro F1 Score: {macro_f1:.4f}")
