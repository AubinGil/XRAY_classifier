# XRAY_classifier-
🩺 ChestX-ray Multi-label Classification with InceptionV3
This repository contains a deep learning pipeline for multi-label classification of chest X-ray images using a pretrained InceptionV3 model. The dataset is derived from the NIH ChestX-ray14 dataset, and the model is trained to identify multiple thoracic pathologies from radiographic images.
📂 Directory Structure
plaintext
chestxray_data/
├── images/                     # Contains all chest X-ray images
├── train_labels/train_labels.csv
└── test_labels/test_labels.csv
🚀 Features
Custom PyTorch Dataset class for flexible data handling
Image transformations with torchvision.transforms
Pretrained InceptionV3 backbone fine-tuned for multi-label output
Auxiliary loss support for improved training stability
Training and validation loop with real-time loss tracking
Model checkpointing after each epoch
🛠️ Dependencies
bash
pip install torch torchvision pandas scikit-learn tqdm pillow
🧠 Model Architecture
Base: InceptionV3 pretrained on ImageNet
Output: Modified fc and AuxLogits.fc layers to match number of labels
Loss: BCEWithLogitsLoss with weighted auxiliary output
Optimizer: Adam with learning rate 1e-4
🧪 Training
Run the training script to begin model fine-tuning:
bash
python train.py
During each epoch, logs will show training and validation losses, and a checkpoint will be saved as:
bash
/content/inceptionv3_epoch{n}.pth
📊 Data Loading and Preprocessing
All image files are loaded via PIL and converted to RGB
Transforms applied:
Resize to (299, 299)
Random horizontal flip
Normalization to ImageNet standards
📈 Evaluation
The model is evaluated on a validation split from the training data (10%). For a rigorous benchmark, integrate the test_labels.csv split.
📎 Label Configuration
all_labels are dynamically determined from CSV column names, excluding the Image Index.
✍️ Citation and Acknowledgement
This project adapts and fine-tunes NIH ChestX-ray14 and leverages pretrained models from PyTorch.


!<img width="580" height="490" alt="image" src="https://github.com/user-attachments/assets/9af957e5-e12a-496f-b394-72411e6d3cf9" />

