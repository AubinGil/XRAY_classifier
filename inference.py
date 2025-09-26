import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# 1) List your checkpoint files (assumes names like inceptionv3_epoch1.pth, …)
ckpts = sorted(glob.glob('/content/inceptionv3_epoch*.pth'),
               key=lambda fn: int(fn.split('epoch')[-1].split('.pth')[0]))

# 2) Function to compute average BCE loss on a loader
def compute_loss(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs)
            # if your model returns (preds, aux), take preds:
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, targets)
            total += loss.item()
            count += 1
    return total / count

# 3) Loop through checkpoints and record losses
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss()

train_losses, val_losses = [], []
for ckpt in ckpts:
    print("Evaluating", ckpt)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)             # load weights
    train_losses.append(compute_loss(model, train_loader, criterion, device))
    val_losses.append(  compute_loss(model, val_loader,   criterion, device))

# 4) Plot loss curves
epochs = np.arange(1, len(ckpts)+1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses,   label="Val   Loss")
plt.xlabel("Epoch"); plt.ylabel("BCE Loss")
plt.title("Train & Val Loss (no retraining)")
plt.legend(); plt.tight_layout(); plt.show()

# 5) For final model only: get raw probabilities on val set for ROC/PR
model.load_state_dict(torch.load(ckpts[-1], map_location=device))
model.eval()

all_targets = []
all_probs   = []
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        if isinstance(out, tuple): out = out[0]
        probs = torch.sigmoid(out).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.numpy())

all_probs   = np.vstack(all_probs)
all_targets = np.vstack(all_targets)

# 6) Compute micro-average ROC/PR
fpr, tpr, roc_auc = {}, {}, {}
prec, rec, ap    = {}, {}, {}
L = all_targets.shape[1]

for i in range(L):
    fpr[i], tpr[i], _    = roc_curve(all_targets[:,i], all_probs[:,i])
    roc_auc[i]           = auc(fpr[i], tpr[i])
    prec[i], rec[i], _   = precision_recall_curve(all_targets[:,i], all_probs[:,i])
    ap[i]                = average_precision_score(all_targets[:,i], all_probs[:,i])

fpr["micro"], tpr["micro"], _ = roc_curve(all_targets.ravel(), all_probs.ravel())
roc_auc["micro"]               = auc(fpr["micro"], tpr["micro"])
prec["micro"], rec["micro"], _ = precision_recall_curve(all_targets.ravel(), all_probs.ravel())
ap["micro"]                    = average_precision_score(all_targets, all_probs, average="micro")

# 7) Plot multi-label ROC
plt.figure(figsize=(6,5))
plt.plot(fpr["micro"], tpr["micro"],
         label=f"micro‐avg ROC (AUC = {roc_auc['micro']:.2f})", linewidth=2)
for i,label in enumerate(all_labels):
    plt.plot(fpr[i], tpr[i], lw=1, label=f"{label} (AUC={roc_auc[i]:.2f})")
plt.plot([0,1],[0,1],'k--', lw=0.5)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Multi-Label ROC Curves")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small")
plt.tight_layout(); plt.show()

# 8) Plot multi-label Precision–Recall
plt.figure(figsize=(6,5))
plt.plot(rec["micro"], prec["micro"],
         label=f"micro‐avg PR (AP = {ap['micro']:.2f})", linewidth=2)
for i,label in enumerate(all_labels):
    plt.plot(rec[i], prec[i], lw=1, label=f"{label} (AP={ap[i]:.2f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Multi-Label Precision-Recall Curves")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small")
plt.tight_layout(); plt.show()
