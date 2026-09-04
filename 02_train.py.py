"""
STEP 2: Train the Face Recognition CNN
Run this in Google Colab, after 01_preprocess.py has created processed_data/

Loads cropped faces -> splits train/val -> trains custom CNN from scratch -> saves model + plots
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import json
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
PROCESSED_DIR = "/content/drive/MyDrive/face_recognition_project/processed_data"
MODEL_DIR = "/content/drive/MyDrive/face_recognition_project/models"
IMG_SIZE = 112
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
VAL_SPLIT = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")


# ----------------------------
# DATASET
# ----------------------------
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        classes = sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        for cls in classes:
            cls_dir = Path(root_dir) / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Training transforms: light augmentation since dataset is small (~70/class)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    transforms.RandomRotation(12),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ----------------------------
# CNN ARCHITECTURE
# ----------------------------
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(3, 32),      # 112 -> 56
            conv_block(32, 64),     # 56 -> 28
            conv_block(64, 128),    # 28 -> 14
            conv_block(128, 256),   # 14 -> 7
            conv_block(256, 512),   # 7 -> 3
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        emb = self.embedding(x)
        x = self.dropout(emb)
        out = self.classifier(x)
        return out


# ----------------------------
# MIXUP AUGMENTATION
# ----------------------------
def mixup_data(x, y, alpha=0.2):
    """Blends pairs of images and labels together - proven small-dataset accuracy booster."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ----------------------------
# TRAINING LOOP
# ----------------------------
def train():
    full_dataset = FaceDataset(PROCESSED_DIR, transform=None)
    num_classes = len(full_dataset.class_to_idx)
    print(f"Total images: {len(full_dataset)} | Classes: {num_classes}")

    # Split indices for train/val
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Apply different transforms by wrapping
    class TransformWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img_path, label = full_dataset.samples[self.subset.indices[idx]]
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label

    train_ds = TransformWrapper(train_subset, train_transform)
    val_ds = TransformWrapper(val_subset, val_transform)

    # Weighted sampler so classes with fewer images (e.g. 55) get sampled as
    # often as classes with more (e.g. 79) - keeps training balanced
    train_labels = [full_dataset.samples[i][1] for i in train_subset.indices]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = FaceCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    MIXUP_ALPHA = 0.2

    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            # Apply mixup on ~half the batches; train on clean data the rest of the time
            use_mixup = np.random.rand() < 0.5
            if use_mixup:
                mixed_imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
                outputs = model(mixed_imgs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validate ----
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": full_dataset.class_to_idx,
                "idx_to_class": full_dataset.idx_to_class,
                "num_classes": num_classes,
            }, os.path.join(MODEL_DIR, "best_model.pth"))

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {MODEL_DIR}/best_model.pth")

    # Save class mapping separately too (handy for inference script)
    with open(os.path.join(MODEL_DIR, "class_mapping.json"), "w") as f:
        json.dump(full_dataset.idx_to_class, f, indent=2)

    # ---- Plot curves ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
    plt.show()

    return model, history


if __name__ == "__main__":
    train()
