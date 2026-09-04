"""
DIAGNOSTIC: Confusion Matrix
Run this after training, to see exactly which people get confused with each other.
Helps pinpoint if 'danush -> vijay' is a real similarity issue or a data problem.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

MODEL_PATH = "/content/drive/MyDrive/face_recognition_project/models/best_model.pth"
PROCESSED_DIR = "/content/drive/MyDrive/face_recognition_project/processed_data"
IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            conv_block(3, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256), conv_block(256, 512),
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
        return self.classifier(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def run_confusion_analysis():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    idx_to_class = checkpoint["idx_to_class"]
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]

    model = FaceCNN(num_classes).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []

    # Run every image in processed_data through the model (full pass, not just val split)
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = Path(PROCESSED_DIR) / cls_name
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(tensor)
                pred = output.argmax(1).item()
            all_preds.append(pred)
            all_labels.append(cls_idx)

    class_names = [idx_to_class[i] for i in range(num_classes)]
    cm = confusion_matrix(all_labels, all_preds)

    # Print classification report (per-class precision/recall/f1)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    # Find the biggest confusion pairs (excluding the diagonal)
    print("\n=== Top confused pairs (true -> predicted) ===")
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)
    for count, true_name, pred_name in confusions[:15]:
        print(f"{true_name:20s} -> predicted as {pred_name:20s}  ({count} times)")

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(num_classes), class_names, rotation=90)
    plt.yticks(range(num_classes), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (full dataset)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/face_recognition_project/models/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    run_confusion_analysis()
