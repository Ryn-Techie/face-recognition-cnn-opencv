"""
STEP 3: Inference - Detect & Recognize Faces in New Photos
Run this in Google Colab, after 02_train.py has produced models/best_model.pth

Upload a new photo -> detects face -> predicts who it is with confidence score
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "/content/drive/MyDrive/face_recognition_project/models/best_model.pth"
IMG_SIZE = 112
CONFIDENCE_THRESHOLD = 0.60   # below this -> "Unknown"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ----------------------------
# SAME CNN ARCHITECTURE AS TRAINING (must match exactly)
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
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
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
# LOAD MODEL
# ----------------------------
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    num_classes = checkpoint["num_classes"]
    idx_to_class = checkpoint["idx_to_class"]

    model = FaceCNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded. Classes ({num_classes}): {list(idx_to_class.values())}")
    return model, idx_to_class


infer_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def detect_faces(img_bgr):
    """Returns list of (x, y, w, h) boxes for ALL detected faces (not just the largest)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces


def predict_face(model, idx_to_class, face_crop_bgr):
    """Takes a cropped face (BGR np array), returns (predicted_name, confidence)."""
    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb).resize((IMG_SIZE, IMG_SIZE))

    tensor = infer_transform(face_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_idx = probs.max(1)

    confidence = confidence.item()
    pred_name = idx_to_class[pred_idx.item()]

    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence
    return pred_name, confidence


def run_inference_on_upload():
    model, idx_to_class = load_model()

    print("\nPlease upload a photo:")
    uploaded = files.upload()
    img_path = list(uploaded.keys())[0]

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Could not read image.")
        return

    faces = detect_faces(img_bgr)
    if len(faces) == 0:
        print("No face detected in this image.")
        return

    print(f"\nDetected {len(faces)} face(s):\n")

    img_display = img_bgr.copy()
    for (x, y, w, h) in faces:
        margin = int(0.15 * w)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(img_bgr.shape[1], x + w + margin), min(img_bgr.shape[0], y + h + margin)
        face_crop = img_bgr[y1:y2, x1:x2]

        name, conf = predict_face(model, idx_to_class, face_crop)
        label = f"{name} ({conf*100:.1f}%)"
        print(f"Face detected -> {label}")

        # Draw box + label on the image for visual confirmation
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_display, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show result
    img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_display_rgb)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    run_inference_on_upload()
