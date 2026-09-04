"""
STEP 1: Face Detection & Preprocessing
Run this in Google Colab.
Detects faces with Haar Cascade, crops, resizes to 112x112, saves to processed_data/
"""

import os
import cv2
import shutil
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
RAW_DATA_DIR = "/content/raw_data"          # folder containing one subfolder per person (your uploaded dataset)
PROCESSED_DIR = "/content/drive/MyDrive/face_recognition_project/processed_data"
IMG_SIZE = 112
MIN_FACE_SIZE = (60, 60)   # ignore tiny false-positive detections

# Load OpenCV's pretrained Haar Cascade frontal face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(img_path):
    """
    Returns the largest detected face crop (BGR np array), resized to IMG_SIZE x IMG_SIZE.
    Returns None if no face is detected.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # improves detection under varied lighting

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE
    )

    if len(faces) == 0:
        return None

    # If multiple faces detected, keep the largest (most likely the main subject)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add a small margin around the detected box so we don't crop too tight
    margin = int(0.15 * w)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    face_crop = img[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
    return face_crop


def process_dataset():
    raw_root = Path(RAW_DATA_DIR)
    out_root = Path(PROCESSED_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    person_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])
    print(f"Found {len(person_dirs)} classes: {[d.name for d in person_dirs]}\n")

    stats = {}

    for person_dir in person_dirs:
        person_name = person_dir.name
        out_person_dir = out_root / person_name
        out_person_dir.mkdir(parents=True, exist_ok=True)

        img_files = [f for f in person_dir.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

        success, failed = 0, 0
        for i, img_path in enumerate(img_files):
            face = detect_and_crop_face(str(img_path))
            if face is not None:
                out_path = out_person_dir / f"{person_name}_{success:03d}.jpg"
                cv2.imwrite(str(out_path), face)
                success += 1
            else:
                failed += 1

        stats[person_name] = (success, failed)
        print(f"{person_name:20s} -> {success} faces detected, {failed} skipped (no face found)")

    print("\n=== Summary ===")
    total_success = sum(s for s, f in stats.values())
    total_failed = sum(f for s, f in stats.values())
    print(f"Total usable images: {total_success}")
    print(f"Total skipped: {total_failed}")

    # Flag classes with low counts after detection failures
    low_count_classes = [name for name, (s, f) in stats.items() if s < 50]
    if low_count_classes:
        print(f"\n⚠️  Classes with <50 usable images (consider adding more source images): {low_count_classes}")

    return stats


if __name__ == "__main__":
    process_dataset()
