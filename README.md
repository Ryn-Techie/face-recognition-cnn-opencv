# face-recognition-cnn-opencv
Face recognition system using OpenCV Haar Cascade and a custom CNN built with PyTorch.
# Face Recognition Using OpenCV and Custom CNN

A face recognition system built with **Python, OpenCV, and PyTorch** that detects faces, preprocesses them, trains a custom Convolutional Neural Network (CNN), and identifies known people from new images.

The project was developed as a machine learning/computer vision project using a celebrity face dataset.

## Project Overview

The system follows a four-step pipeline:

```text
Raw Face Images
       ↓
Face Detection & Preprocessing
       ↓
CNN Training
       ↓
Face Recognition / Inference
       ↓
Performance Analysis
```

The project uses **OpenCV's Haar Cascade classifier** for face detection and a **custom CNN trained from scratch using PyTorch** for face identification.

## Features

* Face detection using Haar Cascade
* Automatic face cropping
* Histogram equalization for improved face detection
* Face resizing to `112 × 112`
* Custom CNN trained from scratch
* Data augmentation
* Weighted sampling for imbalanced classes
* MixUp augmentation
* Label smoothing
* Adam optimizer
* Cosine annealing learning-rate scheduler
* Automatic `"Unknown"` prediction for low-confidence results
* Confusion matrix analysis
* Classification report with precision, recall, and F1-score
* GPU support when available

## Technologies Used

* **Python**
* **OpenCV**
* **PyTorch**
* **Torchvision**
* **NumPy**
* **Pillow**
* **Matplotlib**
* **Scikit-learn**
* **Google Colab**

## Project Structure

```text
face-recognition-cnn/
│
├── src/
│   ├── 01_preprocess.py
│   ├── 02_train.py
│   ├── 03_inference.py
│   └── 04_confusion_analysis.py
│
├── models/
│   ├── best_model.pth
│   ├── class_mapping.json
│   ├── training_curves.png
│   └── confusion_matrix.png
│
├── data/
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

> The raw dataset is not included in this repository. See the Dataset section for the expected directory structure.

## Dataset Structure

The preprocessing script expects one folder for each person:

```text
raw_data/
├── person_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── person_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── person_3/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Each folder name becomes a class label.

The preprocessing script searches for `.jpg`, `.jpeg`, and `.png` images.

## 1. Face Detection & Preprocessing

`01_preprocess.py` detects faces using OpenCV's pretrained Haar Cascade frontal-face detector.

For each image:

1. The image is loaded using OpenCV.
2. It is converted to grayscale for face detection.
3. Histogram equalization is applied.
4. Haar Cascade detects faces.
5. The largest detected face is selected.
6. A small margin is added around the detected face.
7. The face is cropped.
8. The crop is resized to `112 × 112`.
9. The processed image is saved into `processed_data/`.

Images where no face is detected are skipped.

## 2. CNN Training

`02_train.py` loads the processed face images and trains a custom CNN.

### CNN Architecture

The network contains five convolutional blocks:

```text
Input: 112 × 112 RGB image
        ↓
Conv Block: 3 → 32
        ↓
Conv Block: 32 → 64
        ↓
Conv Block: 64 → 128
        ↓
Conv Block: 128 → 256
        ↓
Conv Block: 256 → 512
        ↓
Adaptive Average Pooling
        ↓
128-dimensional embedding
        ↓
Dropout
        ↓
Classification Layer
```

Each convolutional block contains:

* `3 × 3` convolution
* Batch Normalization
* ReLU activation
* Max Pooling

The model is trained from scratch rather than using a pretrained CNN.

## Training Configuration

| Parameter        |            Value |
| ---------------- | ---------------: |
| Image size       |      `112 × 112` |
| Batch size       |             `32` |
| Epochs           |             `60` |
| Learning rate    |          `0.001` |
| Weight decay     |         `0.0001` |
| Label smoothing  |            `0.1` |
| Validation split |            `15%` |
| Optimizer        |             Adam |
| Scheduler        | Cosine Annealing |
| MixUp alpha      |            `0.2` |
| Dropout          |            `0.5` |

## Data Augmentation

The training pipeline applies several augmentations to improve generalization:

* Random horizontal flipping
* Color jitter
* Random rotation
* Random resized cropping
* Random erasing

MixUp is also applied to approximately half of the training batches.

## Handling Class Imbalance

The dataset contains different numbers of images for different people.

A `WeightedRandomSampler` is used during training so classes with fewer training images receive more balanced sampling.

## 3. Face Recognition / Inference

`03_inference.py` allows a new image to be uploaded and processed.

The system:

```text
Input Image
     ↓
Face Detection
     ↓
Face Cropping
     ↓
Image Preprocessing
     ↓
CNN Prediction
     ↓
Identity + Confidence
```

Multiple faces can be detected in a single image.

Each detected face receives a predicted identity and confidence score.

A confidence threshold of `0.60` is used:

```text
Confidence ≥ 60%  → Predicted identity
Confidence < 60%  → Unknown
```

Example:

```text
Face detected → person_name (87.4%)
Face detected → Unknown (42.1%)
```

## 4. Model Evaluation

`04_confusion_analysis.py` generates:

* Confusion matrix
* Classification report
* Per-class precision
* Per-class recall
* Per-class F1-score
* Most frequently confused class pairs

The confusion matrix helps identify which people the model has difficulty distinguishing.

## Results

The model achieved approximately **85% validation accuracy** during training.

> Validation accuracy is used as the primary reported performance metric. The confusion-analysis script currently evaluates the complete processed dataset, so its results should not be interpreted as an independent test-set accuracy.

## Model Output

After training, the following files are generated:

```text
models/
├── best_model.pth
├── class_mapping.json
└── training_curves.png
```

The trained checkpoint stores:

* Model parameters
* Class-to-index mapping
* Index-to-class mapping
* Number of classes

This allows the inference script to reconstruct the correct model and class labels.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

For Google Colab, the required packages can be installed in a notebook cell.

## Running the Project

### Step 1: Prepare the Dataset

Place the dataset in the expected structure:

```text
/content/raw_data/
```

with one folder per person.

### Step 2: Preprocess the Dataset

Run:

```bash
python src/01_preprocess.py
```

This creates the processed face dataset.

### Step 3: Train the CNN

Run:

```bash
python src/02_train.py
```

The best-performing model is saved as:

```text
models/best_model.pth
```

Training curves are also generated.

### Step 4: Run Face Recognition

Run:

```bash
python src/03_inference.py
```

Upload an image when prompted.

The program detects faces and displays the predicted identity and confidence score.

### Step 5: Analyze Model Performance

Run:

```bash
python src/04_confusion_analysis.py
```

This generates a confusion matrix and classification report.

## Limitations

This project has several limitations:

* The dataset is relatively small.
* The model is trained for the specific identities present in the dataset.
* Recognition performance can decrease with poor lighting, unusual poses, occlusion, or low-quality images.
* Haar Cascade may miss faces or produce false detections.
* The `"Unknown"` decision is based on softmax confidence, which should not be treated as a calibrated probability of identity.
* The current confusion-analysis script evaluates the full processed dataset rather than an independent test set.

## Future Improvements

Possible improvements include:

* Creating a separate test dataset
* Using a stratified train/validation/test split
* Evaluating on completely unseen images
* Improving the face detector
* Using a stronger face detection model
* Increasing the dataset size
* Using transfer learning
* Using face embeddings and similarity-based recognition
* Calibrating the unknown/rejection threshold
* Adding real-time webcam recognition
* Adding a proper evaluation pipeline that uses the same validation/test split

## Ethical & Dataset Notice

This project is intended for **educational and experimental purposes**.

The dataset contains images of public figures/celebrities. The original images may be subject to copyright, licensing, and other usage restrictions. They are therefore not included in this repository unless their redistribution rights permit it.

This project should not be used to make consequential decisions about real people.

## Author

**Sabitha G**

BSc Computer Science with Cognitive Systems

### Project Focus

Computer Vision • Machine Learning • Deep Learning • OpenCV • PyTorch

---

⭐ If you found this project useful, consider giving the repository a star.
