# Traffic Sign Data Pipeline & Detection using YOLOv8

## Overview

This project focuses on building a traffic sign detection pipeline using both traditional computer vision techniques and deep learning-based object detection models. The main goal is to detect and classify traffic signs from images and videos, while also organizing the dataset, preprocessing data, training models, and evaluating model performance.

The project was developed as part of a Computer Vision coursework project. It compares classical computer vision methods with modern object detection models based on YOLOv8.

## Objectives

The main objectives of this project are:

* Build a complete traffic sign detection workflow.
* Prepare and organize image datasets and annotation files.
* Apply classical image processing techniques for traffic sign detection.
* Train YOLOv8 models for object detection.
* Compare lightweight and larger YOLOv8 models.
* Evaluate model performance using standard object detection metrics.
* Visualize detection results on images and videos.
* Analyze the advantages and limitations of each approach.

## Project Structure

```text
traffic-sign-data-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── TrafficSignDetection.ipynb
│
└── reports/
    ├── report.pdf
    └── presentation.pdf
```

## Dataset

The dataset contains traffic sign images and corresponding annotation files. The dataset is prepared in YOLO format for object detection training.

Expected dataset structure:

```text
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each image has a corresponding label file in YOLO format:

```text
class_id x_center y_center width height
```

Where:

* `class_id`: traffic sign class index
* `x_center`: normalized x-coordinate of bounding box center
* `y_center`: normalized y-coordinate of bounding box center
* `width`: normalized bounding box width
* `height`: normalized bounding box height

The full dataset is not included in this repository due to file size limitations.

## Main Workflow

The project pipeline includes the following steps:

```text
Raw Images
    ↓
Dataset Organization
    ↓
Image Preprocessing
    ↓
Classical CV Detection / YOLOv8 Training
    ↓
Model Evaluation
    ↓
Image & Video Inference
    ↓
Result Visualization
```

## Methods

### 1. Classical Computer Vision Approach

The classical approach uses traditional image processing and feature extraction methods.

Main techniques:

* HSV color space conversion
* Color thresholding
* Image filtering
* ORB feature extraction
* Feature matching
* Bounding box visualization

This approach is useful for understanding how handcrafted features can be used for object detection. However, it is limited when dealing with complex backgrounds, lighting changes, occlusion, and different traffic sign shapes.

### 2. Deep Learning Approach

The deep learning approach uses YOLOv8 models for traffic sign detection.

Models used:

* YOLOv8n
* YOLOv8s

YOLOv8 is a one-stage object detector, meaning it predicts object locations and classes in a single forward pass. This makes it suitable for real-time object detection tasks.

### YOLOv8n

YOLOv8n is the nano version of YOLOv8. It is lightweight and fast, suitable for devices with limited computational resources.

Advantages:

* Fast inference speed
* Small model size
* Suitable for real-time detection
* Lower hardware requirements

Limitations:

* Lower detection accuracy compared to larger YOLOv8 variants
* May miss small or difficult traffic signs

### YOLOv8s

YOLOv8s is a larger version compared to YOLOv8n. It has more parameters and stronger feature extraction capability.

Advantages:

* Better detection accuracy
* More stable performance on difficult samples
* Better at detecting small or partially visible signs

Limitations:

* Slower than YOLOv8n
* Requires more computational resources

## Evaluation Metrics

The models are evaluated using common object detection metrics:

### Precision

Precision measures how many predicted objects are actually correct.

```text
Precision = True Positives / (True Positives + False Positives)
```

High precision means the model produces fewer false detections.

### Recall

Recall measures how many actual objects are successfully detected.

```text
Recall = True Positives / (True Positives + False Negatives)
```

High recall means the model misses fewer traffic signs.

### mAP@0.5

mAP@0.5 measures mean Average Precision when the IoU threshold is 0.5. A prediction is considered correct if the overlap between the predicted bounding box and the ground truth box is at least 50%.

This metric is commonly used to evaluate object detection performance.

### mAP@0.5:0.95

mAP@0.5:0.95 is a stricter metric. It calculates mAP over multiple IoU thresholds from 0.5 to 0.95.

This metric gives a more complete evaluation of both classification accuracy and bounding box localization quality.

### Confusion Matrix

The confusion matrix is used to analyze which traffic sign classes are correctly classified and which classes are often confused with each other.

## Results

The deep learning-based YOLOv8 models achieved better performance than the classical computer vision approach.

General observations:

* YOLOv8n is faster and more lightweight.
* YOLOv8s provides better detection accuracy in some cases.
* Classical methods are easier to understand but less robust.
* YOLO-based models are more suitable for real-world traffic sign detection.
* Detection performance depends heavily on dataset quality, annotation quality, and class balance.

## Video Processing

The trained model can also be applied to video input.

The general video inference workflow is:

```text
Input Video
    ↓
Read Frame by Frame
    ↓
Run YOLOv8 Detection
    ↓
Draw Bounding Boxes and Class Labels
    ↓
Save or Display Output Video
```

For future improvement, object tracking algorithms such as ByteTrack can be integrated to improve detection stability across frames.

## Tech Stack

* Python
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Ultralytics YOLOv8
* Jupyter Notebook
* Google Colab

## Installation

Clone this repository:

```bash
git clone https://github.com/ntdat0512/traffic-sign-data-pipeline.git
cd traffic-sign-data-pipeline
```

Install required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Open the notebook:

```bash
jupyter notebook notebooks/TrafficSignDetection.ipynb
```

Or run it in Google Colab by uploading the notebook and preparing the dataset in the correct YOLO format.

## Expected Files

The repository should include:

```text
notebooks/
└── TrafficSignDetection.ipynb

reports/
├── report.pdf
└── presentation.pdf
```

The dataset and trained weights are not included due to file size limitations.

## Notes on Dataset and Model Weights

Large files should not be pushed directly to GitHub, including:

* Full image datasets
* Training runs
* Model weights such as `.pt`
* Large video files
* Generated output videos

Recommended alternatives:

* Google Drive
* Kaggle Dataset
* Roboflow
* GitHub Releases
* Hugging Face Hub

## What I Learned

Through this project, I practiced:

* Building a computer vision workflow from dataset preparation to model evaluation.
* Working with image datasets and YOLO-format annotations.
* Applying traditional image processing methods using OpenCV.
* Training and evaluating YOLOv8 object detection models.
* Comparing classical computer vision with deep learning-based detection.
* Understanding object detection metrics such as Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
* Preparing project documentation for GitHub and portfolio use.

## Limitations

Some limitations of the current project:

* Dataset size may not be large enough for all real-world traffic conditions.
* Model performance may decrease under poor lighting, motion blur, or occlusion.
* Some traffic sign classes may be underrepresented.
* Classical methods are sensitive to background noise and lighting changes.
* Video detection may still have unstable predictions between consecutive frames.

## Future Improvements

Possible improvements include:

* Expand the dataset with more diverse traffic conditions.
* Add data augmentation to improve generalization.
* Fine-tune YOLOv8 hyperparameters.
* Compare additional YOLO versions or other object detection models.
* Apply ByteTrack for traffic sign tracking in videos.
* Export the model to ONNX for deployment.
* Build a simple web demo using Streamlit or FastAPI.
* Add an automated data validation script for YOLO labels.

## Reports

* Full report: `reports/report.pdf`
* Presentation slides: `reports/presentation.pdf`

## Author

Nguyễn Tiến Đạt

Computer Science Student
Interested in Data Engineering, Machine Learning, and Computer Vision.

