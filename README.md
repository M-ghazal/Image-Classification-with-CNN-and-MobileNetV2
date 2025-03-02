# Image Classification with CNN and MobileNetV2

## Overview

This project implements image classification using two different approaches:

- **Convolutional Neural Network (CNN)** built from scratch using TensorFlow and Keras.
- **Transfer Learning with MobileNetV2**, a pre-trained deep learning model, for improved performance.
- **Google Cloud Vision API** is used for image analysis and label detection.

## Features

- **Data Augmentation**: Applies transformations such as rotation, shifting, zooming, and flipping.
- **CNN Architecture**: Uses multiple convolutional layers with batch normalization and max pooling.
- **MobileNetV2**: Utilizes a pre-trained deep learning model for efficient feature extraction.
- **Regularization**: Includes L2 regularization and dropout to prevent overfitting.
- **Early Stopping & Learning Rate Reduction**: Monitors validation loss to optimize training.
- **Evaluation Metrics**: Calculates accuracy, precision, recall, F1-score, and Intersection over Union (IoU).
- **Performance Visualization**: Plots training history and confusion matrix.
- **Google Cloud Vision API Integration**: Analyzes images and extracts labels.

## Hardware Requirements

- CPU or GPU (Recommended for faster training)
- Minimum 8GB RAM
- Sufficient storage for dataset and model

## Software Requirements

- Python 3
- TensorFlow & Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV (Optional for dataset preprocessing)
- Google Cloud Vision API (Requires authentication credentials)

## Dataset

- **Training Dataset**: Located in `Train/` directory.
- **Testing Dataset**: Located in `Test/` directory.

Images are resized to `128x128` for CNN and `224x224` for MobileNetV2 before being fed into the model.

## Model Architectures

### CNN Model

- **Convolutional Layers**: Extract spatial features from images.
- **Batch Normalization**: Normalizes activations for stable training.
- **Max Pooling Layers**: Reduces spatial dimensions.
- **Fully Connected Layers**: Processes features for classification.
- **Dropout**: Prevents overfitting by randomly dropping units.
- **Output Layer**: Uses a sigmoid activation for binary classification.

### MobileNetV2 Transfer Learning

- **Pre-trained MobileNetV2**: Used as a feature extractor.
- **Global Average Pooling Layer**: Reduces the feature map.
- **Dense Layers**: Fully connected layers for classification.
- **Dropout**: Helps prevent overfitting.
- **Output Layer**: Uses sigmoid activation for binary classification.

## Training and Evaluation

- **Optimizer**: Adam optimizer with a learning rate of 0.0001.
- **Loss Function**: Binary cross-entropy.
- **Batch Size**: 32 (CNN), 16 (MobileNetV2).
- **Epochs**: 100 (CNN), 10 (MobileNetV2) (Early stopping applied).
- **Validation**: Performed on the test dataset.

### Google Cloud Vision API Integration

- Analyzes test images for label detection.
- Extracts meaningful information from images using Google's AI.
- Results are printed with confidence scores.

## Results

- **Training and Validation Accuracy & Loss**: Displayed using plots.
- **Test Accuracy**: Evaluated on the test dataset.
- **Classification Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - IoU (Intersection over Union)
- **Confusion Matrix**: Displays classification performance.

## Usage

1. Place training images in `Train/` directory and testing images in `Test/` directory.
2. Ensure Google Cloud Vision API credentials are set up.
3. Run the script to train both models.

The trained models are saved as:
- `Model.h5` (CNN)
- `diabetic_kid_present_vs_not_present.h5` (MobileNetV2)

Training history is saved in `history.json`.

Evaluate the models using test images and review performance metrics.

(Optional) Run image analysis using the Google Cloud Vision API.

## Output

- **Model Files**: `Model.h5` (CNN) and `diabetic_kid_present_vs_not_present.h5` (MobileNetV2).
- **Training History**: `history.json` (Contains loss and accuracy logs).
- **Performance Metrics**: Printed and visualized with Matplotlib.
- **Google Cloud Vision API Labels**: Displayed for test images.

## Future Enhancements

- Implement multi-class classification.
- Further fine-tune MobileNetV2.
- Deploy model for real-time inference using TensorFlow Serving or Flask API.

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/api/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
- [Google Cloud Vision API](https://cloud.google.com/vision/docs)
