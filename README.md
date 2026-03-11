# Facial Emotion Recognition using CNN

## Overview

This project implements a **Facial Emotion Recognition (FER)** system using a **Convolutional Neural Network (CNN)** built with **PyTorch**.
The model is trained to classify human facial expressions into different emotional categories.

The trained model achieves **over 90% accuracy** on the test dataset.

The system can also perform **real-time emotion detection using a webcam**.

---

## Features

* CNN-based facial emotion classification
* PyTorch deep learning model
* Data augmentation for improved training
* Real-time emotion detection with webcam
* Trained model with **90%+ accuracy**

---

## Emotions Detected

The model predicts the following emotions:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

## Project Structure

```
EmotionRecognition/
│
├── camera.py
├── cnn.ipynb
├── fer_emotion_vgg.pth
└── README.md
```

### File Descriptions

**camera.py**
Runs real-time emotion detection using the trained model and webcam input.

**cnn.ipynb**
Jupyter Notebook used for training the convolutional neural network.

**fer_emotion_vgg.pth**
Saved PyTorch model weights after training.

---

## Model Architecture

The system uses a **Convolutional Neural Network (CNN)** trained on facial images.

Training includes:

* Image resizing to 48×48
* Grayscale conversion
* Data augmentation
* Normalization
* Cross-entropy loss optimization

Optimizer used:

* **AdamW**

Learning rate scheduling:

* **ReduceLROnPlateau**

---

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

Example dependencies:

```
torch
torchvision
opencv-python
numpy
```

---

## Usage

Run the webcam emotion detection script:

```
python camera.py
```

The program will open your webcam and display the predicted emotion in real time.

---

## Results

The trained CNN model achieves:

**Test Accuracy: 90%+**

This demonstrates the model's effectiveness at recognizing facial expressions from images.

---

## Future Improvements

Possible enhancements include:

* Training on larger emotion datasets
* Adding attention mechanisms
* Improving real-time detection performance
* Deploying the model as a web application

---

## Author

Albert
Computer Science Student interested in **AI, computer vision, and deep learning**.
