# Sign Language Recognition System

## Overview

This project is a real-time Sign Language Recognition System built using Python, OpenCV, MediaPipe, TensorFlow, and Keras. The application captures hand gestures through a webcam, extracts hand landmark features, and predicts the corresponding sign using a trained machine learning model.

The goal of the project is to demonstrate how computer vision and machine learning can be used to recognize hand gestures and assist in sign language communication.

## Features

* Real-time hand detection using MediaPipe
* Hand landmark extraction
* Gesture classification using a trained neural network model
* Live prediction through webcam feed
* Easy to extend with additional gestures or sign vocabulary

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow
* Keras
* NumPy
* Scikit-Learn

## Project Structure

```text
SignLanguageRecognition/
│
├── data_collection.py
├── train.py
├── predict.py
├── function.py
├── model.h5
├── MP_Data/
└── README.md
```

## Workflow

1. Collect gesture samples using the webcam.
2. Extract hand landmark coordinates using MediaPipe.
3. Store the extracted keypoints as training data.
4. Train a neural network model on the collected dataset.
5. Load the trained model for real-time prediction.
6. Display the predicted gesture on the screen.

## Installation

Clone the repository:

```bash
git clone https://github.com/gk2001-cra/Sign-Language-Recognition-System.git
```

Move into the project directory:

```bash
cd Sign-Language-Recognition-System
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

### Collect Training Data

```bash
python data_collection.py
```

### Train the Model

```bash
python train.py
```

### Run Real-Time Prediction

```bash
python predict.py
```

## Future Improvements

* Support for complete words and sentences
* Improved model accuracy with larger datasets
* Integration with speech synthesis
* Support for dynamic sign recognition

## Author

Gaurav Kumbhar

Electronics and Telecommunication Engineering Graduate with an interest in Machine Learning, Computer Vision, and Software Development.
