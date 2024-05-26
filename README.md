# CIFAR-10 Image Classification with ResNet and Flask API

This README provides an overview of a Jupyter Notebook that demonstrates the process of training a ResNet model on the CIFAR-10 dataset using PyTorch and deploying the trained model using a Flask API for image classification.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Model Definition](#model-definition)
4. [Training the Model](#training-the-model)
5. [Creating the Flask API](#creating-the-flask-api)
6. [Testing the API](#testing-the-api)
7. [Results](#results)
8. [Possible Errors](#possible-errors)

## Introduction

This project involves training a ResNet-50 model on the CIFAR-10 dataset and deploying it using a Flask API. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. The ResNet-50 model is a deep residual network with 50 layers, which we modify to fit the CIFAR-10 classification task. The Flask API allows users to send POST requests with image files for predictions.

## Data Preparation

The CIFAR-10 dataset is downloaded and preprocessed using PyTorch's `torchvision` library. The preprocessing steps include:
- Converting images to PyTorch tensors.
- Normalizing the images with a mean and standard deviation of 0.5 for each channel.

Data loaders are created for both the training and test datasets to facilitate batching, shuffling, and parallel loading.

## Model Definition

The ResNet-50 model is loaded with pre-trained weights and modified to output 10 classes instead of the original 1000 classes. The model is then moved to the GPU if available.

## Training the Model

The model is trained using the following setup:
- Loss Function: CrossEntropyLoss
- Optimizer: AdamW with a learning rate of 0.001
- Mixed Precision Training: Enabled using `torch.cuda.amp.GradScaler` and `autocast`
- Early Stopping: Implemented with a patience of 10 epochs

The training process involves:
- Forward and backward passes with mixed precision.
- Updating model parameters using the optimizer.
- Calculating and printing training and test accuracies.
- Saving the best model based on test accuracy.

### Training Results

Example training results over 28 epochs (with early stopping):
- Initial Epochs: 
  - Epoch 1: Training Accuracy ~ 14.38%, Test Accuracy ~ 23.81%
  - Epoch 2: Training Accuracy ~ 27.93%, Test Accuracy ~ 37.16%
- Intermediate Epochs:
  - Epoch 10: Training Accuracy ~ 77.90%, Test Accuracy ~ 73.48%
- Final Epochs:
  - Epoch 20: Training Accuracy ~ 94.36%, Test Accuracy ~ 77.29%
  - Epoch 28: Training Accuracy ~ 97.70%, Test Accuracy ~ 77.43% (Early Stopping)

## Creating the Flask API

A Flask API is created to serve the trained model for image classification. The API includes:
- A function to transform input images.
- A function to get predictions from the model.
- An endpoint `/predict` to handle POST requests with image files.

The Flask app is saved to `flask_app.py` and started in the background.

## Testing the API

The API is tested by sending POST requests with sample images from the CIFAR-10 test set. The expected class and the API response are compared to evaluate the model's performance.

Example API responses:
- Expected class: 3, API Response: {'class_id': 6}
- Expected class: 8, API Response: {'class_id': 8}
- Expected class: 8, API Response: {'class_id': 1}
- Expected class: 0, API Response: {'class_id': 1}
- Expected class: 6, API Response: {'class_id': 6}
- Expected class: 6, API Response: {'class_id': 6}
- Expected class: 1, API Response: {'class_id': 6}
- Expected class: 6, API Response: {'class_id': 6}
- Expected class: 3, API Response: {'class_id': 6}
- Expected class: 1, API Response: {'class_id': 6}

## Results

The model achieved a maximum test accuracy of approximately 77.99% before early stopping. The training accuracy reached up to 97.70%. The results indicate that the model is well-trained but may have some biases or overfitting issues.

## Possible Errors

During testing, the model showed a tendency to predict class 6 more frequently. This could be due to:
- Imbalanced class distribution in the training data.
- Overfitting to certain classes.
- Model biases introduced during training.

Further investigation and potential adjustments to the training process or dataset may be required to address these issues.

## Download the Jupyter Notebook

You can download the Jupyter Notebook used in this project from the following link:

[Download CIFAR-10 Image Classification Notebook](./src/CIFAR_10_Image_Classification_with_ResNet_and_Flask_API.ipynb)

## Conclusion

This project demonstrates the process of training a ResNet model on the CIFAR-10 dataset and deploying it using a Flask API. The model achieved good accuracy on the test set, and the Flask API allows for easy image classification via HTTP requests. Future work could focus on improving model performance and addressing any biases or errors observed during testing.