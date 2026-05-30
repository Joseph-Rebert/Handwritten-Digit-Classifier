# Handwritten Digit Classifier

A PyTorch-based machine learning project that classifies handwritten digits from images. The model is trained on the MNIST dataset and can be used to predict digits from custom uploaded images.

## Project Overview

This project uses a neural network built with PyTorch to recognize handwritten digits from 0 to 9. It trains on the MNIST dataset, a standard dataset of handwritten number images, and then evaluates how well the model can classify unseen digits.

I built this project to practice core machine learning concepts, including data loading, neural networks, training loops, model evaluation, and image preprocessing.

## Features

* Trains a neural network on the MNIST handwritten digit dataset
* Uses PyTorch and TorchVision for model building and data handling
* Evaluates model accuracy on test data
* Supports predicting custom handwritten digit images
* Includes image preprocessing for uploaded files
* Demonstrates a basic computer vision classification workflow

## Tech Stack

**Language**

* Python

**Machine Learning**

* PyTorch
* TorchVision
* Neural Networks
* MNIST Dataset

**Image Processing**

* Pillow

**Version Control**

* Git
* GitHub

## How It Works

The model takes in 28x28 grayscale images of handwritten digits and predicts which number the image represents. The image is flattened, passed through a simple neural network, and classified into one of ten possible outputs: digits 0 through 9.

The general workflow is:

1. Load the MNIST training and testing datasets
2. Preprocess image data using TorchVision transforms
3. Build a neural network with PyTorch
4. Train the model on labeled handwritten digit images
5. Test the model on unseen data
6. Use the trained model to predict custom handwritten digit images

## What I Learned

* How to build a neural network using PyTorch
* How training and testing datasets work
* How to use the MNIST dataset for image classification
* How to write a basic training loop
* How to evaluate model accuracy
* How to preprocess custom images for prediction
* How computer vision models classify image data

## Installation

Clone the repository:

```bash
git clone https://github.com/Joseph-Rebert/Handwritten-Number-ML-Model.git
cd Handwritten-Number-ML-Model
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision pillow
```

## Usage

Run the main Python file:

```bash
python main.py
```

If your project uses a different file name, replace `main.py` with the correct script name.

## Status

This project is currently complete.

## Author

**Joseph Rebert**

* GitHub: [Joseph Rebert](https://github.com/Joseph-Rebert)
* LinkedIn: [Joseph Rebert](https://www.linkedin.com/in/joseph-rebert-9243192b3/)
* Portfolio: [Portfolio Website](https://joseph-rebert.github.io/Portfolio-Website/)
