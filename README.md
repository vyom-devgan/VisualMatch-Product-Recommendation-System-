# VisualMatch: Product Recommendation System Using Content-Based Image Retrieval (CBIR)

**VisualMatch** is an advanced product recommendation system designed to retrieve and recommend products based on their visual features. Using **Content-Based Image Retrieval (CBIR)** techniques, this system leverages deep learning models to extract feature vectors from product images and compute similarity using cosine similarity, resulting in highly relevant recommendations.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
  - [Step 1: Dataset Preparation](#step-1-dataset-preparation)
  - [Step 2: Feature Extraction](#step-2-feature-extraction)
  - [Step 3: Similarity Measurement](#step-3-similarity-measurement)
  - [Step 4: Visualization](#step-4-visualization)
- [Results](#results)

## Introduction
VisualMatch allows users to find visually similar products from a collection of images. This can be useful in various industries, such as e-commerce, where recommending products that visually resemble the one a customer is viewing can enhance the shopping experience.

The core of the system is a **ResNet18** deep learning model that has been pre-trained on ImageNet for feature extraction. By comparing feature vectors using **cosine similarity**, the system recommends the top visually similar products to the user.

## Features
- **Deep Learning-based Feature Extraction:** Uses a pre-trained ResNet18 model to extract robust image features.
- **Cosine Similarity Calculation:** Efficient similarity metric calculation for finding visually similar products.
- **Top K Recommendations:** Returns the top K most similar products for any given query.
- **Visualization of Results:** Displays query images alongside their recommended matches.
- **Modular Design:** Clean, modular code that is easy to extend or modify.

## Technologies Used
- **Python 3.x**
- **PyTorch** for deep learning model loading and inference
- **Torchvision** for image transformations and pre-trained models
- **PIL (Python Imaging Library)** for image loading and manipulation
- **NumPy** for numerical operations
- **scikit-learn** for cosine similarity computation
- **Matplotlib** for visualization

## Dataset
The dataset used for this project was obtained from Kaggle. You can find it [here](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data).

## Model Architecture

The core model is a pre-trained **ResNet18** from the `torchvision` library. The final classification layer is removed, and the rest of the network is used as a feature extractor. Feature vectors are then compared using **cosine similarity** to recommend visually similar products.

## Usage
Follow these steps to utilize the VisualMatch product recommendation system:

### Step 1: Dataset Preparation
Load and preprocess the images from the dataset. Ensure that images are resized and normalized appropriately for the model.

### Step 2: Feature Extraction
Utilize the pre-trained ResNet18 model to extract feature vectors from the images. This step transforms each image into a numerical representation.

### Step 3: Similarity Measurement
Compute cosine similarity between the feature vectors to find images visually similar to a query image. The top K similar images are recommended.

### Step 4: Visualization
Visualize the query image alongside its recommended similar images for easy comparison.

## Results

The VisualMatch system effectively retrieves visually similar products based on the input image. By leveraging the ResNet18 model and cosine similarity, the system demonstrates high accuracy in recommending relevant products.

### Sample Results:
- **Query Image:** Product X
- **Recommended Products:**
    - Product A (Similarity: 0.92)
    - Product B (Similarity: 0.89)
    - Product C (Similarity: 0.87)
