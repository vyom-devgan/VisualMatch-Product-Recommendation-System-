# VisualMatch: Product Recommendation System Using Content-Based Image Retrieval (CBIR)

**VisualMatch** is an advanced product recommendation system designed to retrieve and recommend products based on their visual features. Using **Content-Based Image Retrieval (CBIR)** techniques, this system leverages deep learning models to extract feature vectors from product images and compute similarity using cosine similarity, resulting in highly relevant recommendations.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
  - [Step 1: Dataset Preparation](#step-1-dataset-preparation)
  - [Step 2: Feature Extraction](#step-2-feature-extraction)
  - [Step 3: Similarity Measurement](#step-3-similarity-measurement)
  - [Step 4: Visualization](#step-4-visualization)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

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

The dataset used for this project was obtained from Kaggle. You can find it [here](insert_link_to_kaggle_dataset).
