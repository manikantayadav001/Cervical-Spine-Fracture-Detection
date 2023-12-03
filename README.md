# Cervical-Spine-Fracture-Detection

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)  ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Overview
This repository contains the implementation of a Convolutional Neural Network (CNN) based on the InceptionResNetV2 architecture for detecting cervical spine fractures in medical images, particularly CT scans.

## Getting Started
#### Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

#### Dataset
The model is trained on a diverse dataset of cervical spine CT scans from Kaggle's [RSNA 2022 Cervical Spine Fracture](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/data) dataset. It is a multilabel classification problem, where we'll have to detect the probability of fracture in C1-C7 vertebrae.

## Model Architecture
### 1. Custom CNN
The custom CNN architecture is designed from scratch, incorporating Convolutional, Pooling, and Dropout layers. This approach allows us to tailor the network to the specific characteristics of cervical spine fracture detection.

### 2. Transfer Learning Models
This section explores the utilization of pre-trained deep learning models for improved performance. Models such as DenseNet121, InceptionV3, and InceptionResNetV2 are fine-tuned on our dataset, leveraging the knowledge gained from diverse image datasets.

### 3. Encoder-Decoder Architecture
In this approach, an encoder-decoder architecture is implemented. The encoder employs the U-Net model, known for its effectiveness in segmentation tasks. The best-performing transfer learning model is utilized as the decoder to enhance feature extraction and improve the precision of cervical spine fracture identification. ​

## Future Scope
1. The dataset contains additional data such as bounding boxes which will be useful in building more robust models, we can utilize that for to futher increase the performance of the models.
2. Currently, we are considering only the images whose segmentation data is available due to the hardware restriction. It can be further extended to include images whose segmentation data is not available.
