# Google_Inception
# Implementing Inception Models with Tensorflow-Keras

![alt text](https://th.bing.com/th/id/R.e3feaa84a47cd2c64707ee1c562a08c5?rik=phdWFg7HI4GnWA&pid=ImgRaw&r=0)



## Introduction

This repository contains an implementations of the Inception models developed by Google team 

## Model Architecture

* Inception V1 (GoogLeNet)
  
  The GoogLeNet Inception V1 model is composed of several inception modules. Each inception module applies convolutional operations with different kernel sizes      (1x1, 3x3, 5x5) and a 3x3 max pooling operation in parallel, concatenating their results. This allows the model to capture patterns at different scales. The     
  model also includes two auxiliary classifiers (output_1 and output_2) used during training to inject additional gradient at lower layers.

## Dependencies

This project requires the following libraries:
- Keras
- TensorFlow
- OpenCV
- NumPy

## Dataset


* Inception V1 (GoogLeNet)
  
    The model is trained and tested on the CIFAR10 dataset, which is directly loaded from Keras datasets. The images in the dataset are resized to 224x224 pixels 
    to match the input shape of the GoogLeNet Inception V1 model.

## Training


* Inception V1 (GoogLeNet)
  
    The model is compiled with the SGD optimizer and the categorical cross-entropy loss function. The learning rate is initially set to 0.01 and decays every 8 
    epochs. The model is trained for 1 epoch with a batch size of 256.

## Usage


* Inception V1 (GoogLeNet)
  
    1. Clone this repository.
    2. Run the Python script.

## References


- Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
