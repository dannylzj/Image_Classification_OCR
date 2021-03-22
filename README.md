# Deep Learning Models

A collection of various deep learning architectures and models for TensorFlow in Jupyter Notebooks used while working on a 5 months internship project under the Graduate Certificate in Intelligent Reasoning Systems Programme with NUS Institute of Systems Science and Keppel FELS Ltd.

## Aim

The aim of these models was to accurately classify text regions of interest (ROI) on images taken from Steel Stock into 3 classes:
1) Heat Number
2) Dimension
3) Grade

The ROI will then be put through an OCR engine such as Tesseract to do text recognition. The classification and recognized text will be tabulated onto a csv file.

## Image Dataset

The image dataset was a collection of tags manually photographed. The ROI is then labelled using [SuperAnnotate](https://superannotate.com/). The .json file generated is processed to batch crop ROI from the images and resized to 224px x 224px as input into the CNN models.

A copy of the processed images into train and test sets can be downloaded from the following [link](https://drive.google.com/file/d/1D5pIKTIhzl5UDrNeoXvYQlfWWsFzGDs-/view?usp=sharing).

## Convolutional Neural Networks

1) Convolutional Neural Network VGG-16
2) Convolutional Neural Network ResNet152v2
3) Convolutional Neural Network InceptionResNetv2
4) Convolutional Neural Network Xception

