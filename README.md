# Steel Stock Heat Number Digitization System for Logistics Supply Chain Management

## Project Description

The current Logistics Steel Stock supply chain process is highly manual and time consuming. Steel stock tags labelled by the respective Suppliers are currently being manually identified and handwritten by workers before submission to the Logistics Administrative Team for verification and entry into SAP. The aim of this project is to help automate the Steel Stock supply chain process by automatically identifying Steel Stock tags thus reducing the overall no. of manhours required for the receiving and issuing process.

The deliverables are a trained CNN model with a Python script to automatically detect and recognize text based on input images then output it onto a .csv file which can then be used to do auto verification with Mill Certificates and RPA for data entry into SAP in future.

## Original System Design

The original system design was to classify and OCR text regions based on user interaction via a mobile application.

### Image Acquisition & Preprocessing

The image dataset is a collection of tags labelled on Steel Stock by various Suppliers manually photographed using various mobile phone cameras. The ROI is then labelled using [SuperAnnotate](https://superannotate.com/). The .json file generated is processed to batch crop ROI from the images and resized to 224px x 224px as input into the CNN models.

### Image Augmentation

Due to the lack of image data provided by the Sponsor, image augmentation was done in Keras to artificially expand the image dataset to improve the performance and ability of the model to generalize. As we are working with OCR of text data, image flipping was not used in augmentation.

### Convolutional Neural Networks

As this is primarily an image classification task, CNN was chosen as the preferred algorithm. A collection of 4 CNN models built can be found [here](https://github.com/dannylzj/Image_Classification_OCR/tree/main/notebooks).

## Change in System Requirements

A change in system requirements was initiated by the Sponsor as the original system design still required the user to select regions of interest before sending it into the model for inference. The Sponsor feels that this is not automated enough and a new employee not trained in the task would not know what to select.

This prompted a new approach which requires not just a basic classification model but a detection and recognition algorithm. 2 end-to-end deep learning models were considered for this application
1) YOLO
2) PaddleOCR

After consideration, PaddleOCR was chosen as the preferred solution as it was built for the purpose of text detection and recognition as opposed to YOLO which was built for object detection.

### LogisticsOCR

LogisticsOCR was built using PaddleOCR's deep learning models to specifically suit the the change in system requirements. More details on LogisticsOCR can be found [here](https://github.com/dannylzj/Image_Classification_OCR/tree/main/LogisticsOCR).
