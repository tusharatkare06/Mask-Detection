# Mask Detection on Jetson Nano 2GB using Yolov5.

## Aim And Objectives

## Aim

#### To create a Helmet detection system which will detect Human face and then check if Mask is worn or not.

## Objectives

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To show on the optical viewfinder of the camera module whether a person is wearing a Mask or not.

## Abstract

• A person’s face is classified whether a Mask is worn or not and is detected by the live feed from the system’s camera.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

• A Mask is by far the most important and effective piece of protective equipment a person can wear.


## Introduction

• This project is based on a Mask detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.
    
• This project can also be used to gather information about who is wearing a Mask and who is not.
    
• Neural networks and machine learning have been used for these tasks and have obtained good results.
    
• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Mask detection as well.

## Literature Review

• Wearing a Mask helps to reduce the impact of an viral infections on your health.
    
• This layer of safety blocks the cool breeze thus helps you to stay healthy & prevent you from getting sick in the cold.

## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.
    
• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.
    
• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

## Jetson Nano 2GB







https://user-images.githubusercontent.com/89011801/151480525-4615fe18-cbf2-4dd2-a954-2efbe6627558.mp4








## Proposed System

1] Study basics of machine learning and image recognition.

2]Start with implementation
    
    • Front-end development
    • Back-end development

3] Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether a person is wearing a Mask or not.

4] use datasets to interpret the object and suggest whether the person on the camera’s viewfinder is wearing a Mask or not.

## Methodology

#### The Mask detection system is a program that focuses on implementing real time Mask detection.
#### It is a prototype of a new product that comprises of the main module:
#### Mask detection and then showing on viewfinder whether the person is wearing a Mask or not.
#### Mask Detection Module

### This Module is divided into two parts:


#### 1] face  detection


• Ability to detect the location of a person’s face in any input image or frame. The output is the bounding box coordinates on the detected face of a person.
   
• For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.
    
• This Datasets identifies person’s face in a Bitmap graphic object and returns the bounding box image with annotation of Mask or no Mask present in each image.

#### 2] Mask Detection


• Recognition of the face and whether Mask is worn or not.
    
• Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.
    
• YOLOv5 was used to train and test our model for whether the Mask was worn or not. We trained it for 149 epochs and achieved an accuracy of approximately 92%. 



## Installation

#### Initial Configuration

```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*

```
#### Create Swap 
```bash
udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
# make entry in fstab file
/swapfile1	swap	swap	defaults	0 0
```
#### Cuda env in bashrc
```bash
vim ~/.bashrc

# add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

```
#### Update & Upgrade
```bash
sudo apt-get update
sudo apt-get upgrade
```
#### Install some required Packages
```bash
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev
```
#### Install Torch
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
#### Install Torchvision
```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
#### Clone Yolov5 
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
```
#### Download weights and Test Yolov5 Installation on USB webcam
```bash
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt  --source 0
```
## Mask Dataset Training

### We used Google Colab And Roboflow

#### train your model on colab and download the weights and pass them into yolov5 folder.


## Running Helmet Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```
## Demo




https://github.com/tusharatkare06/Mask-Detection/assets/151806937/c3375455-5491-4f1e-a7ef-bc63c1a01426


https://youtu.be/DEX8YBgBjO0





## Application

• Detects a person’s face and then checks whether Mask is worn or not in each image frame or viewfinder using a camera module.
    
• Can be used anywhere where traffic lights are installed as their people usually stop on red lights and Mask detection becomes even more accurate.

• Can be used as a reference for other ai models based on Mask Detection.

## Future Scope

• As we know technology is marching towards automation, so this project is one of the step towards automation.
    
• Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.
    
• Mask detection will become a necessity in the future due to rise in population and hence our model will be of great help to tackle the situation in an efficient way.

## Conclusion

• In this project our model is trying to detect a person’s face and then showing it on viewfinder, live as to whether Mask is worn or not as we have specified in Roboflow.
    
• The model is efficient and highly accurate and hence reduces the workforce required.

## Reference

#### 1] Roboflow:- https://roboflow.com/

#### 3] Google images

## Articles :-

#### https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/when-and-how-to-use-masks
