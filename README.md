# DeepFake_Detection
## Table of Contents:
- What is DeepFake?
- Demo of the Project
- Impact of DeepFake Videos
- Project Objectives
- Project Pipeline
  - Pre-processing WorkFlow
  - Prediction WorkFlow
- Models Usage and their Architecture
- Deploy
  - Code Running Commands
- Technologies Used
- Conclusion
- Team


## What is DeepFake?
- DeepFakes are images or videos which have been altered to feature the face of
someone else, like an advanced form of Face Swapping, using an AI DeepFake
Converter.
- Many Deep Fakes are done by superimposing or combining existing images into source
images and videos using Generative Adversarial Networks (GAN) and these networks
are developing better every day

## Demo of the Project
Link : https://www.youtube.com/watch?v=wy8mVnBZ6pY&ab_channel=BalajiKartheek

## Impact of DeepFake Videos
- DeepFakes can be used to create fake news, celebrity unusual videos, politician
content videos, and financial fraud.
- False Rumours can be spread using DeepFake videos which causes unrest and
mental anxiety among people.
- Many fields in Film Industry, content providers, and social media platforms are
fighting against DeepFake.
 
 # Project Objectives:
 
Identification of deepfakes is necessary to prevent the use of malicious AI.
We intend to,
-  Build a model that processes the given video and classifies it as REAL or FAKE.
-  Dploy a feature in the social media apps that can detect and give a warning to
the content provider who is willing to do viral by uploading deepFaked images or
videos.

![image](https://user-images.githubusercontent.com/77656115/206965843-6ac74168-3e31-43d6-9bbf-3e3d25e17522.png)

### Goal:
To Crate a deep learning model that is capable of recognizing deepfake images. A
thorough analysis of deepfake video frames to identify slight imperfections in the face
head and the model will learn what features differentiate a real image from a deepfake.

![image](https://user-images.githubusercontent.com/77656115/206965890-a1c345cf-8ae9-49f7-b498-ae4c7168666a.png)

### Project Pipeline

| Steps | Dscription |
| --- | --- |
| Step1 |    Loading the datasets |
| Step2 | Extracting videos from the dataset |
| Step3  | Extract all frames in the video for both real and fake |
| Step4 | Recognize the face subframe |
| Step5 |Locating the facial landmarks |
| Step6 |Frame-by-frame analysis to address any changes in the face landmarks |
| Step7 | To Classify the video either as REAL or Fake.|


## General WorkFlow:
### Pre-processing:
![image](https://user-images.githubusercontent.com/77656115/206968030-1e9729e7-8d34-4295-a110-d05ad0ade7bb.png)

### Prediction WorkFlow:
![image](https://user-images.githubusercontent.com/77656115/206968272-73db6238-79a0-46a1-ad5b-e651ad002322.png)

# Models Usage: 
### Models with CNN Architecture

Implemented the following models with CNN architecture
**MesoNet**
- This model is pre-trained to detect deepfake images, but it is bad at detecting Fake 
video frames
**ResNet50v**
- This model is trained using dee fake images cropped from the videos with preset 
weights of imagenet dataset
**EfficientNetB0**
- This model is also trained using deepfake images cropped from the videos with 
preset weights of imagenet dataset

### Models with CNN + Seqential Architecture
**InceptionV3(CNN Model) + GRU(sequential)**

-  This model works well because of both CNN and Sequential architecture.
- Test Accuracy is approx. 82%
- For Each Frame in the Video, it will generate the feature Vectors
- HyperParameters used: 
- Optimizer: Adam ( Adam Works fine as it changes the Learning Rate over time )
- Metric as Accuracy
- loss as sparse_categorical_crossentropy (loss function when there are two or more 
label classes )
- Among all the Optimizers Adam is Working Well.
- The accuracy of the model increases as the epochs are increasing.

**Limitations**
This model doesn’t work well when there are multiple faces in the Video, as it needs to 
detect the multiple faces in each Frame.

**EfficientNetB2(CNN Model) + GRU(sequential)**

- This model works well because of both CNN and Sequential architecture
- Test Accuracy is approx. 85%
- For Each Frame in the Video, it will generate the feature Vectors
- HyperParameters used: 
- Optimizer: Adam ( Adam Works fine as it changes the Learning Rate over time )
- Metric as Accuracy
- loss as sparse_categorical_crossentropy (loss function when there are two or more 
label classes )
- Among all the Optimizers Adam is Working Fine.
- The accuracy of the model increases as the epochs are increasing.
**Limitations**
- This model doesn’t work well when there is dark background in the video frames. As it is 
difficult to detect the faces in the Video Frame.

## Running Code
- Combination of CNN and RNN model is used to detect Fake Videos. We achieved a test accuracy ~85% on sample DFDC dataset
- To run this code first run this command.
```bash
  pip install -r requirements.txt
```

**Run main.py file in deploy folder**
```bash
  python main.py
```
*Make sure the required packages are installed, and it is preferred to run on GPU. The results are given in about a minute for a 10 second 30fps video.*

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.w3schools.com/css/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg" alt="css3" width="40" height="40"/> </a> <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> </a> <a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

## Conclusion:

- In this project, we have implemented a method for the detection of Deep-Fake videos using the 
combination of CNN and RNN architecture. We have kept our focus on Face-Swapped Deep-Fake 
videos.

- We primarily experimented only with various pre-trained CNN models like EfficientNet, and 
ResNet by finding the probability of each video frame being fake and predicting the output based on an aggregate of these probabilities. But the results weren’t satisfactory, so we went forward by combining CNN and RNN models.

- For the CNN + RNN model, the features of face-cropped video frames are extracted using pretrained CNN models and it is passed onto the RNN model which classifies the video as REAL or 
FAKE. We Experimented with EfficientNet and inception net for the feature extraction part and 
GRU is used to make the classification. We have obtained a maximum Test Accuracy of ~85% 
using this approach. Our model has high precision for FAKE videos which is obtained by giving 
more FAKE videos during the training of the Model.


## Team :
1.  [Balaji Kartheek](https://github.com/Balaji-Kartheek)
2.  [Aaron Dsouza](https://github.com/DsouzaAaron)



