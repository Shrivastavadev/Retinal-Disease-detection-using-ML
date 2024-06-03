
# ENHANCING RETINAL DISEASE DETECTION USING RETINAL FUNDUS IMAGES BASED ON ENSEMBELING DEEP LEARNING HETEROGENEOUS MODELS

Automatic multi-disease detection models have showed promise in tackling the
widespread issue of avoidable or undetected blindness and visual impairment. 

In this project, we propose a multi-disease detection pipeline for retinal disease identification, the pipeline was inspired from the work of [Muller et. al.]("https://www.nature.com/articles/s41467-021-25138-w") did and the modifications were done on their proposed pipeline along with construction of a new data set for better training and testing of the models used. The pipeline uses ensemble techniques that combines the prediction power of many heterogeneous deep convolutional neural network models using ensemble learning. 

Modern techniques including transfer learning, class weighting, real-time picture augmentation, and the use of focal loss are all incorporated into the pipeline. Moreover, we use ensemble learning methods including stacked logistic regression models, bagging via 5-fold cross-validation, and heterogeneous deep learning models.We were able to validate and show excellent accuracy and dependability of our pipeline.


## Proposed Pipeline

![pipeline](https://github.com/Shrivastavadev/Retinal-Disease-detection-using-ML/assets/137807080/169a62b8-c762-448d-9b89-37c3351bcdc9)




## RFMiD 3.0

It is a custom dataset created for the project purpose which was made by combining pre existing datasets, which are:
[RFMiD]("https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification?resource=download-directory"), [RFMiD 2.0]("https://www.mdpi.com/2306-5729/8/2/29"), [IDRiD]("https://www.mdpi.com/2306-5729/3/3/25"), [Eye Disease Dataset (kaggle)]("https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification"). 

These datasets were processed so factors like dimensions, retinal image circle, clearity are roughly similar. Then the datasets were split into training, testing and validation sets. Finally all after all these datasets were processed individually, the training, testing, and validation sets were mixed to form a bigger train-test dataset. By doing this we were able to make a dataset where we had 11,372 retinal images (we had more but got filtered during initial processing rounds.)

Since the dataset was highly imbalanced and classification of all 58 diseases might not be possible. Hence the classification of 38 and one other category was made.

After data collection all the images were square padded so as to not lose the retinal image during resizing.


<img src="https://github.com/Shrivastavadev/Retinal-Disease-detection-using-ML/assets/137807080/14c4c7da-503a-49d7-8120-0c1442ad35c5" width = "450" height = "250" alt="Data" />

## Understanding The Pipeline

Our pipeline consists of first the data cleaning part where square padding was done, followed by upsampling by introducing flip, brightness variation, and variation in Hue. After this the images are first fed into a two CNN architecture consisting of DenseNet121, and ResNet152, thereafter a weighted average ensemble is created for disease risk detector which simply classifies the image into diseased or not. After This The image is fed into a series of CNNs which are two DensNet121, VGG-16, VGG-19, two DensNet201, ResNet152, EfficientNetB0, EfficientNetB7. All these CNNs are trained majorly on different subset of data depending upon the distribution.

This step is crucial as the dataset is highly imbalanced getting high accuracy with just one classifier for multi-label problem will not be a fruitful attempt.  Then these CNNs are stacked together by using their output to create a support vector regression ensemble. 

Another approach of using stacked logistic regression for individual classes is hypothesized for testing. The CNNs selected for the ensemble were first tested along with some other CNNs on part of training data as to compare accuracy of other networks and then selected, the training was done for 50 epochs for each CNN. 

The tried and tested networks with their accuracy is given in the following Tabel 4. After selecting the architectures, the disease risk detector was created, which is a binary classifier that classifies whether a given image is diseased or not. This classifier should be of high accuracy and for that purpose we selected the two highest scoring CNNs from tabel 4, that is DensNet121 and ResNet152. The classifiers were all initialized with weights of imagenet dataset, and last 8 layers were unfreezed for transfer learning application. Binnary crossentropy loss function was used along with sigmoid or tanh
activation function. The Ensemble of these 2 classifiers was created using weighted average stacking, where DensNet121 was given weight 0.6 as it performed better during training, and ResNet152 was given weight 0.4.

        Table 4: CNN architecture selection

        | CNN             | Train Accuracy | Test Accuracy |
        | ------          | -------------- | ------------- |
        | DensNet121      | 59.43%         | 76.64%        |
        | DensNet169      | 58.11%         | 71.23%        |
        | DensNet201      | 58.48%         | 75.40%        |
        | ResNet152       | 80.85%         | 91.18%        |
        | VGG-19          | 63.28%         | 79.82%        |
        | VGG-16          | 59.94%         | 81.82%        |
        | EfficientNetB7  | 61.11%         | 74.31%        |
        | EfficientNetB0  | 63.28%         | 79.82%        |
        | EfficientNetB3  | 59.55%         | 72.66%        |
        | InceptionV3     | 54.36%         | 59.16%        |

Selected CNNs: DensNet121, VGG-16, VGG-19, DensNet201, ResNet152, EfficientNetB0, EfficientNetB7.
## Results and Discussion:

In the piepeline proposed we were able to achieve the following goals:
- Data collection and processing
- Training and testing individual CNNs
- Developing Disease Detector Ensemble

Things proposed in pipeline but were not achieved:
- Stacking of trained CNNs using support vector regression 
- Comparision of stacking via support vector regression vs logistic regression

The Disease detector ensemble showed promising results, the train test accuracies are shown in the table below:

        Table: Disease Detector Enasemble
        
        | CNN         | Train Accuracy | Test Accuracy |
        | --------    | -------------- | ------------- |
        | DensNet121  | 93.86%         | 96.17%        |
        | DensNet201  | 95.11%         | 91.13%        |
        | Ensemble    | 92.85%         | 95.63%        |        

## Appendix

The detailed report of the work done can be find below:

- [Retinal Disease detection using ML]("https://github.com/Shrivastavadev/Retinal-Disease-detection-using-ML/blob/main/Retinal%20Disease%20Deection%20using%20ML.pdf")


## Acknowledgements

 - [Subhankar Mishra](https://niser.ac.in/~smishra)
 - [Shubhanshu Prasad](https://linkedin.com/in/shubhanshu-prasad)


