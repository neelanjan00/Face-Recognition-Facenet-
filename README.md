# FaceNet Face Recognition
FaceNet model is an implementation of the Siamese Neural Network, trained using a triplet loss function, which uses a similarity function to measure how similar are the images of two given individuals in order to recognise them. This particular implementation uses a Keras model for the Siamese Network to compute the embeddings and later a SVC classifier is trained on the embeddings to perform the actual face recognition.

## Installation of Required Packages
Install the python packages mentioned in 'Requirements.txt' file via the command "pip install -r Requirements.txt"

## 1. Add Your Images to Dataset
Add your images to the faces-dataset folder for training via webcam, with the command "python Image_Dataset_Generator.py"

## 2. Calculate the Embeddings and Train the Classifier
Train the dataset using the command "python TrainModel.py"

## 3. Test the model
Test the model on any test image with the command "python PredictFaces.py --image 'path-to-the-test-image-here' "