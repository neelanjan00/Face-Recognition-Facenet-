# FaceNet Face Recognition
FaceNet model is an implementation of the Siamese Neural Network, trained using a triplet loss function, which uses a similarity function to measure how similar are the images of two given individuals in order to recognise them.

##Installation
This particular implementation uses a Keras model for the Siamese Network to compute the embeddings and later a SVC classifier is trained on the embeddings to perform the actual face recognition.

1. Install the python packages mentioned in 'Requirements.txt' file via the command "pip install -r Requirements.txt"
