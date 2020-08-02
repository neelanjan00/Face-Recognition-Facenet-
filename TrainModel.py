import numpy as np
from tensorflow.python.keras.models import load_model
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
import os, pickle
from Preprocess import extract_face, load_face, load_dataset, get_embedding

# load train dataset
trainX, trainy = load_dataset('./faces-dataset/train/')
print(trainX.shape, trainy.shape)

# load test dataset
testX, testy = load_dataset('./faces-dataset/val/')
print(testX.shape, testy.shape)

# load facenet pretrained model
facenet_model = load_model('facenet_keras.h5')
print('Loaded Model')

# convert each face in the train set into embedding
emdTrainX = list()
for face in trainX:
    emd = get_embedding(facenet_model, face)
    emdTrainX.append(emd)
    
emdTrainX = np.asarray(emdTrainX)
print(emdTrainX.shape)

# convert each face in the test set into embedding
emdTestX = list()
for face in testX:
    emd = get_embedding(facenet_model, face)
    emdTestX.append(emd)
emdTestX = np.asarray(emdTestX)
print(emdTestX.shape)

# print dataset characteristics
print("Dataset: train=%d, test=%d" % (emdTrainX.shape[0], emdTestX.shape[0]))

# normalize input vectors
in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)

# Save the encoder
np.save('classes.npy', out_encoder.classes_)

trainy_enc = out_encoder.transform(trainy)
testy_enc = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(emdTrainX_norm, trainy_enc)

# predict
yhat_train = model.predict(emdTrainX_norm)
yhat_test = model.predict(emdTestX_norm)

# score
score_train = accuracy_score(trainy_enc, yhat_train)
score_test = accuracy_score(testy_enc, yhat_test)

# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

# save the model
with open('SVCtrainedModel.pkl', 'wb') as f:
    pickle.dump(model, f)