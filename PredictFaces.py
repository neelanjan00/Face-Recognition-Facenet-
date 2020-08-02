from Preprocess import extract_face, get_embedding
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
import argparse
import pickle
import numpy as np

in_encoder = Normalizer()
out_encoder = LabelEncoder()
out_encoder.classes_ = np.load('classes.npy')
facenet_model = load_model('facenet_keras.h5')

with open('SVCtrainedModel.pkl', 'rb') as f:
    model = pickle.load(f)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Test Image Path")
# ap.add_argument("-n", "--name", required=True,
# 	help="Name of the person (same as the class name)")
args = vars(ap.parse_args())

random_face = extract_face(args['image'])
random_face_emd = in_encoder.transform([get_embedding(facenet_model, random_face)])[0]
# random_face_name = args['name']

samples = np.expand_dims(random_face_emd, axis = 0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predicted_name = out_encoder.inverse_transform(yhat_class)[0]
all_names = out_encoder.inverse_transform([i for i in range(len(out_encoder.classes_))])

print("Predicted Probabilities: ")
for i, name in enumerate(all_names):
    print(name, ": ", yhat_prob[0][i] * 100)
# print('Expected: %s' % random_face_name)
print('Predicted: %s' % predicted_name)
