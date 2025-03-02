import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model.h5')

single_image_path = r"image path"

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_single_image(image_path, model):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    return prediction

prediction = predict_single_image(single_image_path, model)
print(f"Prediction for the single image: {prediction}")

if prediction[0] > 0.5:
    print("Diabetic Kid Present")
else:
    print("Diabetic Kid Not Present")