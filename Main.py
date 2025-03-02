import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from google.cloud import vision
import io

tf.keras.mixed_precision.set_global_policy('mixed_float16')

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"json file path"

output_folder = r"output path"
splitting_folder = r"splitting path"
train_dir = r"train path"
test_dir = r"test path"
single_image_path = r"image path"

for directory in [output_folder, splitting_folder, train_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.3
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid', dtype=tf.float32)(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
)

model.save('diabetic_kid_present_vs_not_present.h5')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

evaluation = model.evaluate(test_generator)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

def analyze_image_with_vision_api(image_path):
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    return labels

image_directory = test_dir

for image_file in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_file)
    if os.path.isfile(image_path) and image_path.endswith(('.png', '.jpg', '.jpeg')):
        print(f"Analyzing {image_path}")
        labels = analyze_image_with_vision_api(image_path)
        for label in labels:
            print(label.description, label.score)

