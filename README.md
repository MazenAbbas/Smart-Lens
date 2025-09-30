Image Recognition with CIFAR-100
This project demonstrates a complete image recognition workflow using a Convolutional Neural Network (CNN). It is built to classify images from the CIFAR-100 dataset, a challenging collection of 60,000 tiny images spread across 100 distinct classes. The project showcases proficiency in data preparation, model building, and evaluation for computer vision tasks.

Key Features
Deep Learning Model: Utilizes a Sequential CNN model built with Keras.

Advanced Dataset: Trains and evaluates the model on the complex CIFAR-100 dataset.

Data Preprocessing: Normalizes image data and applies one-hot encoding for class labels.

Performance Visualization: Plots training and validation accuracy and loss to monitor model performance.

Technologies Used
Python: The core programming language.

TensorFlow & Keras: The primary libraries for building and training the deep learning model.

Matplotlib: Used for visualizing images and plotting model performance.

NumPy: For numerical operations, especially with arrays.

Google Colab: The development environment for this project.

How to Run the Project
To run this project, you will need a Google Colab or Jupyter Notebook environment.

Install Dependencies:

Python

!pip install tensorflow matplotlib
Load and Preprocess Data:

Python

import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)
Build the CNN Model:

Python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(100, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
Train the Model:

Python

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
Evaluate and Predict:

Python

import numpy as np
import matplotlib.pyplot as plt

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.2f}")

# You can plot accuracy and loss charts using the code provided in the notebook

# Example prediction
test_image_index = 100 
sample_image = x_test[test_image_index]
sample_image_reshaped = np.expand_dims(sample_image, axis=0)
prediction = model.predict(sample_image_reshaped)
predicted_class_index = np.argmax(prediction[0])

# Make sure to define `class_names` from the original notebook to display the result
# predicted_class_name = class_names[predicted_class_index]
# print(f"Predicted Class: {predicted_class_name}")
Project Results
The model's final test accuracy was a significant step in the right direction, showing it learned to recognize features in the images. The prediction of "tiger" for an image of a "crab" highlights the inherent complexity of the CIFAR-100 dataset, with its 100 classes presenting a notable challenge. This outcome serves as a basis for further model improvements, such as data augmentation or implementing transfer learning with a pre-trained model.
