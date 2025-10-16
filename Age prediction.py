import os
import cv2
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, GlobalMaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf




# Define the path to the dataset directory
DATASET_DIR = 'datasets/UTKFace/'
# Get a list of all filenames in the dataset directory
file_names = os.listdir(DATASET_DIR)

# List to store preprocessed image data
image_data = []




# Loop through each filename to load and preprocess the images
for file_name in file_names:
    # Read the image using OpenCV
    image = cv2.imread(DATASET_DIR + file_name)
    # Resize the image to a fixed size (128x128)
    image_resized = cv2.resize(image, (128, 128))
    # Convert the image from BGR (OpenCV default) to grayscale, reducing channel depth to 1
    image_grayscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    # Append the processed image to the list
    image_data.append(image_grayscale)





# Convert the list of images into a NumPy array
X_data_raw = np.array(image_data)

# Reshape the array to include the channel dimension (1 for grayscale): (N, 128, 128, 1)
# N is the number of samples (images)
X_data_reshaped = X_data_raw.reshape(len(X_data_raw), 128, 128, 1)

# Normalize the pixel values to the range [0, 1] for better model performance
X_data_normalized = X_data_reshaped / 255.0





# List to store the age part of the filenames (the target variable)
age_strings = []
# Extract the age from the filename (assuming the UTKFace naming convention where age is the first part)
for file_name in file_names:
    # Age is typically the first part of the filename, separated by an underscore
    age_part = file_name.split('_')[0]
    age_strings.append(age_part)

# Convert the list of age strings to a NumPy array of integers (the actual target ages)
y_targets = np.array([int(age) for age in age_strings])





# --- Model Definition (Convolutional Neural Network - CNN) ---
age_prediction_model = Sequential()

# First Convolutional Block
# Conv2D: Learns 32 features/filters from the 128x128x1 input.
age_prediction_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
# MaxPooling2D: Downsamples the feature map by taking the maximum value over a 2x2 window, reducing dimensions by half [[3](https://keras.io/api/layers/pooling_layers/max_pooling2d/), [6](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)]
age_prediction_model.add(MaxPooling2D((2, 2)))

# Second Convolutional Block
age_prediction_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
age_prediction_model.add(MaxPooling2D((2, 2)))

# Third Convolutional Block
age_prediction_model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
age_prediction_model.add(MaxPooling2D((2, 2)))

# Fourth Convolutional Block
age_prediction_model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
age_prediction_model.add(MaxPooling2D((2, 2)))

# Replaces the need for a Flatten layer by taking the maximum value across all spatial dimensions
# of the final feature map, preparing it for the Dense layers.
# Note: Flatten or GlobalMaxPooling2D is needed to convert 2D feature maps to 1D vectors [[2](https://abadis.ir/entofa/flatten/), [5](https://dictionary.cambridge.org/dictionary/english/flatten)]
age_prediction_model.add(GlobalMaxPooling2D())

# Fully Connected Layer (Dense) - A hidden layer for non-linear learning
age_prediction_model.add(Dense(132, activation='relu'))

# Output Layer (Dense) - Single neuron with no activation (linear) for regression output (age prediction)
age_prediction_model.add(Dense(1))

# Print a summary of the model architecture
age_prediction_model.summary()

# Compile the model
# loss='mse' (Mean Squared Error) is standard for regression tasks (predicting a continuous value like age)
age_prediction_model.compile(loss='mse', optimizer='adam', metrics=['mae']) # Changed 'accuracy' to 'mae' (Mean Absolute Error) for regression






# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data_normalized, y_targets, test_size=0.2, random_state=42)

# Train the model
# 'validation_data' is used to evaluate the model's performance on the test set after each epoch
history = age_prediction_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Make predictions on the test set
predictions = age_prediction_model.predict(X_test)

# Save the trained model weights for later use
age_prediction_model.save_weights(filepath='age_prediction_weights.h5')