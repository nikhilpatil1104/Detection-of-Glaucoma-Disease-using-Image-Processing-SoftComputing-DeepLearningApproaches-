import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Image size for resizing
image_size = 224  # Image resizing to fit the model

# Function to process a single frame (simulating the load and preprocess step for dataset)
def preprocess_frame(frame):
    try:
        # Convert the frame to grayscale
        img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize the image
        img_array = cv2.resize(img_array, (image_size, image_size))
        # Normalize the image
        img_array = img_array / 255.0
        # Reshape for compatibility with CNN input
        img_array = img_array.reshape(1, image_size, image_size, 1)
        return img_array
    except Exception as e:
        print("Error processing frame:", e)
        return None

# Initialize the webcam (0 is the default camera, change if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# List to hold processed frames (simulating dataset loading)
data = []

while True:
    # Capture frame-by-frame from webcam
    ret, frame = cap.read()

    # If frame is read correctly, ret will be True
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the current frame
    processed_frame = preprocess_frame(frame)

    # If processing is successful, append to data (simulating dataset creation)
    if processed_frame is not None:
        data.append(processed_frame)

    # Display the resulting frame (before preprocessing)
    cv2.imshow('Live Feed', frame)

    # Press 'q' to stop the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Convert data to NumPy array for further processing (like splitting into train/test sets)
data = np.array(data).reshape(-1, image_size, image_size, 1)

# Simulate labels (as we're not actually classifying here, just generating sample labels)
labels = np.zeros(len(data))  # This is a placeholder, you can adapt it as needed

# Split the data into training and testing sets (you can later use this for training models)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# At this point, you can feed `X_train`, `X_test`, `y_train`, `y_test` into a CNN for training.
