import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Example preprocessing using CLAHE for contrast enhancement
def preprocess_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

# Your actual training and testing image paths
train_image_dir = "D:\\Vs code\\Python\\model\\dataset_directory\\Glaucoma"
test_image_dir = "D:\\Vs code\\Python\\model\\dataset_directory\\Normal"

# Check if the directories exist
if not os.path.exists(train_image_dir):
    raise ValueError(f"Training image directory does not exist: {train_image_dir}")

if not os.path.exists(test_image_dir):
    raise ValueError(f"Testing image directory does not exist: {test_image_dir}")

# Create lists to hold images
X_train = []
X_test = []

# Define the desired size for resizing
desired_size = (224, 224)  # You can change this size as needed

# Load training images
for filename in os.listdir(train_image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Handle different cases
        img_path = os.path.join(train_image_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Resize the image
            img_resized = cv2.resize(img, desired_size)
            X_train.append(img_resized)
        else:
            print(f"Warning: Unable to read image {img_path}")

# Load testing images
for filename in os.listdir(test_image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(test_image_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Resize the image
            img_resized = cv2.resize(img, desired_size)
            X_test.append(img_resized)
        else:
            print(f"Warning: Unable to read image {img_path}")

# Convert lists to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Check if images are loaded
if len(X_train) == 0:
    raise ValueError("No training images found in the specified directory.")
if len(X_test) == 0:
    raise ValueError("No testing images found in the specified directory.")

# Apply preprocessing to the training data
X_train_processed = np.array([preprocess_image(img) for img in X_train])
X_test_processed = np.array([preprocess_image(img) for img in X_test])

# Display the first preprocessed image
plt.imshow(X_train_processed[0], cmap='gray')
plt.title('Preprocessed Image Example')
plt.axis('off')
plt.show()
