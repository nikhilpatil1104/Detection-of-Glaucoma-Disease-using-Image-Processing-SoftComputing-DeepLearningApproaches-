import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Set the absolute path for the dataset directory
data_dir = r'D:\Vs code\Python\model\dataset_directory'  # Use absolute path
categories = ['Glaucoma', 'Normal']

image_size = 224  # Image resizing to fit the model

# Ensure the dataset directory exists
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Function to load and preprocess the dataset
def load_data():
    data = []
    labels = []
    
    for category in categories:
        path = os.path.join(data_dir, category)
        class_label = categories.index(category)  # Assign numerical label based on category
        
        # Print the path being checked and the files in it
        print(f"Checking path: {path}")  # Full path check
        files = os.listdir(path)
        if files:
            print(f"Files in {category} directory: {files}")
        else:
            print(f"No files found in {category} directory.")
        
        for img in files:
            try:
                # Load the image in grayscale (or color if preferred)
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                
                # Check if the image was loaded properly
                if img_array is None:
                    print(f"Failed to load image: {img}")
                    continue
                
                # Resize the image to the desired size
                img_array = cv2.resize(img_array, (image_size, image_size))
                
                # Normalize the image
                img_array = img_array / 255.0
                
                # Append the image and label to their respective lists
                data.append(img_array)
                labels.append(class_label)

            except Exception as e:
                print(f"Error processing image {img}: {e}")
                continue
            
    return np.array(data), np.array(labels)

# Load the preprocessed data
X, y = load_data()

# Check if any images were loaded
if len(X) == 0:
    raise ValueError("No images were loaded. Please check your dataset directory.")

# Reshape the data for CNN input (adding an extra channel dimension for grayscale)
X = X.reshape(-1, image_size, image_size, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")
