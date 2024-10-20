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

# Create folders for each category inside the dataset directory
for category in categories:
    category_path = os.path.join(data_dir, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)

# Function to load, preprocess, and save the dataset
def load_and_save_data():
    data = []
    labels = []
    
    for category in categories:
        path = os.path.join(data_dir, category)
        class_label = categories.index(category)
        img_index = 1  # To keep track of image numbering
        
        # Print the path being checked and the files in it
        print(f"Checking path: {path}")  # Full path check
        files = os.listdir(path)
        if files:
            print(f"Files in {category} directory: {files}")
        else:
            print(f"No files found in {category} directory.")
        
        for img in files:
            try:
                # Load the image in grayscale
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                
                # Resize the image to the desired size
                img_array = cv2.resize(img_array, (image_size, image_size))
                
                # Normalize the image
                img_array = img_array / 255.0
                
                # Save the preprocessed image
                save_name = f"{category}_{img_index:03d}.jpg"  # Save as category_001.jpg, etc.
                save_path = os.path.join(path, save_name)
                cv2.imwrite(save_path, img_array * 255)  # Multiplying by 255 to restore pixel values for saving

                data.append(img_array)
                labels.append(class_label)
                
                img_index += 1  # Increment the image index
                
            except Exception as e:
                print(f"Error processing image {img}: {e}")
                continue
            
    return np.array(data), np.array(labels)

# Load and save the preprocessed data
X, y = load_and_save_data()

# Reshape the data for CNN input (adding an extra channel dimension for grayscale)
X = X.reshape(-1, image_size, image_size, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
