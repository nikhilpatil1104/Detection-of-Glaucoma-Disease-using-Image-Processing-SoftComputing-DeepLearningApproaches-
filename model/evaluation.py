from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Set constants
model_file_path = r'D:\Vs code\Python\model\my_model.keras'  # Path to your saved model
image_size = 20  # Image size should match the one used during training (20x20)
num_classes = 2  # Glaucoma and Normal categories

# Function to preprocess images in the test directory
def preprocess_image(img_path):
    try:
        # Load the image in grayscale
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            raise ValueError(f"Image not found or cannot be opened: {img_path}")
        
        # Resize the image to the size used by the model (20x20)
        img_array = cv2.resize(img_array, (image_size, image_size))
        
        # Normalize the image
        img_array = img_array / 255.0
        
        # Flatten the image to fit model input (batch size, features)
        img_array = img_array.flatten()  # Flatten the image
        return img_array
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# Load the trained model
try:
    model = load_model(model_file_path)
    print("Model loaded successfully")
except OSError as e:
    print(f"Error loading model: {e}")
    exit()

# Test image directory (replace with your actual test images folder path)
test_data_dir = r'D:\Vs code\Python\model\dataset_directory\Normal'  # Path to your test set

# Check if the test directory exists
if not os.path.exists(test_data_dir):
    print(f"Test data directory not found: {test_data_dir}")
    exit()

test_images = os.listdir(test_data_dir)

# Prepare test data
X_test = []
y_test = []

# Use the actual class label for Normal (1) and Glaucoma (0)
for img in test_images:
    img_path = os.path.join(test_data_dir, img)
    preprocessed_img = preprocess_image(img_path)
    if preprocessed_img is not None:
        X_test.append(preprocessed_img)
        y_test.append(1)  # Assuming these are 'Normal' class images

# Convert lists to arrays
X_test = np.array(X_test)  # Convert list to numpy array
y_test = np.array(y_test)

# Check if there are any images to evaluate
if len(X_test) == 0:
    print("No images found in the test directory.")
    exit()

# Convert labels to categorical
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Plot training history (ensure 'history' contains the correct data)
# Example data; replace with your actual training history
history = {
    'accuracy': [0.8, 0.85, 0.9, 0.95],  # Example training accuracies
    'val_accuracy': [0.75, 0.8, 0.85, 0.9]  # Example validation accuracies
}

# Check if 'history' contains the required keys
if 'accuracy' in history and 'val_accuracy' in history:
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.show()
else:
    print("History data not available for plotting.")
