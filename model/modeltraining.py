import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define a function to load your data
def load_data(data_directory):
    images = []
    labels = []
    # Loop through each category
    for label in os.listdir(data_directory):
        category_path = os.path.join(data_directory, label)
        if os.path.isdir(category_path):
            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)
                # Load image and resize
                image = load_img(image_path, target_size=(224, 224), color_mode='grayscale')
                image = img_to_array(image)
                images.append(image)
                labels.append(label)  # Assuming labels are the directory names
    return np.array(images), np.array(labels)

# Load your data
data_directory = 'D:\\Vs code\\Python\\model\\dataset_directory'  # Update this to your data directory
X, y = load_data(data_directory)

# Convert labels to categorical (one-hot encoding)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create the model
num_classes = 2  # Update this based on your actual number of classes
model = models.Sequential([
    layers.Input(shape=(224, 224, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Adding dropout for regularization
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model training with data augmentation
history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=32), 
                    epochs=20, 
                    validation_data=(X_test, y_test_cat),
                    callbacks=[early_stopping])

# Save the trained model
model.save('my_model.keras')
