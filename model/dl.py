import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Example data - replace with your actual dataset
# Assuming you have image data and labels loaded in X_train, y_train, X_test, and y_test
# X_train = ...  # Load your training images here
# X_test = ...   # Load your testing images here
y_train = np.array([0, 1, 0, 1])  # Example labels for training data
y_test = np.array([0, 1])          # Example labels for testing data

# Define image size
image_size = 224  # Example image size

# Convert labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# Load the pre-trained VGG16 model without the top layers
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze VGG16 layers
for layer in vgg.layers:
    layer.trainable = False

# Build the model
model = Sequential()

# Add the VGG16 base model
model.add(vgg)

# Add custom layers on top
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 2 classes: Glaucoma, Normal

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Assuming you have training data loaded in X_train and y_train_cat
# Train the model (uncomment and provide your actual training data)
# history = model.fit(X_train, y_train_cat, validation_split=0.2, epochs=10, batch_size=32)

# Save the model for later use
model.save('my_model.keras')  # Save the trained model to a file
