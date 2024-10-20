import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

# Define input dimensions
input_dim = 20  # Adjust according to your dataset
model_path = 'D:\\Vs code\\Python\\model\\my_model.keras'  # Update the path if necessary

# Ensure the model directory exists
model_dir = os.path.dirname(model_path)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Create a simple neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),  # Input layer
    Dense(10, activation='softmax')  # Output layer for a 10-class classification problem
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate dummy training data
X_train = np.random.rand(100, input_dim)  # 100 samples, each with `input_dim` features
y_train = np.random.randint(0, 10, 100)  # Random class labels for 10 classes
y_train = np.eye(10)[y_train]  # One-hot encode the labels

# Train the model
model.fit(X_train, y_train, epochs=10)

# Save the model in the recommended format
model.save(model_path)  # Save the model

# Check if the model file exists and load the model
if os.path.exists(model_path):
    print("File exists, attempting to load the model...")
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
        model.summary()  # Display the model's architecture
    except OSError as e:
        print(f"Error loading the model. OSError: {e}")
else:
    print("File not found after saving. Please check the path.")

# Example of making predictions (if desired)
X_test = np.random.rand(10, input_dim)  # 10 samples for testing
predictions = model.predict(X_test)  # Make predictions
print("Predictions:", predictions)  # Display predictions
