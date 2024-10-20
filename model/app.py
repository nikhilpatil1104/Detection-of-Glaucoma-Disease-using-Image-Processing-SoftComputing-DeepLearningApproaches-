from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the glaucoma detection model
model = load_model('glaucoma_detection_model.h5')

image_size = 224  # Image input size for the model

# Preprocessing function (matches model preprocessing)
def preprocess_image(image):
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = image.reshape(1, image_size, image_size, 1)  # Reshape for model input
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Read the image from the uploaded file
        file = request.files['image']
        image = np.fromfile(file, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Preprocess image for the model
        processed_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(processed_image)
        glaucoma_prob = prediction[0][1] * 100  # Assuming the 2nd class is 'Glaucoma'
        
        # Return prediction as JSON
        return jsonify({'glaucoma_probability': f'{glaucoma_prob:.2f}%'})

if __name__ == '__main__':
    app.run(debug=True)
