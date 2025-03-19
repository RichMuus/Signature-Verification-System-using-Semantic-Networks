from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from joblib import load
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained SVM model
svm = load('model.pkl')

# Define function to extract HOG features from an image
def hog_features(img):
    win_size = (64, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_feats = hog.compute(img)
    hog_feats = hog_feats.ravel()
    return hog_feats

# Define route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle image uploads
@app.route('/verify', methods=['POST'])
def verify_signature():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the image
    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Process the image
    img_processed = cv2.resize(img, (64, 124))
    img_hog = hog_features(img_processed)

    # Predict using SVM model
    label = svm.predict(img_hog.reshape(1, -1))

    # Return result
    result = "This is a Genuine User's Signature " if label == 0 else "Signature Not Genuine"
    return jsonify({'result': result})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
