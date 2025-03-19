# Signature Verification System
This project trains an LSTM-based deep learning model to classify handwritten signatures as either genuine or forged. The model uses convolutional layers for feature extraction and LSTM layers to learn sequential patterns in signature strokes.

## IMPORTANT -NOTICE
I HAVE ALREADY TRAINED MY MODEL. TO USE THE ALREADY TRAINED MODEL:

RUN ***python3 app.py***

🛠 Features
✅ Supports categorized datasets stored in separate folders (offline_genuine/ and offline_forgery/).
✅ Uses CNN for spatial feature extraction & LSTM for temporal signature pattern recognition.
✅ Automatically resizes and normalizes images for consistency.
✅ Early stopping prevents overfitting.
✅ Saves the trained model as a .pkl file for later us

# 📂 Dataset Structure
Organize your dataset as follows:


/dataset/
    ├── offline_genuine/   # Genuine signatures
    │   ├── genuine1.jpg
    │   ├── genuine2.jpg
    │   ├── ...
    │
    ├── offline_forgery/   # Forged signatures
    │   ├── forgery1.jpg
    │   ├── forgery2.jpg
    │   ├── ...

# 🚀 Installation
1️⃣ Create a Virtual Environment (Optional but Recommended)


python -m venv signature_env
source signature_env/bin/activate  # On Linux/Mac
signature_env\Scripts\activate     # On Windows
2️⃣ Install Dependencies



pip install tensorflow numpy opencv-python matplotlib argparse pickle5


# 🧑‍🏫 Training the Model
To train the model, run:



python train.py --dataset path_to_dataset --save_model signature_model.pkl

--dataset: Path to the dataset folder containing subfolders offline_genuine and offline_forgery.

--save_model: (Optional) Filename to save the trained model (.pkl file).

YOU CAN ALSO EDIT THE train.py FILE TO INCLUDE THE FILE PATHS TO DATASET AND FILE PATH WHERE TO SAVE THE TRAINED MODEL.

# 📈 Evaluating Model Performance
During training, the model logs accuracy and loss for training and validation. After training, the final accuracy and loss will be displayed:



Final Loss: 3.42%
Final Accuracy: 94.87%

# 📊 Visualizing Training Performance

The script will generate two plots:

1️⃣ Training vs Validation Accuracy

2️⃣ Training vs Validation Loss

These help analyze overfitting and learning trends.

🧐 Making Predictions
After training, you can load the model and predict whether a given signature is genuine or forged.

Example Prediction Script (predict.py)
***

import pickle

import cv2

import numpy as np


with open("my_signature_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

***Load and preprocess a new signature image***
img_path = "new_signature.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128)) / 255.0  # Resize and normalize
img = img.reshape(1, 128, 128, 1)  # Add batch and channel dimensions

*** Predict***
prediction = loaded_model.predict(img)
label = "Genuine" if prediction[0] > 0.5 else "Forgery"
print(f"Prediction: {label}")
Run Prediction
***


Run---

***python3 predict.py --image path_to_image.jpg***


# 📜 License
This project is licensed under MIT License.

***Developer -- Richard Musya Paul***

