import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump
# Set up directories for genuine and forged signatures
genuine_dir = "C:\\Users\\richa\\Desktop\\projo\\TrainingSet\\Offline Genuine"
forged_dir ="C:\\Users\\richa\\Desktop\\projo\\TrainingSet\\Offline Forgeries"
# Load signature images and labels
images = []
labels = []
for folder, label in [(genuine_dir, 0), (forged_dir, 1)]:
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(label)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
# Extract HOG features from signature images
def hog_features(images):
    features = []
    for img in images:
        hog = cv2.HOGDescriptor()
        resized_img = cv2.resize(img, (64, 124))
        features.append(hog.compute( resized_img))
    return np.array(features)

# Extract features from training and testing sets
X_train_features = hog_features(X_train)
X_test_features = hog_features(X_test)
from sklearn.svm import SVC

# Train an SVM model on the extracted features
svm = SVC(kernel='linear')
svm.fit(X_train_features, y_train)
# Evaluate the SVM model on the testing set
svm_accuracy = svm.score(X_test_features, y_test)
print("SVM accuracy:", svm_accuracy)
dump(svm, 'C:\\Users\\richa\\Desktop\\model.pkl')
# Load a new signature image and extract features
new_img = cv2.imread("C:\\Users\\richa\\Desktop\\012_18.PNG", cv2.IMREAD_GRAYSCALE)
new_features = hog_features([new_img])

# Predict the authenticity of the new signature using the trained SVM model
prediction = svm.predict(new_features)[0]

if prediction == 0:
    print("This is a genuine signature.")
else:
    print("This is a forged signature.")
