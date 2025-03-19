import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse
import os
import pickle
import cv2
import random
import matplotlib.pyplot as plt

# Function to load images from categorized folders
def load_images_from_folders(folder_path, label, img_size=(128, 128)):
    images = []
    labels = []
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, img_size)  # Resize for consistency
            
            images.append(img)
            labels.append(label)  # Assign label (0 = forgery, 1 = genuine)

    return images, labels

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train Signature Verification LSTM Model")
parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset folder")
parser.add_argument('--save_model', type=str, default="signature_model.pkl", help="Path to save the trained model")
args = parser.parse_args()

# Define dataset structure
genuine_path = os.path.join(args.dataset, 'offline_genuine')
forgery_path = os.path.join(args.dataset, 'offline_forgery')

if not os.path.exists(genuine_path) or not os.path.exists(forgery_path):
    raise FileNotFoundError("Dataset should have 'offline_genuine' and 'offline_forgery' subfolders")

# Load images from folders
genuine_images, genuine_labels = load_images_from_folders(genuine_path, label=1)
forgery_images, forgery_labels = load_images_from_folders(forgery_path, label=0)

# Combine and shuffle dataset
all_images = np.array(genuine_images + forgery_images, dtype=np.float32) / 255.0  # Normalize
all_labels = np.array(genuine_labels + forgery_labels, dtype=np.int32)

# Shuffle data
combined = list(zip(all_images, all_labels))
random.shuffle(combined)
all_images, all_labels = zip(*combined)
all_images, all_labels = np.array(all_images), np.array(all_labels)

# Reshape images to fit LSTM input format
all_images = all_images.reshape(all_images.shape[0], all_images.shape[1], all_images.shape[2], 1)  # Add channel dimension
train_size = int(0.8 * len(all_images))

train_data, test_data = all_images[:train_size], all_images[train_size:]
train_labels, test_labels = all_labels[:train_size], all_labels[train_size:]

# Define the LSTM-based model
model = keras.Sequential([
    layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu', input_shape=(None, 128, 128, 1))),
    layers.TimeDistributed(layers.MaxPooling2D((2,2))),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Define callbacks
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(train_data, train_labels, batch_size=16, epochs=10, validation_split=0.2, callbacks=[early_stop])

# Evaluate the model
final_loss, final_accuracy = model.evaluate(test_data, test_labels)
print("Final Loss: {:.2f}%".format(final_loss * 100))
print("Final Accuracy: {:.2f}%".format(final_accuracy * 100))

# Save the trained model as a .pkl file
with open(args.save_model, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved successfully as {args.save_model}")

# Plot accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
