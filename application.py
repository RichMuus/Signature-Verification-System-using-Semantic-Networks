import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from joblib import load

# Load the trained SVM model
svm = load('model.pkl')

# Define a function to extract HOG features from an image
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

# Create a Tkinter window
window = Tk()
window.title('Signature Verification System')

# Create a canvas for displaying the signature image
canvas = Canvas(window, width=300, height=200)
canvas.pack()

# Create a function for selecting a signature image
def select_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    
    # Read the selected image file
    img = cv2.imread(file_path)
    
    # Display the image on the canvas
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    canvas.create_image(0, 0, anchor=NW, image=img_tk)
    canvas.image = img_tk
    
    # Perform signature verification using the trained SVM model
    img_processed = cv2.resize(img, (64, 124))
    img_hog = hog_features(img_processed)
    label = svm.predict(img_hog.reshape(1, -1))
    if label == 1:
        result_label.config(text='Signature not verified.😫')
    else:
        result_label.config(text='Signature  verified.👍')
    
# Create a button for selecting a signature image
select_button = Button(window, text='Select Image', command=select_image)
select_button.pack()

# Create a label for displaying the signature verification result
result_label = Label(window, text='')
result_label.pack()

# Start the Tkinter event loop
window.mainloop()
