import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tensorflow as tf
import os
import pickle

# Define the filename to store the training data
DATA_FILE = "training_data.pkl"

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def save_data():
    '''Function to save the training data to a file'''
    with open(DATA_FILE, 'wb') as f:
        pickle.dump([train_x, train_y], f)

def load_data():
    '''Function to load the training data from a file'''
    global train_x, train_y
    try:
        with open(DATA_FILE, 'rb') as f:
            loaded_data = pickle.load(f)
            train_x, train_y = loaded_data
    except (pickle.UnpicklingError, FileNotFoundError, EOFError) as e:
        messagebox.showerror("Error Loading Data", f"Error loading data: {e}")

def submit_data():
    '''
    Function to submit training data.

    1. Validates the input from the `entry` widget, ensuring it is a single digit (0-9).
    2. Processes the drawn image:
       - Resizes it to 28x28 pixels and converts it to grayscale.
       - Converts it to a NumPy array, normalizes pixel values to [0, 1], and inverts the colors.
       - Adds a batch dimension to prepare it for storage.
    3. Appends the processed image and the digit label to the training datasets (`train_x` and `train_y`).
    4. Clears the canvas and entry widget, resets the drawing surface, and saves the updated training data.
    '''

    if len(entry.get()) != 1:
        messagebox.showerror("Error", "The expected value is between 0 and 9.")
        return
    try:
        digit = int(entry.get())
    except ValueError:
        messagebox.showerror("Error", "The expected value is between 0 and 9.")
        return

    global drawn_image, train_x, train_y
    image = drawn_image.resize((28, 28)).convert('L')
    image_array = np.array(image) / 255.0
    image_array = 1.0 - image_array
    
    train_x = np.append(train_x, [image_array], axis=0)
    train_y = np.append(train_y, [digit], axis=0)

    canvas.delete("all")
    entry.delete(0, tk.END)
    drawn_image = Image.new("L", (250, 250), "white")
    save_data()

def train_model():
    '''Function to train the model'''
    global train_x, train_y
    if len(train_x) == 0:
        canvas.delete("all")
        messagebox.showerror("Error", "No training data submitted.")
        return
    model.fit(train_x, train_y, epochs=20)

def preprocess_image(image):
    '''
    Preprocesses a drawn image for model input.

    1. Resizes the image to 28x28 pixels.
    2. Converts the image to grayscale.
    3. Transforms the image into a NumPy array and normalizes pixel values to [0, 1].
    4. Inverts the pixel values to assume a dark background.
    5. Adds a batch dimension for compatibility with model input requirements.
    '''
    image = image.resize((28, 28)).convert('L')
    image_array = np.array(image) / 255.0
    image_array = 1.0 - image_array
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_number():
    '''
    Uses the trained model to predict the number from the drawn image.

    1. Checks if the model has been trained by verifying `train_x` is not empty.
       If not trained, displays an error message and resets the canvas.
    2. Preprocesses the drawn image for prediction.
    3. Makes a prediction using the trained model.
    4. Extracts the predicted number from the model's output.
    5. Displays the predicted number in a message box and resets the canvas.
    '''
    global drawn_image
    if len(train_x) == 0:
        drawn_image = Image.new("L", (250, 250), "white")
        canvas.delete("all")
        messagebox.showerror("Error", "The model has not been trained yet.")
        return

    processed_img = preprocess_image(drawn_image)
    prediction = model.predict(processed_img)
    predicted_number = np.argmax(prediction)

    canvas.delete("all")
    messagebox.showinfo("Predicted Number", "The predicted number is: " + str(predicted_number))
    drawn_image = Image.new("L", (250, 250), "white")
    print(prediction)

def draw(event):
    '''Function to draw on the canvas'''
    global drawing, prev_x, prev_y, drawn_image
    x, y = event.x, event.y
    if drawing and prev_x is not None and prev_y is not None:
        canvas.create_line(prev_x, prev_y, x, y, width=11, fill='black')
        drawn_image_draw = ImageDraw.Draw(drawn_image)
        drawn_image_draw.line([prev_x, prev_y, x, y], width=11, fill='black')
    prev_x, prev_y = x, y

def start_draw(event):
    '''Function to start drawing'''
    global drawing
    drawing = True

def stop_draw(event):
    '''Function to stop drawing'''
    global drawing, prev_x, prev_y
    drawing = False
    prev_x, prev_y = None, None

# Check if the training data file exists and is not empty
if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
    # Load the training data from the file
    with open(DATA_FILE, 'rb') as f:
        loaded_data = pickle.load(f)
        train_x, train_y = loaded_data
else:
    # If the file doesn't exist or is empty, initialize with empty arrays
    train_x = np.zeros((0, 28, 28))
    train_y = np.zeros((0,), dtype=int)

# Create the graphical user interface for drawing and making predictions
root = tk.Tk()
root.title("Handwritten Digit Recognition")
canvas = tk.Canvas(root, width=250, height=250, bg='white')
canvas.pack()
train_button = tk.Button(root, text="Submit Data", command=submit_data)
train_button.pack(side=tk.RIGHT, padx=5)
train_model_button = tk.Button(root, text="Train Model", command=train_model)
train_model_button.pack(side=tk.RIGHT, padx=5)
entry = tk.Entry(root)
entry.pack(side=tk.RIGHT, padx=5)
predict_button = tk.Button(root, text="Predict", command=predict_number)
predict_button.pack(side=tk.LEFT, padx=5)

drawing = False
prev_x, prev_y = None, None
drawn_image = Image.new("L", (250, 250), "white")

canvas.bind("<B1-Motion>", draw)
canvas.bind("<Button-1>", start_draw)
canvas.bind("<ButtonRelease-1>", stop_draw)

root.mainloop()
