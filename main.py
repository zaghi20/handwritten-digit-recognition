import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tensorflow as tf
import os
import pickle

# Define the filename to store the training data
DATA_FILE = "training_data.pkl"

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

# Function to save the training data
def save_data():
    with open(DATA_FILE, 'wb') as f:
        pickle.dump([train_x, train_y], f)

# Function to load the training data
def load_data():
    global train_x, train_y
    try:
        with open(DATA_FILE, 'rb') as f:
            loaded_data = pickle.load(f)
            train_x, train_y = loaded_data
    except (pickle.UnpicklingError, FileNotFoundError, EOFError) as e:
        messagebox.showerror("Error Loading Data", f"Error loading data: {e}")

# Function to add training data
def submit_data():
    if len(entry.get()) != 1:
        messagebox.showerror("Error", "The expected value is between 0 and 9.")
    else:
        try:
            int(entry.get())
        except:
            messagebox.showerror("Error", "The expected value is between 0 and 9.")
            return
        
        global drawn_image, train_x, train_y
        digit = int(entry.get())
        # Resize the image to 28x28 pixels
        image = drawn_image.resize((28, 28))
        # Convert the image to grayscale
        image = image.convert('L')
        # Convert the image to a numpy array
        image_array = np.array(image)
        # Normalize pixel values to the [0, 1] range
        image_array = image_array / 255.0
        # Invert pixel values (since the model is trained to assume the background is dark)
        image_array = 1.0 - image_array
        # Add an extra batch dimension (since the model expects a batch of images)
        train_x = np.append(train_x, [image_array], axis=0)
        train_y = np.append(train_y, [digit], axis=0)
        print(len(train_x))
        
        canvas.delete("all")
        entry.delete(0, tk.END)
        
        # Clear the drawn image
        drawn_image = Image.new("L", (250, 250), "white")
        
        save_data()

# Step 2: Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Function to train the model
def train_model():
    global train_x, train_y
    if len(train_x) == 0:
        canvas.delete("all")
        messagebox.showerror("Error", "No training data submitted.")
        return
    model.fit(train_x, train_y, epochs=20)

# Function to preprocess the drawn image
def preprocess_image(image):
    # Resize the image to 28x28 pixels (same size as the MNIST dataset)
    image = image.resize((28, 28))
    # Convert the image to grayscale
    image = image.convert('L')
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize pixel values to the [0, 1] range
    image_array = image_array / 255.0
    # Invert pixel values (since the model is trained to assume the background is dark)
    image_array = 1.0 - image_array
    # Add an extra batch dimension (since the model expects a batch of images)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to make predictions with the model
def predict_number():
    global drawn_image
    if len(train_x) == 0:
        drawn_image = Image.new("L", (250, 250), "white")
        canvas.delete("all")
        messagebox.showerror("Error", "The model has not been trained yet.")
        return
    # Preprocess the drawn image
    processed_img = preprocess_image(drawn_image)
    # Make a prediction using the model
    prediction = model.predict(processed_img)
    # Get the predicted number
    predicted_number = np.argmax(prediction)
    canvas.delete("all")
    messagebox.showinfo("Predicted Number", "The predicted number is: " + str(predicted_number))
    drawn_image = Image.new("L", (250, 250), "white")
    print(prediction)

# Create the graphical user interface for drawing and making predictions
root = tk.Tk()
root.title("Handwritten Digit Recognition")

canvas = tk.Canvas(root, width=250, height=250, bg='white')
canvas.pack()

# Button to submit training data
train_button = tk.Button(root, text="Submit Data", command=submit_data)
train_button.pack(side=tk.RIGHT, padx=5)

# Button to train the model
train_model_button = tk.Button(root, text="Train Model", command=train_model)
train_model_button.pack(side=tk.RIGHT, padx=5)

entry = tk.Entry(root)
entry.pack(side=tk.RIGHT, padx=5)

# Button to make a prediction
predict_button = tk.Button(root, text="Predict", command=predict_number)
predict_button.pack(side=tk.LEFT, padx=5)

drawing = False
prev_x, prev_y = None, None
drawn_image = Image.new("L", (250, 250), "white")

# Function to handle drawing on the canvas
def draw(event):
    global drawing, prev_x, prev_y, drawn_image
    x, y = event.x, event.y
    if drawing and prev_x is not None and prev_y is not None:
        canvas.create_line(prev_x, prev_y, x, y, width=11, fill='black')
        drawn_image_draw = ImageDraw.Draw(drawn_image)
        drawn_image_draw.line([prev_x, prev_y, x, y], width=11, fill='black')
    prev_x, prev_y = x, y

# Function to start drawing
def start_draw(event):
    global drawing
    drawing = True

# Function to stop drawing
def stop_draw(event):
    global drawing, prev_x, prev_y
    drawing = False
    prev_x, prev_y = None, None

canvas.bind("<B1-Motion>", draw)
canvas.bind("<Button-1>", start_draw)
canvas.bind("<ButtonRelease-1>", stop_draw)

root.mainloop()
