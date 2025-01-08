# handwritten-digit-recognition
 This project demonstrates a simple handwritten digit recognition system built to demonstrate in a simple way how a deep learning AI model works. All the code is in one unique python file and everything was made to be as intuitive as possible.

## Features

- Draw digits (0-9) on a canvas.
- Save your drawings and use them to train the model.
- Train the model to recognize your handwritten digits.
- Predict the digit of your drawing.

## Requirements

You need to install a few Python packages to run the project:

- `tensorflow`
- `numpy`
- `pillow`
- `tkinter`

To install the required packages, run:
- pip install tensorflow numpy pillow


**Note:** `tkinter` is usually included with Python, but if you don't have it, you may need to install it separately.

## How to Use

1. Run the program.
2. Draw a digit on the canvas.
3. Enter the digit you just drew in the input box (between 0 and 9).
4. Click "Submit Data" to save your drawing.
5. Train the model by clicking "Train Model" (it is recomended to send all the digits and several variations of each digit for better performance).
6. Once trained, click "Predict" to see the model's prediction of the digit you drew.

## How It Works

- You draw a digit on the canvas.
- The program saves your drawing and trains a model to recognize it.
- The model uses the training data to make predictions on your new drawings.
