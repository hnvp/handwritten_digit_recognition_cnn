# Handwritten Digit Recognition using CNN

A convolutional neural network (CNN) model for recognizing handwritten digits (0-9) built with TensorFlow/Keras.

Results:

Validation Accuracy: ~98%

Validation Loss: ~0.08

Epochs: 40


Project Structure:

automated_handwritten_digit_recognition/

├── data/

│   ├── X.npy            # input images (5000, 400)

│   └── y.npy            # labels (5000, 1)

├── models.py            # main training script

├── utils.py             # helper functions

├── deeplearning.mplstyle

└── README.md


Model Architecture:

Input (20, 20, 1)

│

├── Conv2D(32, 3x3, relu) + MaxPooling(2x2) + Dropout(0.3)

│      Block 1: learn basic features (edges, corners)

│

├── Conv2D(64, 3x3, relu) + MaxPooling(2x2) + Dropout(0.3)

│      Block 2: learn complex features (curves, strokes)

│

├── Flatten

├── Dense(128, relu) + Dropout(0.5)

└── Dense(10, softmax)