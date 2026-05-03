import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from utils import load_data

# config
LEARNING_RATE = 0.001
EPOCHS = 40
RANDOM_SEED = 42
FONTSIZE_TITLE = 14
FONTSIZE_LABEL = 10

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# load data
X, y = load_data()
print("The shape of X is:", X.shape)
print("The shape of y is:", y.shape)
m = X.shape[0]

# preprocessing
X_cnn = X.reshape(-1, 20, 20, 1)
print("The shape of X_cnn is:", X_cnn.shape)

# visualize sample data
fig_sample, axes_sample = plt.subplots(8, 8, figsize=(6, 6))
fig_sample.tight_layout(pad=0.5, rect=[0, 0.03, 1, 0.91])
random_indices = np.random.randint(m, size=64)

for i, ax in enumerate(axes_sample.flat):
    ax.imshow(X_cnn[random_indices[i]].reshape(20, 20).T, cmap="gray")
    ax.set_title(y[random_indices[i], 0], fontsize=FONTSIZE_LABEL)
    ax.set_axis_off()

fig_sample.suptitle("Label & Image", fontsize=FONTSIZE_TITLE)
plt.show()

# split data
X_train, X_val, y_train, y_val = train_test_split(
    X_cnn, y.reshape(-1), test_size=0.2, random_state=RANDOM_SEED
)
print("X_train shape:", X_train.shape)
print("X_val shape:  ", X_val.shape)

m_val = len(X_val)
y_val_column = y_val.reshape(-1, 1)


# helper functions
def plot_training_history(history):
    """Plot training and validation accuracy and loss over epochs."""
    val_accuracy = history.history["val_accuracy"][-1] * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f"Validation Accuracy: {val_accuracy:.2f}%", fontsize=FONTSIZE_TITLE)
    plt.tight_layout()
    plt.show()


def display_errors_cnn(model, X, y):
    """Display up to 8 misclassified digits with true and predicted labels."""
    predictions = model.predict(X, verbose=0)
    yhat = np.argmax(predictions, axis=1)
    error_idx = np.where(yhat != y[:, 0])[0]

    if len(error_idx) == 0:
        print("No errors found.")
        return 0

    cnt = min(8, len(error_idx))
    fig_errors, axes_errors = plt.subplots(1, cnt, figsize=(10, 2))
    fig_errors.tight_layout(pad=0.5, rect=[0, 0.03, 1, 0.80])

    for i in range(cnt):
        j = error_idx[i]
        axes_errors[i].imshow(X[j].reshape(20, 20).T, cmap="gray")
        axes_errors[i].set_title(
            f"True: {y[j, 0]}\nPredicted: {yhat[j]}", fontsize=FONTSIZE_LABEL
        )
        axes_errors[i].set_axis_off()

    fig_errors.suptitle("Misclassified Digits", fontsize=FONTSIZE_TITLE)
    plt.show()
    return len(error_idx)


# build model
model = Sequential(
    [
        tf.keras.Input(shape=(20, 20, 1)),
        # Block 1: learn basic features (edges, corners)
        Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        # Block 2: learn complex features (curves, strokes)
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        # Flatten and classify
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)
model.summary()

# compile
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy"],
)

# train
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val))
plot_training_history(history)

# single prediction
random_index = np.random.randint(m_val)
sample = X_val[random_index]
prediction = model.predict(sample.reshape(1, 20, 20, 1), verbose=0)
yhat = np.argmax(prediction)
print(f"True label:      {y_val[random_index]}")
print(f"Predicted digit: {yhat}")
plt.imshow(sample.reshape(20, 20).T, cmap="gray")
plt.title(f"Label: {y_val[random_index]} | Predicted: {yhat}")
plt.axis("off")
plt.show()

# evaluate
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Loss:     {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
num_errors = display_errors_cnn(model, X_val, y_val_column)
print(f"Errors found: {num_errors}")

# batch visualization
fig_batch, axes_batch = plt.subplots(8, 8, figsize=(6, 6))
fig_batch.tight_layout(pad=0.5, rect=[0, 0.03, 1, 0.91])
random_indices = np.random.randint(m_val, size=64)
predictions = model.predict(X_val[random_indices], verbose=0)
yhats = np.argmax(predictions, axis=1)

for i, ax in enumerate(axes_batch.flat):
    ax.imshow(X_val[random_indices[i]].reshape(20, 20).T, cmap="gray")
    ax.set_title(f"{y_val[random_indices[i]]}, {yhats[i]}", fontsize=FONTSIZE_LABEL)
    ax.set_axis_off()

fig_batch.suptitle("True Label vs Predicted Label", fontsize=FONTSIZE_TITLE)
plt.show()

# save & verify model
os.makedirs("./models", exist_ok=True)
model.save("./models/handwritten_digit_recognition_cnn.keras")
print("Model saved!")

loaded_model = tf.keras.models.load_model(
    "./models/handwritten_digit_recognition_cnn.keras"
)
print("Model loaded successfully!")
