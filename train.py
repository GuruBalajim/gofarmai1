import os
import pandas as pd
import cv2
import random
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn import metrics

# Set the data directory
data_dir = "dataset"  # Update with your data directory
batch_size = 16
input_shape = (50, 50, 3)  # Image input shape
test_size = 0.1  # Test size

# Get subfolders (classes) in the data directory
subfolders = os.listdir(data_dir)

# Prepare image paths and class labels
data = []
for cls in subfolders:
    cls_dir = os.path.join(data_dir, cls)
    for img_name in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, img_name)
        data.append((img_path, cls))

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

# Create DataFrames for train and test data
train_df = pd.DataFrame(train_data, columns=["image_path", "class"])
test_df = pd.DataFrame(test_data, columns=["image_path", "class"])

# Define image preprocessing function
def preprocess_image(img_path):
    try:
        img_array = cv2.imread(img_path, 1)
        img_array = cv2.medianBlur(img_array, 1)
        new_array = cv2.resize(img_array, input_shape[:2])
        return new_array
    except Exception as e:
        return None

# Create training data
IMG_SIZE = input_shape[0]
training_data = []

for index, row in train_df.iterrows():
    img_path = row["image_path"]
    class_num = subfolders.index(row["class"])
    img_data = preprocess_image(img_path)
    if img_data is not None:
        training_data.append([img_data, class_num])

# Shuffle the training data
random.shuffle(training_data)

X = []  # Features
y = []  # Labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X*3)
y = np.array(y*3)

print("Image preprocessing completed.")

# Normalize the data
X = X / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Print the shape of the training and testing sets
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

# Building the CNN model
model = Sequential()
# 3 convolutional layers
model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2 hidden layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

# Output layer with 16 neurons (for 16 classes)
model.add(Dense(16))
model.add(Activation("softmax"))

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Training the model
history = model.fit(X, y, batch_size=32, epochs=15, validation_split=0.2)

# Evaluate model on training data
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f'Training accuracy: {train_acc:.4f}')

# Save the model
print("Saved model to disk")
model.save('CNN.model')  # You can also save it as a .h5 or .keras file

# Plot training and validation accuracy & loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(15)

plt.figure(figsize=(15, 15))

plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Predicting on the test set
y_pred = model.predict(X_test)

# Convert predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Print accuracy
accuracy = metrics.accuracy_score(y_test, y_pred_labels) * 100
print("Accuracy is:", accuracy)

# Confusion matrix
cm1 = confusion_matrix(y_test, y_pred_labels)

# Plot confusion matrix
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm1, annot=True, fmt='.1f', cmap="Blues", linewidths=0.01, ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred_labels))
