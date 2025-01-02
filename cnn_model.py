import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train_dir = r"Z:\kizX\dataset\xtwibee\processed\train"
test_dir = r"Z:\kizX\dataset\xtwibee\processed\test"
model_output_path = r"Z:\kizX\projectz\xtwibee\xtwibee.h5"

def load_data(data_dir):
    data = np.load(os.path.join(data_dir, "data.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    return data, labels

def prepare_data(data, labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, label_encoder

train_data, train_labels = load_data(train_dir)
test_data, test_labels = load_data(test_dir)

X_train, X_val, y_train, y_val, label_encoder = prepare_data(train_data, train_labels)

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (48, 48, 1)
num_classes = len(np.unique(train_labels))

model = build_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=25,
    verbose=1
)

test_labels_encoded = label_encoder.transform(test_labels)
test_labels_onehot = to_categorical(test_labels_encoded)
test_loss, test_accuracy = model.evaluate(test_data, test_labels_onehot, verbose=1)
print(f"Test Accuracy: {test_accuracy:.2f}")

os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
model.save(model_output_path)
print(f"Model saved at {model_output_path}")
