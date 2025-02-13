import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import shutil

processed_train_dir = r"Z:/kizX/dataset/xtwibee/set-2/processed/train"
processed_test_dir = r"Z:/kizX/dataset/xtwibee/set-2/processed/test"
train_labels_file = r"Z:/kizX/dataset/xtwibee/set-2/train_labels.csv"
test_labels_file = r"Z:/kizX/dataset/xtwibee/set-2/test_labels.csv"
model_path = r"Z:/kizX/projectz/xtwibee/xtwibee.h5"
backup_model_path = r"Z:/kizX/projectz/xtwibee/xtwibee_backup.h5"
fine_tuned_model_path = r"Z:/kizX/projectz/xtwibee/xtwibee_finetuned.h5"

shutil.copy(model_path, backup_model_path)
print(f"Backup created at {backup_model_path}")

def load_data(data_dir, label_file):
    file_path = os.path.join(data_dir, "data.npy")
    print(f"Loading data from: {file_path}")
    data = np.load(file_path)
    labels_df = pd.read_csv(label_file)
    labels = labels_df['label'].values
    return data, labels

train_data, train_labels = load_data(processed_train_dir, train_labels_file)
test_data, test_labels = load_data(processed_test_dir, test_labels_file)

train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

label_encoder = LabelEncoder()
all_labels = np.concatenate((train_labels, test_labels), axis=0)
label_encoder.fit(all_labels)

train_labels_encoded = label_encoder.transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

num_classes = len(np.unique(all_labels))
train_labels_onehot = to_categorical(train_labels_encoded, num_classes)
test_labels_onehot = to_categorical(test_labels_encoded, num_classes)

X_train, X_val, y_train, y_val = train_test_split(
    train_data, train_labels_onehot, test_size=0.2, random_state=42
)

model = load_model(model_path)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=15,
    callbacks=[early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(test_data, test_labels_onehot, verbose=1)
print(f"Updated Test Accuracy: {test_accuracy:.2f}")

model.save(fine_tuned_model_path)
print(f"Fine-tuned model saved at {fine_tuned_model_path}")

shutil.copy(fine_tuned_model_path, model_path)
print(f"Original model updated with fine-tuned model at {model_path}")
