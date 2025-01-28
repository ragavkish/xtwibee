import os
import cv2
import numpy as np

train_dir = r"Z:/kizX/dataset/xtwibee/set-2/train"
test_dir = r"Z:/kizX/dataset/xtwibee/set-2/test"

image_size = (48, 48)
output_dir = r"Z:/kizX/dataset/xtwibee/set-2/processed"

def preprocess_images(input_dir, output_subdir, image_size):
    data = []
    labels = []
    
    output_path = os.path.join(output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        print(f"Processing label: {label}")
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, image_size)
                image = image / 255.0
                
                data.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    data = np.array(data, dtype="float32").reshape(-1, image_size[0], image_size[1], 1)
    labels = np.array(labels)

    np.save(os.path.join(output_path, "data.npy"), data)
    np.save(os.path.join(output_path, "labels.npy"), labels)
    print(f"Saved processed data to {output_path}")

preprocess_images(train_dir, "train", image_size)
preprocess_images(test_dir, "test", image_size)
