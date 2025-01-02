import cv2
import tensorflow as tf
import numpy as np
import time
import logging

MODEL_PATH = 'xtwibee.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit()

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    logging.error("Failed to load Haar cascade. Check the file path.")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Error: Unable to access the webcam.")
    exit()

logging.info("Press 'q' to exit.")

def preprocess_face(face):
    face = cv2.resize(face, (48, 48)) / 255.0
    return np.reshape(face, (1, 48, 48, 1))

try:
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = preprocess_face(face)

            prediction = model.predict(face, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]
            confidence = np.max(prediction)

            label_position = (x, y - 10 if y - 10 > 10 else y + 10)
            cv2.putText(frame, f"{emotion} ({confidence:.2f})", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (36, 255, 12), 2)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('xtwibee', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    logging.error(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Application closed.")
