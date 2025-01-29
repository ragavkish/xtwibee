import cv2
import tensorflow as tf
import numpy as np
import time
import logging

MODEL_PATH = 'xtwibee.h5'
CASCADE_PATH = r'Z:/kizX/dataset/xtwibee/haarcascade/haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = r'Z:/kizX/dataset/xtwibee/haarcascade/haarcascade_eye.xml'
MOUTH_CASCADE_PATH = r'Z:/kizX/dataset/xtwibee/haarcascade/haarcascade_mcs_mouth.xml'

INPUT_SIZE = (48, 48)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model(path):
    try:
        model = tf.keras.models.load_model(path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit()

def preprocess_face(face):
    try:
        face = cv2.resize(face, INPUT_SIZE) / 255.0
        return np.reshape(face, (1, *INPUT_SIZE, 1))
    except Exception as e:
        logging.error(f"Error preprocessing face: {e}")
        return None

def draw_frame(frame, faces, predictions, eye_cascade, mouth_cascade):
    for (x, y, w, h), (emotion, confidence) in zip(faces, predictions):
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2
        cv2.circle(frame, center, radius, (36, 255, 12), 2, cv2.LINE_AA)

        roi_gray = frame[y:y + h, x:x + w]

        try:
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_center = (x + ex + ew // 2, y + ey + eh // 2)
                eye_radius = ew // 4
                cv2.circle(frame, eye_center, eye_radius, (255, 255, 255), -1, cv2.LINE_AA)
        except:
            pass

        try:
            mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=15, minSize=(40, 20))
            for (mx, my, mw, mh) in mouth[:1]:
                mouth_center = (x + mx + mw // 2, y + my + mh)
                cv2.ellipse(frame, mouth_center, (mw // 2, mh // 4), 0, 0, 180, (0, 0, 255), 2, cv2.LINE_AA)  # Red smiling mouth
        except:
            pass

        label_position = (x, y - 10 if y - 10 > 10 else y + 10)
        cv2.putText(frame, f"{emotion} ({confidence:.2f})", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

def detect_and_predict(frame, face_cascade, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    predictions = []

    for (x, y, w, h) in faces:
        face = preprocess_face(gray[y:y + h, x:x + w])
        if face is not None:
            prediction = model.predict(face, verbose=0)
            emotion_index = np.argmax(prediction)
            emotion = EMOTION_LABELS[emotion_index]
            confidence = np.max(prediction)
            predictions.append((emotion, confidence))
        else:
            predictions.append(("Unknown", 0.0))
    
    return faces, predictions

def main():
    model = load_model(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    mouth_cascade = cv2.CascadeClassifier(MOUTH_CASCADE_PATH)

    if face_cascade.empty() or eye_cascade.empty() or mouth_cascade.empty():
        logging.error("Failed to load one or more Haar cascade files. Check file paths.")
        exit()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Unable to access the webcam.")
        exit()

    logging.info("Press 'q' to exit.")

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to capture frame.")
                break

            faces, predictions = detect_and_predict(frame, face_cascade, model)
            draw_frame(frame, faces, predictions, eye_cascade, mouth_cascade)

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
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

if __name__ == "__main__":
    main()
