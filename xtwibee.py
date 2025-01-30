import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import time
import logging

MODEL_PATH = 'xtwibee.h5'
INPUT_SIZE = (48, 48)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

FACIAL_LANDMARKS = {
    "left_eye": [33, 160, 158, 133, 153, 144],
    "right_eye": [362, 385, 387, 263, 373, 380],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191]
}

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

def draw_landmarks(frame, landmarks, color):
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)

def detect_and_predict(frame, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    predictions = []
    faces = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            face_points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]
            
            x_min = min([p[0] for p in face_points])
            y_min = min([p[1] for p in face_points])
            x_max = max([p[0] for p in face_points])
            y_max = max([p[1] for p in face_points])
            
            face = preprocess_face(cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY))
            
            if face is not None:
                prediction = model.predict(face, verbose=0)
                emotion_index = np.argmax(prediction)
                emotion = EMOTION_LABELS[emotion_index]
                confidence = np.max(prediction)
                predictions.append((emotion, confidence))
            else:
                predictions.append(("Unknown", 0.0))
                
            faces.append((x_min, y_min, x_max - x_min, y_max - y_min, face_points))
    
    return faces, predictions

def main():
    model = load_model(MODEL_PATH)
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
            
            faces, predictions = detect_and_predict(frame, model)
            
            for (x, y, w, h, landmarks), (emotion, confidence) in zip(faces, predictions):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
                label_position = (x, y - 10 if y - 10 > 10 else y + 10)
                cv2.putText(frame, f"{emotion} ({confidence:.2f})", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
                
                draw_landmarks(frame, [landmarks[i] for i in FACIAL_LANDMARKS["left_eye"]], (255, 255, 255))
                draw_landmarks(frame, [landmarks[i] for i in FACIAL_LANDMARKS["right_eye"]], (255, 255, 255))
                draw_landmarks(frame, [landmarks[i] for i in FACIAL_LANDMARKS["mouth"]], (0, 0, 255))
            
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
