import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('xtwibee.h5')

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        cv2.putText(frame, str(emotion), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (36,255,12), 2)

    cv2.imshow('xtwibee', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
