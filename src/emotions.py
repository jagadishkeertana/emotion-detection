import argparse
import os
import sys
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------

MODEL_PATH = "model.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Order of emotions – may depend on how the model was trained
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

# ----------------------------------------


def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file '{MODEL_PATH}' not found.")
        sys.exit(1)

    print("[INFO] Loading emotion model...")
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded.")
    return model


def load_face_detector():
    if not os.path.exists(CASCADE_PATH):
        print(f"[ERROR] Haar Cascade file '{CASCADE_PATH}' not found.")
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return face_cascade


def predict_emotion(model, face_gray):
    face_resized = cv2.resize(face_gray, (48, 48))
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=-1)
    face_resized = np.expand_dims(face_resized, axis=0)

    preds = model.predict(face_resized, verbose=0)[0]
    idx = int(np.argmax(preds))
    emotion = EMOTION_LABELS[idx]
    confidence = float(np.max(preds))
    return emotion, confidence


def run_display_mode():
    model = load_emotion_model()
    face_cascade = load_face_detector()

    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam not detected!")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_gray = gray[y:y + h, x:x + w]
            emotion, conf = predict_emotion(model, face_gray)
            label = f"{emotion} ({conf*100:.1f}%)"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection (Press Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited webcam.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="display")
    args = parser.parse_args()

    if args.mode == "display":
        run_display_mode()
    else:
        print("[ERROR] Only --mode display is supported.")


if __name__ == "__main__":
    main()
