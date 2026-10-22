# preprocess_and_predict.py
import sys, os
import cv2, numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

MODEL = "model.h5"
IMG = sys.argv[1] if len(sys.argv)>1 else "file:///mnt/data/7df4eda5-0fd9-42e1-a4a5-37fdcacf658f.png".replace("file://","")

detector = MTCNN()

def align_crop(img, box, keypoints):
    x,y,w,h = box
    x,y = max(0,x), max(0,y)
    # rotate to make eyes horizontal
    left = keypoints.get('left_eye')
    right = keypoints.get('right_eye')
    if left and right:
        dy = right[1]-left[1]; dx = right[0]-left[0]
        angle = np.degrees(np.arctan2(dy, dx))
        cx, cy = x + w//2, y + h//2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        # recompute box after rotation (approx)
    face = img[y:y+h, x:x+w].copy()
    return face

def clahe_preprocess(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    out = clahe.apply(gray)
    out = cv2.resize(out, (48,48))
    out = out.astype('float32')/255.0
    out = out.reshape(1,48,48,1)
    return out

img = cv2.imread(IMG)
if img is None:
    print("Could not open:", IMG); sys.exit(1)

faces = detector.detect_faces(img)
print("Detected faces:", len(faces))
if not faces:
    sys.exit(0)

faces = sorted(faces, key=lambda f: f['box'][2]*f['box'][3], reverse=True)
f = faces[0]
face = align_crop(img, f['box'], f['keypoints'])
inp = clahe_preprocess(face)

print("Loading model:", MODEL)
model = load_model(MODEL)
preds = model.predict(inp)[0]
labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
top3 = sorted([(labels[i], float(preds[i])) for i in range(len(labels))], key=lambda x:-x[1])[:3]
print("Top-3:", top3)
