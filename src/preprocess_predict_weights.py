# preprocess_predict_weights.py
import os, sys
import cv2
import numpy as np
import h5py
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

MODEL_WEIGHTS = "model.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
EMOTION_LABELS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def build_model(input_shape=(48,48,1), n_classes=7):
    inp = Input(shape=input_shape, name='input_1')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_1')(inp)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_2')(x)
    x = MaxPooling2D((2,2), name='max_pooling2d_1')(x)
    x = Dropout(0.25, name='dropout_1')(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv2d_3')(x)
    x = MaxPooling2D((2,2), name='max_pooling2d_2')(x)
    x = Dropout(0.25, name='dropout_2')(x)

    x = Conv2D(256, (3,3), activation='relu', padding='same', name='conv2d_4')(x)
    x = MaxPooling2D((2,2), name='max_pooling2d_3')(x)
    x = Flatten(name='flatten_1')(x)
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_3')(x)
    out = Dense(n_classes, activation='softmax', name='dense_2')(x)

    model = Model(inputs=inp, outputs=out)
    return model

def try_load_weights(model, weights_path):
    # try direct load_weights (most likely will work if shapes match)
    try:
        model.load_weights(weights_path)
        print("[INFO] Loaded weights with model.load_weights()")
        return model
    except Exception as e:
        print("[INFO] Direct load_weights failed:", e)
        print("[INFO] Attempting manual load + transpose heuristics...")

    # manual load: open h5 and set weights by layer name
    with h5py.File(weights_path, 'r') as f:
        for layer in model.layers:
            name = layer.name
            if name in f:
                grp = f[name]
                arrays = {}
                for k in grp.keys():
                    arrays[k] = np.array(grp[k])
                # Heuristics to pick kernel and bias
                kernel = None; bias = None
                # find 4-d array -> kernel
                for v in arrays.values():
                    if v.ndim == 4:
                        kernel = v; break
                # find 1-d array -> bias
                for v in arrays.values():
                    if v.ndim == 1:
                        bias = v; break
                # If kernel looks like (out,in,kh,kw), transpose to (kh,kw,in,out)
                if kernel is not None and kernel.ndim == 4:
                    if kernel.shape[0] == getattr(layer, 'filters', None):
                        # file is (out,in,kh,kw)
                        kernel = np.transpose(kernel, (2,3,1,0))
                        print(f"[INFO] Transposed kernel for layer {name}")
                    # else, try to detect other order - we assume now it's correct
                weights_list = []
                if kernel is not None:
                    weights_list.append(kernel)
                if bias is not None:
                    weights_list.append(bias)
                if weights_list:
                    try:
                        layer.set_weights(weights_list)
                        print(f"[INFO] Set weights for layer {name}")
                    except Exception as ex:
                        print(f"[WARN] Could not set weights for {name}: {ex}")
    return model

def align_and_clahe(face_bgr, desired_size=(48,48)):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    out = clahe.apply(gray)
    out = cv2.resize(out, desired_size)
    out = out.astype('float32') / 255.0
    out = np.expand_dims(out, axis=(0,-1))  # shape (1,H,W,1)
    return out

def detect_face_mtcnn(img):
    detector = MTCNN()
    res = detector.detect_faces(img)
    if not res:
        return None
    # choose largest box
    res = sorted(res, key=lambda r: r['box'][2]*r['box'][3], reverse=True)[0]
    x,y,w,h = res['box']
    x,y = max(0,x), max(0,y)
    # simple rotate to align eyes if keypoints present
    keypoints = res.get('keypoints',{})
    if 'left_eye' in keypoints and 'right_eye' in keypoints:
        left = keypoints['left_eye']; right = keypoints['right_eye']
        dy = right[1]-left[1]; dx = right[0]-left[0]
        angle = np.degrees(np.arctan2(dy, dx))
        cx, cy = x + w//2, y + h//2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        # recompute crop approximately
    face = img[y:y+h, x:x+w].copy()
    return face, (x,y,w,h)

def annotate_and_save(img, box, topk, outpath):
    x,y,w,h = box
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    label = f"{topk[0][0]} ({topk[0][1]*100:.1f}%)"
    cv2.putText(img,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    # add second and third lines
    for i,(emo,prob) in enumerate(topk[1:3], start=1):
        cv2.putText(img, f"{emo}: {prob*100:.1f}%", (x, y-10 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
    cv2.imwrite(outpath, img)
    print("[INFO] Saved annotated image:", outpath)

def main():
    if len(sys.argv) < 2:
        print("Usage: py preprocess_predict_weights.py <path-to-image>")
        sys.exit(1)
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print("Image not found:", img_path); sys.exit(1)
    if not os.path.exists(MODEL_WEIGHTS):
        print("model.h5 not found in current folder. Place weights file here."); sys.exit(1)

    img = cv2.imread(img_path)
    if img is None:
        print("Failed to open image:", img_path); sys.exit(1)

    det_res = detect_face_mtcnn(img)
    if det_res is None:
        print("No face detected in image.")
        sys.exit(0)
    face, box = det_res

    # preprocess
    inp = align_and_clahe(face, desired_size=(48,48))

    # build model and load weights
    model = build_model()
    model = try_load_weights(model, MODEL_WEIGHTS)

    # predict
    preds = model.predict(inp, verbose=0)[0]
    sorted_idx = np.argsort(preds)[::-1]
    topk = [(EMOTION_LABELS[i], float(preds[i])) for i in sorted_idx[:3]]
    print("Top-3:", topk)

    # save annotated image
    outp = os.path.splitext(img_path)[0] + "_annot.png"
    annotate_and_save(img, box, topk, outp)

if __name__ == "__main__":
    main()
