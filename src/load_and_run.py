# load_fix_and_run.py
import os, sys
import h5py
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

MODEL_WEIGHTS = "model.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
EMOTION_LABELS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def build_model(input_shape=(48,48,1), n_classes=7):
    inp = Input(shape=input_shape, name="input_1")
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

def inspect_h5(path):
    print("[INFO] Inspecting", path)
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        print("Top-level keys:", keys)
        # If groups exist, print their dataset names and shapes
        for k in keys:
            try:
                grp = f[k]
                subkeys = list(grp.keys())
                print(f" Group '{k}' contains: {subkeys}")
                for sk in subkeys:
                    ds = grp[sk]
                    print("   -", sk, "shape:", getattr(ds, 'shape', None))
            except Exception as e:
                print("  (couldn't list group)", e)

def load_weights_by_name_and_fix(model, h5_path):
    """Load weights from h5 file into the model, transposing conv kernels if needed."""
    print("[INFO] Opening weights file:", h5_path)
    with h5py.File(h5_path, 'r') as f:
        file_layers = list(f.keys())
        print("[INFO] Layers in H5:", file_layers)

        # Iterate model layers and try to load weights from file by matching layer name
        for layer in model.layers:
            name = layer.name
            if name in f:
                grp = f[name]
                print(f"[INFO] Loading for layer: {name}  keys in file group: {list(grp.keys())}")
                # collect arrays present (commonly 'kernel:0' and 'bias:0' or just 'weights' etc.)
                arrays = {}
                for k in grp.keys():
                    try:
                        arr = np.array(grp[k])
                    except Exception as e:
                        print("   - failed to read", k, e); continue
                    arrays[k] = arr

                # Heuristics: find kernel and bias names
                kernel = None
                bias = None
                # common names seen in many weight-only files:
                for candidate in ['kernel:0', 'W:0', 'weights', 'conv2d/kernel:0', '0']:
                    if candidate in arrays:
                        kernel = arrays[candidate]
                        break
                # fallback: any 4-d array is likely kernel
                if kernel is None:
                    for k,v in arrays.items():
                        if v.ndim == 4:
                            kernel = v
                            break
                for candidate in ['bias:0', 'b:0', 'bias', '1']:
                    if candidate in arrays:
                        bias = arrays[candidate]
                        break
                # fallback bias: any 1-d array
                if bias is None:
                    for k,v in arrays.items():
                        if v.ndim == 1:
                            bias = v
                            break

                # If kernel is found and layer expects conv2d, we may need to transpose
                try:
                    if hasattr(layer, 'kernel') or 'conv2d' in name:
                        # kernel present?
                        if kernel is None:
                            print("   - No kernel array found for conv layer", name)
                            continue
                        print("   - kernel shape in file:", kernel.shape, " layer expected kernel shape:", getattr(layer, 'kernel', None) and layer.kernel.shape or '(unknown)')
                        # If file kernel shape is (out, in, kh, kw) -> transpose to (kh, kw, in, out)
                        if kernel.ndim == 4:
                            # detect ordering: if first dim equals layer.filters or equals expected out channels?
                            # We will transpose (0,1,2,3) -> (2,3,1,0) if file appears as (out,in,kh,kw)
                            # If kernel shape first dim is small and 3rd dim is small etc we check
                            # We'll attempt both strategies safely.
                            # Candidate transpose from (out,in,kh,kw) to (kh,kw,in,out):
                            try_transpose = False
                            # if kernel.shape[0] == layer.output_shape[-1] or kernel.shape[0] == getattr(layer, 'filters', None):
                            try:
                                filters = getattr(layer, 'filters', None)
                                if filters is not None and kernel.shape[0] == filters:
                                    try_transpose = True
                            except Exception:
                                pass
                            if try_transpose:
                                kernel_t = np.transpose(kernel, (2,3,1,0))
                                kernel = kernel_t
                                print("   - Transposed kernel to", kernel.shape)
                            else:
                                # fallback: try to detect standard Keras ordering (kh,kw,in,out)
                                if kernel.shape[2] in (1,3) and kernel.shape[3] in (1,3,5,7):
                                    # perhaps already (kh,kw,in,out)
                                    print("   - Kernel seems already in KH,KW,IN,OUT ordering; not transposing.")
                                else:
                                    # try safe transpose and catch errors on set_weights
                                    kernel = np.transpose(kernel, (2,3,1,0))
                                    print("   - Attempted transpose (fallback); new shape", kernel.shape)

                        # prepare final weights list
                        weights_to_set = []
                        weights_to_set.append(kernel)
                        if bias is not None:
                            weights_to_set.append(bias)
                        else:
                            # some conv layers do not have bias; check layer expects bias
                            if layer.use_bias:
                                print("   - Warning: layer expects bias but none found in file")
                        # try to set weights
                        try:
                            layer.set_weights(weights_to_set)
                            print(f"   - Set weights on layer {name}")
                        except Exception as e:
                            print("   - Failed to set weights on layer", name, ":", e)
                    else:
                        # Non-conv e.g. Dense layer
                        if kernel is None and len(arrays) > 0:
                            # Try pick the largest array as kernel
                            cand = sorted(arrays.items(), key=lambda kv: kv[1].size, reverse=True)[0][1]
                            kernel = cand
                        if kernel is None:
                            continue
                        # For dense, the file-wise kernel might be transposed (out,in) vs (in,out)
                        if kernel.ndim == 2:
                            # check if transpose needed by comparing shapes
                            expected_shape = None
                            try:
                                expected_shape = layer.get_weights()[0].shape
                            except Exception:
                                pass
                            if expected_shape is not None and kernel.shape != expected_shape:
                                if kernel.T.shape == expected_shape:
                                    kernel = kernel.T
                                    print("   - Transposed dense kernel to", kernel.shape)
                        weights_to_set = [kernel]
                        if bias is not None:
                            weights_to_set.append(bias)
                        try:
                            layer.set_weights(weights_to_set)
                            print(f"   - Set weights on dense/other layer {name}")
                        except Exception as e:
                            print("   - Failed to set weights on dense/other layer", name, ":", e)
                except Exception as e:
                    print("   - Unexpected error while setting layer weights:", e)
            else:
                # layer name not found in file
                # print(f"[DEBUG] Layer {name} not found in H5 file")
                pass

def run_inference(model):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print("ERROR: failed to load cascade:", CASCADE_PATH)
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: webcam not available")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48)).astype('float32')/255.0
            face = np.expand_dims(face, -1)
            face = np.expand_dims(face, 0)
            preds = model.predict(face, verbose=0)[0]
            idx = int(np.argmax(preds))
            label = f"{EMOTION_LABELS[idx]} ({preds[idx]*100:.1f}%)"
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.imshow("Emotion (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    if not os.path.exists(MODEL_WEIGHTS):
        print("ERROR: model weights file not found:", MODEL_WEIGHTS)
        sys.exit(1)

    inspect_h5(MODEL_WEIGHTS)
    model = build_model()
    print("[INFO] Built model. Now loading/fixing weights...")
    load_weights_by_name_and_fix(model, MODEL_WEIGHTS)
    print("[INFO] Starting webcam inference...")
    run_inference(model)

if __name__ == "__main__":
    main()
