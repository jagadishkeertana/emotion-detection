# collect_emotions.py
import cv2, os
labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
out_root = "mydata"
os.makedirs(out_root, exist_ok=True)
for l in labels:
    os.makedirs(os.path.join(out_root,l), exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press keys 0..6 to save frame to class index 0..6 (0=Angry, 6=Neutral). Press q to quit.")
while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow("Collect (press 0..6)", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if ord('0') <= k <= ord('6'):
        cls = k - ord('0')
        fname = os.path.join(out_root, labels[cls], f"{int(cv2.getTickCount())}.jpg")
        cv2.imwrite(fname, frame)
        print("Saved", fname)
cap.release(); cv2.destroyAllWindows()
