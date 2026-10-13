import sys, cv2
sys.path.append(".")
from src.detection import FaceDetector, draw_faces

det = FaceDetector()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW = plus fiable sous Windows

if not cap.isOpened():
    print("Webcam introuvable. Essaie l'index 1 : cv2.VideoCapture(1, cv2.CAP_DSHOW)")
    sys.exit(1)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    draw_faces(frame, det.detect(frame))
    cv2.imshow("YuNet - q pour quitter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()