import sys, cv2
sys.path.append(".")
from src.detection import FaceDetector, draw_faces, save_capture, save_face_crops

det = FaceDetector()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam introuvable. Essaie l'index 1.")
    sys.exit(1)

print("s = sauvegarder la capture | v = sauvegarder les visages | q = quitter")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    faces = det.detect(frame)
    display = frame.copy()          # copie pour dessiner dessus
    draw_faces(display, faces)
    cv2.imshow("YuNet", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        print("Capture enregistrée :", save_capture(display))
    elif key == ord("v"):
        print("Visages enregistrés :", save_face_crops(frame, faces))
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()