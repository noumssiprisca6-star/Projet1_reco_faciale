import sys, cv2
sys.path.append(".")
from src.detection import FaceDetector, draw_faces

img = cv2.imread(sys.argv[1])
if img is None:
    print("Image introuvable, vérifie le chemin :", sys.argv[1])
    sys.exit(1)

det = FaceDetector()
faces = det.detect(img)
print(f"{len(faces)} visage(s) détecté(s)")
for f in faces:
    print("Boîte :", f["box"], "| Score :", round(f["score"], 2))

draw_faces(img, faces)
cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()