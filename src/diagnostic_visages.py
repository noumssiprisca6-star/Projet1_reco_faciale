import cv2
import numpy as np
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = RACINE / 'data' / 'raw' / 'bias'
MODELE = RACINE / 'models' / 'face_detection_yunet_2023mar.onnx'
COTE_MAX = 800

detecteur = cv2.FaceDetectorYN.create(str(MODELE), '', (320, 320), 0.3, 0.3, 5000)

for fichier in sorted(DOSSIER.iterdir()):
    donnees = np.fromfile(str(fichier), dtype=np.uint8)
    image = cv2.imdecode(donnees, cv2.IMREAD_COLOR)
    if image is None:
        print(fichier.name, ': illisible')
        continue
    h, w = image.shape[:2]
    echelle = COTE_MAX / max(h, w)
    if echelle < 1:
        image = cv2.resize(image, (int(w * echelle), int(h * echelle)))
        h, w = image.shape[:2]
    detecteur.setInputSize((w, h))
    _, visages = detecteur.detect(image)
    if visages is None:
        print(fichier.name, '(', w, 'x', h, ') : aucun visage, meme avec seuil 0.3')
    else:
        print(fichier.name, '(', w, 'x', h, ') :', len(visages), 'visage(s), score max', round(float(visages[:, -1].max()), 2))