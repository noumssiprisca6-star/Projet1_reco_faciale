import os
import urllib.request

import cv2
import numpy as np
import time  


def save_capture(image, folder="captures", prefix="capture"):
    """Enregistre l'image complète (avec ou sans dessin) dans un dossier."""
    os.makedirs(folder, exist_ok=True)
    name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(folder, name)
    cv2.imwrite(path, image)
    return path


def save_face_crops(image, faces, folder="captures/visages", prefix="visage"):
    """Enregistre chaque visage détecté, recadré, dans un dossier."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for i, face in enumerate(faces):
        x, y, w, h = face["box"]
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        path = os.path.join(folder, f"{prefix}_{stamp}_{i}.jpg")
        cv2.imwrite(path, crop)
        paths.append(path)
    return paths

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "yunet.onnx")


def ensure_model(path=MODEL_PATH):
    """Télécharge le modèle YuNet s'il n'existe pas encore."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Téléchargement de YuNet...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Modèle enregistré :", path)
    return path


class FaceDetector:
    def __init__(self, model_path=MODEL_PATH, score_threshold=0.7,
                 nms_threshold=0.3, top_k=5000):
        ensure_model(model_path)
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320),
            score_threshold, nms_threshold, top_k
        )

    def detect(self, image, taille_min=0):
        """
        image : tableau NumPy BGR (lu par cv2.imread ou la webcam).
        Retourne une liste de dictionnaires (vide si aucun visage) :
          {
            "box": (x, y, w, h),
            "landmarks": [(x, y) x 5],  # oeil droit, oeil gauche, nez,
                                         # coin bouche droit, coin bouche gauche
            "score": float,
            "raw": ndarray de 15 valeurs   # à passer tel quel à SFace
          }
        """
        if image is None:
            return []
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))   # indispensable : taille de CETTE image
        _, faces = self.detector.detect(image)
        if faces is None:
            return []

        results = []
        for f in faces:
            x, y, bw, bh = f[0:4]
            # on garde la boîte à l'intérieur de l'image
            x1, y1 = max(int(x), 0), max(int(y), 0)
            x2, y2 = min(int(x + bw), w), min(int(y + bh), h)
            if (x2 - x1) < taille_min or (y2 - y1) < taille_min:
                            continue                      # visage trop petit : on l'ignore
            landmarks = [(int(f[4 + 2 * i]), int(f[5 + 2 * i])) for i in range(5)]
            results.append({
                "box": (x1, y1, x2 - x1, y2 - y1),
                "landmarks": landmarks,
                "score": float(f[14]),
                "raw": f,
            })
        return results
    
def plus_grand_visage(faces):
    """Parmi une liste de visages (sortie de detect), retourne le plus grand ; None si vide."""
    if not faces:
        return None
    return max(faces, key=lambda f: f["box"][2] * f["box"][3])


def draw_faces(image, faces, color=(0, 255, 0), label=None):
    """Dessine bounding box, 5 points et score sur l'image (modifie l'image)."""
    for face in faces:
        x, y, w, h = face["box"]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        for (lx, ly) in face["landmarks"]:
            cv2.circle(image, (lx, ly), 3, (0, 0, 255), -1)
        text = label if label else f"{face['score']:.2f}"
        cv2.putText(image, text, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image