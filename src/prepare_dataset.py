import cv2
import numpy as np
import random
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
RAW = RACINE / 'data' / 'raw'
SORTIE = RACINE / 'data' / 'split'
MODELE = RACINE / 'models' / 'face_detection_yunet_2023mar.onnx'
RATIO_TRAIN = 0.8
COTE_MAX = 800
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

random.seed(42)

detecteur = cv2.FaceDetectorYN.create(str(MODELE), '', (320, 320), 0.8, 0.3, 5000)


def lire(chemin):
    donnees = np.fromfile(str(chemin), dtype=np.uint8)
    return cv2.imdecode(donnees, cv2.IMREAD_COLOR)


def ecrire(chemin, image):
    ok, tampon = cv2.imencode('.jpg', image)
    if ok:
        tampon.tofile(str(chemin))


def reduire(image):
    h, w = image.shape[:2]
    echelle = COTE_MAX / max(h, w)
    if echelle < 1:
        image = cv2.resize(image, (int(w * echelle), int(h * echelle)))
    return image


def nombre_visages(image):
    h, w = image.shape[:2]
    detecteur.setInputSize((w, h))
    _, visages = detecteur.detect(image)
    return 0 if visages is None else len(visages)


def main():
    rapport = []
    for dossier in sorted(RAW.iterdir()):
        if not dossier.is_dir():
            continue
        nom = dossier.name
        valides = []
        rejetes = []
        for fichier in sorted(dossier.iterdir()):
            if fichier.suffix.lower() not in EXTENSIONS:
                continue
            image = lire(fichier)
            if image is None:
                rejetes.append((fichier.name, 'illisible'))
                continue
            image = reduire(image)
            n = nombre_visages(image)
            if n != 1:
                rejetes.append((fichier.name, str(n) + ' visage(s) detecte(s)'))
                continue
            valides.append(image)

        random.shuffle(valides)
        if nom == 'inconnu':
            parties = {'test': valides}
        else:
            k = int(len(valides) * RATIO_TRAIN)
            parties = {'train': valides[:k], 'test': valides[k:]}

        for partie, images in parties.items():
            cible = SORTIE / partie / nom
            cible.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(images, start=1):
                ecrire(cible / (nom + '_' + str(i).zfill(3) + '.jpg'), img)

        rapport.append((nom, len(valides), len(rejetes)))
        print('==', nom, ':', len(valides), 'photos gardees,', len(rejetes), 'rejetees')
        for nom_fichier, raison in rejetes:
            print('   rejetee :', nom_fichier, '->', raison)

    print()
    print('Resume (personne, gardees, rejetees) :')
    for ligne in rapport:
        print(ligne)


if __name__ == '__main__':
    main()