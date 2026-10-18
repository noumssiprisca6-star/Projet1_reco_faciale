import cv2
import sys
import os

# Nos modules (dans le même dossier "src/" : lancez "python src/main.py ...").
from detection import FaceDetector, draw_faces, plus_grand_visage
import recognition

# ======================================================================
# RÉGLAGES
# ======================================================================
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_PHOTOS = os.path.join(RACINE, "data", "raw")   # data/raw/<nom_personne>/*.jpg

SEUIL_DETECTION = 0.9     # confiance minimale de YuNet
TAILLE_MIN_VISAGE = 80    # largeur et hauteur minimales de la boîte (pixels)
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

VERT = (0, 255, 0)        # couleurs BGR
ROUGE = (0, 0, 255)

AIDE = """Utilisation :
  python src/main.py build                      reconstruit la base depuis data/raw
  python src/main.py enroll <nom> <dossier>     ajoute une personne
  python src/main.py image <photo.jpg>          identifie les visages d'une photo
  python src/main.py webcam                     reconnaissance en direct (q, d)"""


# ======================================================================
# ENRÔLEMENT
# ======================================================================
def creer_detecteur():
    """Crée le détecteur YuNet (le modèle est téléchargé s'il manque)."""
    return FaceDetector(score_threshold=SEUIL_DETECTION)


def collecter_detections(detecteur, dossier_personne, nom):
    """
    Détecte le visage principal de chaque photo d'un dossier.

    Retourne une liste [{"image": chemin_absolu, "raw": 15 nombres}, ...] prête pour
    recognition.construire_base().
    Règles : on garde le PLUS GRAND visage ; on ignore les visages trop petits ;
    on signale les photos à plusieurs visages (cause fréquente de base polluée).
    """
    resultats = []
    for fichier in sorted(os.listdir(dossier_personne)):
        if not fichier.lower().endswith(EXTENSIONS):
            continue
        chemin = os.path.join(dossier_personne, fichier)
        image = cv2.imread(chemin)
        if image is None:
            print(f"  [ignorée] {nom}/{fichier} : illisible")
            continue

        visages = detecteur.detect(image, taille_min=TAILLE_MIN_VISAGE)
        if len(visages) == 0:
            print(f"  [ignorée] {nom}/{fichier} : aucun visage assez grand")
            continue
        if len(visages) > 1:
            print(f"  [attention] {nom}/{fichier} : {len(visages)} visages, on garde le "
                  f"plus grand (vérifiez que c'est bien {nom})")

        visage = plus_grand_visage(visages)
        resultats.append({"image": os.path.abspath(chemin), "raw": visage["raw"]})
    return resultats


def mode_build(dossier_photos=DOSSIER_PHOTOS):
    """Reconstruit toute la base à partir d'un dossier contenant un sous-dossier par personne."""
    if not os.path.isdir(dossier_photos):
        print(f"Dossier de photos introuvable : {dossier_photos}")
        return
    detecteur = creer_detecteur()
    detections = {}
    for nom in sorted(os.listdir(dossier_photos)):
        dossier = os.path.join(dossier_photos, nom)
        if os.path.isdir(dossier):
            liste = collecter_detections(detecteur, dossier, nom)
            if liste:
                detections[nom] = liste
    recognition.construire_base(detections)                  # fusionner=False : tout reconstruire
    #recognition.verifier_visage(recognition.charger_base())    # alerte si photos suspectes


def mode_enroll(nom, dossier):
    """Ajoute (ou remplace) UNE personne sans effacer les autres."""
    if not os.path.isdir(dossier):
        print(f"Dossier introuvable : {dossier}")
        return
    detecteur = creer_detecteur()
    liste = collecter_detections(detecteur, dossier, nom)
    if not liste:
        print("Aucune photo exploitable : rien n'a été enregistré.")
        return
    recognition.construire_base({nom: liste}, fusionner=True)
   # recognition.verifier_base(recognition.charger_base(), )


# ======================================================================
# TEMPS RÉEL
# ======================================================================
def traiter_image(detecteur, image, base, debug=False):
    """
    Détecte, identifie et dessine. Retourne (image_annotee, resultats).
    resultats = liste de (nom, score, statut), un par visage identifié.
    """
    affichage = image.copy()      # on dessine sur la COPIE : l'original reste propre pour SFace
    resultats = []

    for visage in detecteur.detect(image, taille_min=TAILLE_MIN_VISAGE):
        nom, score, statut = recognition.identifier_visage(image, visage, base)
        resultats.append((nom, score, statut))

        couleur = ROUGE if statut == recognition.INCONNU else VERT
        draw_faces(affichage, [visage], color=couleur, label=f"{statut} ({score:.2f})")

        if debug:
            classement = recognition.classer_personnes(image, visage, base)
            print("Scores :", [(n, round(s, 2)) for n, s in classement], "->", statut)
    return affichage, resultats


def charger_base_ou_construire():
    """Charge la base ; si elle n'existe pas encore, tente de la construire depuis data/raw."""
    base = recognition.charger_base()
    if not base and os.path.isdir(DOSSIER_PHOTOS):
        print("Aucune base trouvée : construction depuis", DOSSIER_PHOTOS)
        mode_build()
        base = recognition.charger_base()
    return base


def mode_image(chemin_photo):
    image = cv2.imread(chemin_photo)
    if image is None:
        print(f"Image introuvable ou illisible : {chemin_photo}")
        return
    base = charger_base_ou_construire()
    affichage, resultats = traiter_image(creer_detecteur(), image, base, debug=True)
    if not resultats:
        print("Aucun visage détecté (ou trop petit).")
    for i, (nom, score, statut) in enumerate(resultats, start=1):
        print(f"Visage {i} : {statut} (plus proche : {nom}, score {score:.2f})")
    cv2.imshow("Resultat (appuyez sur une touche)", affichage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mode_webcam():
    base = charger_base_ou_construire()
    if not base:
        print("Base vide : tout le monde sera inconnu.")
    else:
        print("Personnes dans la base :", list(base.keys()))

    detecteur = creer_detecteur()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return
    print("Touches : q = quitter, d = afficher/masquer les scores dans la console.")
    debug = False

    while True:
        ok, image = camera.read()
        if not ok:
            print("Erreur de lecture de la webcam.")
            break

        affichage, _ = traiter_image(detecteur, image, base, debug)
        cv2.imshow("Reconnaissance faciale (q pour quitter)", affichage)

        touche = cv2.waitKey(1) & 0xFF
        if touche == ord("q"):
            break
        if touche == ord("d"):
            debug = not debug

    camera.release()
    cv2.destroyAllWindows()


# ======================================================================
# POINT D'ENTRÉE
# ======================================================================
def main(args=None):
    if args is None:
        args = sys.argv[1:]
    try:
        if len(args) == 1 and args[0] == "build":
            mode_build()
        elif len(args) == 3 and args[0] == "enroll":
            mode_enroll(args[1], args[2])
        elif len(args) == 2 and args[0] == "image":
            mode_image(args[1])
        elif len(args) == 1 and args[0] == "webcam":
            mode_webcam()
        else:
            print(AIDE)
    except (FileNotFoundError, ValueError) as erreur:
        print(f"Erreur : {erreur}")


if __name__ == "__main__":
    main()
