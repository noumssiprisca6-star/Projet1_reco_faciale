"""
src/recognition.py
==================
Module RECONNAISSANCE : utilise SFace (cv2.FaceRecognizerSF).

Son travail : a  partir d'un visage deja detecte (par detection.py / YuNet),
  1. l'aligner et le recadrer en 112 x 112 pixels   -> alignCrop
  2. calculer son empreinte de 128 nombres          -> feature (SFace)
  3. la comparer a celles de la base                -> similarita cosinus (NumPy)
  4. decider : un nom, ou "Unknown"                 -> seuil 0.363

Ce module NE detecte PAS les visages et n'importe PAS detection.py :
il reçoit simplement ce que detection.py produit.
Modele requis :  models/face_recognition_sface_2021dec.onnx

"""
import itertools
import os
import pickle

import cv2
import numpy as np

# Dossier racine du projet = le dossier au-dessus de "src/".
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHEMIN_SFACE = os.path.join(RACINE, "models", "face_recognition_sface_2021dec.onnx")
CHEMIN_BASE = os.path.join(RACINE, "data", "embeddings.pkl")


# seuil de similarite demande par OpenCV
SEUIL_SIMILARITE = 0.45

# Écart minimal exigé entre le meilleur candidat et le deuxième (si >= 2 personnes).
MARGE_MIN = 0.08

#Message affiche lorsqu'une personne n'est pas reconnu
INCONNU = "UNKNOWN"

""" le chargement du model prend considerablement de temps, et plus lorsque l'operation est repetee
le modele sera charge par la fonction ci dessous une seule fois grace a "global", qui pemettra l'acces aux varible globale
ce qui evitera la suppression de la variable a la fin de l'execuction de cette fonction
"""
_sface = None

# ======================================================================
# 1. OUTILS DE BASE : chargement, Recuperation des nombres yunet, empreinte, similarite
# ======================================================================

# FONCTION DE CHARGEMENT DU MODEL
def _charger_sface():
    #Charge SFace au premier appel, puis le reutilise.
    global _sface
    if _sface is None:
        if not os.path.isfile(CHEMIN_SFACE):
            raise FileNotFoundError(f"Modèle introuvable : {CHEMIN_SFACE}")
        # Le second parametre ("") est un fichier de configuration, inutile ici.
        _sface = cv2.FaceRecognizerSF.create(CHEMIN_SFACE, "")
    return _sface

def _ligne_yunet(donnees):
    """Retourne les 15 nombres YuNet d'un visage ("raw" de detection.py, ou "ligne")."""
    if "raw" in donnees:
        return donnees["raw"]
    return donnees["ligne"]
    
def extraire_embedding(image, ligne):
    """
    Calcule l'empreinte (128 nombres) d'un visage.
    image : l'image complete (BGR) dans laquelle le visage a ete détecte.
    ligne : les 15 nombres bruts de YuNet pour ce visage.

    Retourne un tableau NumPy de forme (128,) et de longueur 1.
    """
    sface = _charger_sface()

    # Étape A - alignement : a partir des points cles, OpenCV redresse le visage
    # et le recadre en 112 x 112 pixels (taille attendue par SFace).
    visage_aligne = sface.alignCrop(image, np.asarray(ligne, dtype=np.float32))

    # Étape B - SFace calcule l'empreinte. Resultat de forme (1, 128).
    empreinte = sface.feature(visage_aligne)

    # Étape C - on aplatit en (128,) puis on NORMALISE (longueur = 1).
    # Grace a cela, comparer deux empreintes revient a un simple produit scalaire,
    # qui est exactement la similarité cosinus.
    empreinte = np.asarray(empreinte, dtype=np.float32).flatten()
    return empreinte / np.linalg.norm(empreinte)

def _similarite(empreinte_a, empreinte_b):
    """
    Similarite cosinus entre deux empreintes normalisees.
    Resultat entre -1 et 1 : plus c'est proche de 1, plus les visages se ressemblent.
    """
    return float(np.dot(empreinte_a, empreinte_b))

# ======================================================================
# 2. BASE D'EMPREINTES : charger, sauvegarder, construire, contrôler
# ======================================================================

def charger_base(chemin_base=CHEMIN_BASE):
    """
    Lit la base d'empreintes sur le disque.
    Retourne un dictionnaire vide si le fichier n'existe pas encore.
    """
    if not os.path.isfile(chemin_base):
        return {}
    with open(chemin_base, "rb") as f:        # "rb" = lecture binaire
        return pickle.load(f)


def sauvegarder_base(base, chemin_base=CHEMIN_BASE):
    """Ecrit la base d'empreintes sur le disque (cree le dossier data/ si besoin)."""
    os.makedirs(os.path.dirname(chemin_base), exist_ok=True)
    with open(chemin_base, "wb") as f:        # "wb" = écriture binaire
        pickle.dump(base, f)


def construire_base(detections, chemin_base=CHEMIN_BASE, fusionner=False):
    """
    Calcule les empreintes et sauvegarde la base.

    detections : {"alice": [visage, visage, ...], "bob": [...]}
                 Chaque visage est un dictionnaire avec "raw" (ou "ligne") et
                 "image" (chemin de la photo, relue ici pour calculer l'empreinte).
    fusionner  : False -> la base est reconstruite entièrement (remplace l'ancienne).
                 True  -> on part de la base existante ; seules les personnes
                          présentes dans `detections` sont remplacées.

    Retourne {nom: nombre_de_photos_utilisées}.
    """
    base = charger_base(chemin_base) if fusionner else {}
    resume = {}

    for nom, liste_visages in detections.items():
        empreintes = []
        for visage in liste_visages:
            chemin_image = visage.get("image")
            image = cv2.imread(chemin_image) if chemin_image else None
            if image is None:                      # photo déplacée ou illisible
                print(f"  [ignorée] image illisible : {chemin_image}")
                continue
            empreintes.append(extraire_embedding(image, _ligne_yunet(visage)))

        if len(empreintes) > 0:
            base[nom] = empreintes
            resume[nom] = len(empreintes)
            print(f"  [ok] {nom} : {len(empreintes)} empreinte(s)")
        else:
            print(f"  [ignorée] {nom} : aucune empreinte calculable")

    if len(resume) == 0:
        raise ValueError("Aucune empreinte calculée : la base n'a pas été écrite.")

    sauvegarder_base(base, chemin_base)
    print(f"Base enregistrée ({len(base)} personne(s)) -> {chemin_base}")
    return resume



# ======================================================================
# 3. TEMPS REEL : identification et vérification
# ======================================================================

def classer_personnes(image, donnees_detection, database):
    """
    Compare un visage à chaque personne de la base.

    Retourne la liste [(nom, meilleur_score), ...] triée du plus au moins ressemblant
    (liste vide si la base est vide). Pour chaque personne, on garde sa MEILLEURE photo.
    """
    if len(database) == 0:
        return []
    empreinte = extraire_embedding(image, _ligne_yunet(donnees_detection))
    scores = {nom: max(_similarite(empreinte, ref) for ref in refs)
              for nom, refs in database.items()}
    return sorted(scores.items(), key=lambda paire: paire[1], reverse=True)

def identifier_visage(image, donnees_detection, database, seuil=SEUIL_SIMILARITE,  marge=MARGE_MIN):
    """
    Identification 1:N : qui est ce visage parmi les personnes de la base ?

    image             : image complète (BGR) de la webcam.
    donnees_detection : UN visage fourni par detection.detecter_flux()
                        (dictionnaire contenant la cle "ligne").
    database          : base chargee avec charger_base().
    seuil             : score minimal pour accepter une identite.

    Retourne un tuple (nom, score, statut) :
        nom    : la personne la plus ressemblante de la base, meme si le score est
                 trop bas ("Unknown" si la base est vide)
        score  : sa similarite cosinus (-1.0 si la base est vide)
        statut : le verdict final : le nom de la personne si score >= seuil,
                 sinon "Unknown"
    """
    classement = classer_personnes(image, donnees_detection, database)
    if len(classement) == 0:
        return INCONNU, -1.0, INCONNU
    
    meilleur_nom, meilleur_score = classement[0]
    second_score = classement[1][1] if len(classement) > 1 else -1.0
    
    accepte = meilleur_score >= seuil and (meilleur_score - second_score) >= marge
    statut = meilleur_nom if accepte else INCONNU
    return meilleur_nom, meilleur_score, statut


def verifier_visage(image, donnees_detection, empreinte_reference, seuil=SEUIL_SIMILARITE):
    """
    Verification 1:1 : ce visage est-il celui de CETTE personne ?

    empreinte_reference : une empreinte (128,) ou une liste d'empreintes
                          (par exemple base["alice"]).

    Retourne (score, est_la_meme_personne) : un decimal et un booleen.
    """
    empreinte = extraire_embedding(image, _ligne_yunet(donnees_detection))
    references = np.atleast_2d(np.asarray(empreinte_reference, dtype=np.float32))
    score = max(_similarite(empreinte, ref / np.linalg.norm(ref)) for ref in references)
    return score, score >= seuil


# ======================================================================
# 4. PETIT TEST (sans YuNet, juste pour verifier que le module se charge)
# ======================================================================

if __name__ == "__main__":
    base = charger_base()
    print(f"Base actuelle : {list(base.keys()) or 'vide'}")
    print(f"Seuil : {SEUIL_SIMILARITE} | marge : {MARGE_MIN}")
    print("Ce module ne détecte pas les visages : lancez main.py pour l'utiliser.")
