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

import os
import pickle

import cv2
import numpy as np

# Dossier racine du projet = le dossier au-dessus de "src/".
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHEMIN_SFACE = os.path.join(RACINE, "models", "face_recognition_sface_2021dec.onnx")
CHEMIN_BASE = os.path.join(RACINE, "data", "embeddings.pkl")

# ======================================================================
# 1. OUTILS DE BASE : chargement, empreinte, similarite
# ======================================================================

# seuil de similarite demande par OpenCV
SEUIL_SIMILARITE = 0.365

#Message affiche lorsqu'une personne n'est pas reconnu
INCONNU = "UNKNOWN"

""" le chargement du model prend considerablement de temps, et plus lorsque l'operation est repetee
le modele sera charge par la fonction ci dessous une seule fois grace a "global", qui pemettra l'acces aux varible globale
ce qui evitera la suppression de la variable a la fin de l'execuction de cette fonction
"""
_sface = None

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
# 2. ENROLEMENT : construire et sauvegarder la base d'empreintes
# ======================================================================