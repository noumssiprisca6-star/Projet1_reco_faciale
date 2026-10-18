# coding: utf-8

"""
Module pipeline.py
------------------
Relie la détection, la reconnaissance et l'entraînement.

Responsabilités :
1. Parcourir les dossiers d'images.
2. Détecter les visages avec YuNet.
3. Extraire les empreintes avec SFace.
4. Préparer les données pour l'entraînement et l'évaluation.
"""

from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Importer les modules du projet.
from detection import FaceDetector
from recognition import extraire_embedding
# Importer le classificateur et la fonction d'évaluation.
from training import FaceClassifier, evaluer_classificateur


# ============================================================
# 1. CONFIGURATION DES DOSSIERS
# ============================================================

# Racine du projet : dossier parent de src/.
RACINE_PROJET = Path(__file__).resolve().parent.parent

# Dossiers contenant les images d'entraînement et de test.
DOSSIER_TRAIN = RACINE_PROJET / "data" / "split" / "train"
DOSSIER_TEST = RACINE_PROJET / "data" / "split" / "test"

# Extensions d'images autorisées.
EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# 2. EXTRACTION DES EMBEDDINGS
# ============================================================

def extraire_donnees(dossier, inclure_inconnu=False):
    """
    Parcourt les sous-dossiers d'un dossier d'images.

    Pour chaque image :
    - charge l'image ;
    - détecte les visages ;
    - extrait l'empreinte du visage ;
    - associe l'empreinte à l'identité du dossier.

    Paramètres :
        dossier : dossier contenant un sous-dossier par identité.
        inclure_inconnu : autorise le dossier 'inconnu' pour les tests.

    Retourne :
        embeddings : tableau NumPy de forme (nombre_visages, 128).
        etiquettes : tableau contenant les identités correspondantes.
    """

    dossier = Path(dossier)

    # Vérifier que le dossier existe.
    if not dossier.is_dir():
        raise FileNotFoundError(
            f"Le dossier de données est introuvable : {dossier}"
        )

    # Initialiser les résultats.
    embeddings = []
    etiquettes = []

    # Créer le détecteur une seule fois.
    detecteur = FaceDetector()

    # Parcourir les dossiers des identités.
    for dossier_personne in sorted(dossier.iterdir()):

        # Ignorer les fichiers et les éléments qui ne sont pas des dossiers.
        if not dossier_personne.is_dir():
            continue

        identite = dossier_personne.name

        # Ignorer les personnes inconnues pendant l'entraînement.
        if identite.lower() == "inconnu" and not inclure_inconnu:
            continue

        print(f"\nTraitement de l'identité : {identite}")

        # Parcourir les images de cette personne.
        images = sorted(
            fichier
            for fichier in dossier_personne.iterdir()
            if fichier.is_file()
            and fichier.suffix.lower() in EXTENSIONS_IMAGES
        )

        for chemin_image in images:

            # Charger l'image au format BGR attendu par OpenCV.
            image = cv2.imread(str(chemin_image))

            if image is None:
                print(f"  Image illisible : {chemin_image.name}")
                continue

            # Détecter les visages présents dans l'image.
            visages = detecteur.detect(image)

            if not visages:
                print(f"  Aucun visage détecté : {chemin_image.name}")
                continue

            # Une image d'identification doit contenir un seul visage.
            if len(visages) > 1:
                print(
                    f"  Plusieurs visages détectés : {chemin_image.name}"
                    " — image ignorée."
                )
                continue

            # Récupérer les données brutes produites par YuNet.
            ligne_yunet = visages[0]["raw"]

            # Extraire l'empreinte faciale avec SFace.
            embedding = extraire_embedding(image, ligne_yunet)

            # Enregistrer l'empreinte et son identité.
            embeddings.append(embedding)
            etiquettes.append(identite)

            print(f"  Empreinte extraite : {chemin_image.name}")

    # Vérifier qu'au moins une empreinte a été obtenue.
    if not embeddings:
        return np.empty((0, 128), dtype=np.float32), np.array([], dtype=str)

    # Convertir les listes en tableaux NumPy.
    return (
        np.asarray(embeddings, dtype=np.float32),
        np.asarray(etiquettes, dtype=str)
    )
    
# ============================================================
# 3. ENTRAÎNEMENT DU CLASSIFICATEUR
# ============================================================

def entrainer_modele(seuil=0.85):
    """
    Extrait les empreintes des images d'entraînement et entraîne
    le classificateur sur les identités connues.

    Paramètre :
        seuil : similarité minimale pour accepter une identité.

    Retourne :
        classificateur entraîné.
    """

    print("\n=== EXTRACTION DES DONNÉES D'ENTRAÎNEMENT ===")

    # Extraire les empreintes des personnes connues.
    embeddings_train, etiquettes_train = extraire_donnees(
        DOSSIER_TRAIN,
        inclure_inconnu=False
    )

    # Vérifier que des données ont été obtenues.
    if len(embeddings_train) == 0:
        raise ValueError(
            "Aucune empreinte d'entraînement disponible. "
            "Vérifiez que les images sont présentes et exploitables."
        )

    # Créer le classificateur avec le seuil choisi.
    classificateur = FaceClassifier(seuil=0.85)

    # Construire les références des différentes personnes.
    classificateur.entrainer(
        embeddings_train,
        etiquettes_train
    )

    print("\n=== ENTRAÎNEMENT TERMINÉ ===")
    print(f"Nombre d'empreintes : {len(embeddings_train)}")
    print(f"Nombre de personnes : {len(classificateur.embeddings_par_personne)}")
    print(f"Seuil initial : {seuil}")

    # Retourner le modèle entraîné pour les tests.
    return classificateur

# ============================================================
# 4. ÉVALUATION DU CLASSIFICATEUR
# ============================================================

def evaluer_modele(classificateur):
    """
    Évalue un classificateur entraîné sur les images de test.

    Les images du dossier 'inconnu' servent à tester le rejet
    des personnes qui ne figurent pas dans la base d'entraînement.

    Paramètre :
        classificateur : modèle déjà entraîné.

    Retourne :
        dictionnaire contenant les métriques d'évaluation.
    """

    print("\n=== EXTRACTION DES DONNÉES DE TEST ===")

    # Extraire les empreintes des personnes connues et inconnues.
    embeddings_test, etiquettes_test = extraire_donnees(
        DOSSIER_TEST,
        inclure_inconnu=True
    )

    # Vérifier qu'il existe des données de test.
    if len(embeddings_test) == 0:
        raise ValueError(
            "Aucune empreinte de test disponible. "
            "Vérifiez que les images de test sont présentes et exploitables."
        )

    # Le classificateur renvoie 'UNKNOWN' pour les visages rejetés.
    # Nous utilisons la même étiquette pour le dossier des inconnus.
    etiquettes_test = np.asarray([
        "UNKNOWN" if etiquette.lower() == "inconnu" else etiquette
        for etiquette in etiquettes_test
    ])

    # Calculer la précision, la matrice de confusion et le rapport.
    resultats = evaluer_classificateur(
        classificateur,
        embeddings_test,
        etiquettes_test
    )

    # Afficher les résultats.
    print("\n=== RÉSULTATS DE L'ÉVALUATION ===")
    print(f"Précision globale : {resultats['accuracy']:.2%}")

    print("\nRapport de classification :")
    print(resultats["rapport"])

    print("\nÉtiquettes de la matrice :")
    print(resultats["etiquettes"])

    print("\nMatrice de confusion :")
    print(resultats["matrice_confusion"])

    return resultats

# ============================================================
# SAUVEGARDE DES RÉSULTATS D'ÉVALUATION
# ============================================================

def sauvegarder_resultats(resultats, chemin_fichier=None):
    """
    Enregistre les résultats d'évaluation au format JSON.

    Paramètres :
        resultats : dictionnaire renvoyé par evaluer_classificateur()
        chemin_fichier : chemin du fichier de sauvegarde (facultatif)

    Retour :
        Le chemin du fichier créé.
    """

    # Définir le chemin par défaut dans le dossier results
    if chemin_fichier is None:
        dossier_resultats = RACINE_PROJET / "results"

        # Créer le dossier s'il n'existe pas
        dossier_resultats.mkdir(parents=True, exist_ok=True)

        chemin_fichier = dossier_resultats / "evaluation.json"
    else:
        # Convertir le chemin fourni en objet Path
        chemin_fichier = Path(chemin_fichier)

        # Créer le dossier parent si nécessaire
        chemin_fichier.parent.mkdir(parents=True, exist_ok=True)

    # Préparer les données à enregistrer
    donnees = {
        "accuracy": float(resultats["accuracy"]),
        "etiquettes": list(resultats["etiquettes"]),
        "matrice_confusion": resultats["matrice_confusion"].tolist(),
        "rapport": resultats["rapport"],
        "predictions": list(resultats["predictions"])
    }

    # Écrire les résultats dans le fichier JSON
    with open(chemin_fichier, "w", encoding="utf-8") as fichier:
        json.dump(donnees, fichier, ensure_ascii=False, indent=4)

    print(f"Résultats sauvegardés dans : {chemin_fichier}")

    # Retourner le chemin du fichier créé
    return chemin_fichier

# ============================================================
# VISUALISATION DE LA MATRICE DE CONFUSION
# ============================================================

def afficher_matrice_confusion(resultats, sauvegarder=False):
    """
    Affiche la matrice de confusion sous forme de graphique.

    Paramètres :
        resultats : dictionnaire renvoyé par evaluer_classificateur()
        sauvegarder : si True, enregistre le graphique dans results/
    """

    # Récupérer la matrice et les noms des classes
    matrice = np.asarray(resultats["matrice_confusion"])
    etiquettes = resultats["etiquettes"]

    # Créer la figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Afficher la matrice sous forme d'image
    image = ax.imshow(matrice, interpolation="nearest")

    # Ajouter une barre indiquant les valeurs représentées
    fig.colorbar(image, ax=ax)

    # Définir les noms des classes sur les axes
    positions = np.arange(len(etiquettes))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(etiquettes, rotation=45, ha="right")
    ax.set_yticklabels(etiquettes)

    # Ajouter les titres et les légendes
    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Identités prédites")
    ax.set_ylabel("Identités réelles")

    # Inscrire les valeurs dans les cellules
    for i in range(matrice.shape[0]):
        for j in range(matrice.shape[1]):
            ax.text(j, i, str(matrice[i, j]),
                    ha="center", va="center")

    # Ajuster les marges pour rendre le graphique lisible
    fig.tight_layout()

    # Enregistrer le graphique uniquement si demandé
    if sauvegarder:
        dossier_resultats = RACINE_PROJET / "results"
        dossier_resultats.mkdir(parents=True, exist_ok=True)

        chemin_image = dossier_resultats / "matrice_confusion.png"
        fig.savefig(chemin_image, dpi=150, bbox_inches="tight")

        print(f"Matrice enregistrée dans : {chemin_image}")

    # Afficher le graphique
    plt.show()

