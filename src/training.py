# coding: utf-8 

"""
Module d'entraînement et d'évaluation de la reconnaissance faciale.

Ce module exploite les embeddings faciaux produits par SFace.
Il permet de construire une base de référence, de prédire une identité
et de rejeter les visages qui ne correspondent pas suffisamment
aux identités connues.
"""
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class FaceClassifier:
    """Classificateur de visages basé sur la similarité cosinus."""

    def __init__(self, seuil=0.365):
        # Seuil initial à ajuster avec les données de validation.
        self.seuil = seuil

        # Embeddings de référence regroupés par identité.
        self.embeddings_par_personne = {}

    def entrainer(self, embeddings, etiquettes):
        """
        Construit la base de référence à partir des embeddings.

        Paramètres :
            embeddings : tableau contenant les vecteurs faciaux.
            etiquettes : identité correspondant à chaque vecteur.
        """

        # Convertir les données en tableaux NumPy.
        embeddings = np.asarray(embeddings, dtype=np.float32)
        etiquettes = np.asarray(etiquettes)

        # Vérifier que les données sont compatibles.
        if embeddings.ndim != 2:
            raise ValueError(
                "Les embeddings doivent former un tableau à deux dimensions."
            )

        if len(embeddings) != len(etiquettes):
            raise ValueError(
                "Le nombre d'embeddings doit correspondre au nombre d'étiquettes."
            )

        if len(embeddings) == 0:
            raise ValueError(
                "Impossible de construire une référence sans données."
            )

        # Vérifier que les valeurs sont valides.
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Les embeddings contiennent des valeurs invalides.")

        # Regrouper les embeddings par identité.
        self.embeddings_par_personne = {}

        for identite in np.unique(etiquettes):

            # Récupérer les embeddings de cette personne.
            vecteurs = embeddings[etiquettes == identite]

            # Calculer le vecteur moyen de référence.
            reference = np.mean(vecteurs, axis=0)

            # Normaliser le vecteur pour faciliter la comparaison.
            norme = np.linalg.norm(reference)

            if norme == 0:
                raise ValueError(
                    f"Le vecteur de référence de {identite} est nul."
                )

            self.embeddings_par_personne[str(identite)] = reference / norme

    def predire(self, embedding):
        
        """
        Prédit l'identité d'un visage.

        Retourne :
            (identité, similarité)

        Retourne ("Unknown", similarité) si le seuil n'est pas atteint.
        """

        # Vérifier que la base de référence a été construite.
        if not self.embeddings_par_personne:
            raise RuntimeError(
                "Le classificateur doit être entraîné avant toute prédiction."
            )

        # Convertir et normaliser le nouvel embedding.
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if not np.all(np.isfinite(embedding)):
            raise ValueError("L'embedding fourni contient des valeurs invalides.")

        norme = np.linalg.norm(embedding)

        if norme == 0:
            raise ValueError("Impossible de comparer un embedding nul.")

        embedding = embedding / norme

        meilleure_identite = "UNKNOWN"
        meilleure_similarite = -1.0

        # Comparer le visage aux références de toutes les personnes.
        for identite, reference in self.embeddings_par_personne.items():

            # Vérifier que les vecteurs ont la même dimension.
            if embedding.shape != reference.shape:
                raise ValueError(
                    "La dimension de l'embedding ne correspond pas "
                    "à celle des références."
                )

            # Calculer la similarité cosinus entre les deux vecteurs.
            similarite = float(np.dot(embedding, reference))

            # Conserver la meilleure correspondance.
            if similarite > meilleure_similarite:
                meilleure_similarite = similarite
                meilleure_identite = identite

        # Rejeter le visage si la similarité est insuffisante.
        if meilleure_similarite < self.seuil:
            return "UNKNOWN", meilleure_similarite

        return meilleure_identite, meilleure_similarite


def evaluer_classificateur(classificateur, embeddings_test, etiquettes_reelles):
    """
    Évalue le classificateur sur des embeddings de test.

    Les étiquettes réelles peuvent inclure "inconnu" pour les personnes
    absentes de la base de référence.
    """

    # Préparer les données de test.
    embeddings_test = np.asarray(embeddings_test, dtype=np.float32)
    etiquettes_reelles = np.asarray(etiquettes_reelles, dtype=str)

    # Vérifier que les données sont cohérentes.
    if embeddings_test.ndim != 2:
        raise ValueError(
            "Les embeddings de test doivent former un tableau à deux dimensions."
        )

    if len(embeddings_test) != len(etiquettes_reelles):
        raise ValueError(
            "Le nombre d'embeddings de test doit correspondre "
            "au nombre d'étiquettes réelles."
        )

    if len(embeddings_test) == 0:
        raise ValueError("Aucune donnée de test n'est disponible.")

    # Effectuer une prédiction pour chaque visage.
    etiquettes_predites = []

    for embedding in embeddings_test:
        identite, _ = classificateur.predire(embedding)
        etiquettes_predites.append(identite)

    # Calculer les métriques de classification.
    accuracy = accuracy_score(
        etiquettes_reelles,
        etiquettes_predites
    )

    # Inclure les identités réelles et prédites dans la matrice.
    etiquettes = sorted(
        set(etiquettes_reelles.tolist())
        | set(etiquettes_predites)
    )

    matrice = confusion_matrix(
        etiquettes_reelles,
        etiquettes_predites,
        labels=etiquettes
    )

    rapport = classification_report(
        etiquettes_reelles,
        etiquettes_predites,
        labels=etiquettes,
        zero_division=0
    )

    # Retourner les résultats pour le notebook.
    return {
        "accuracy": accuracy,
        "matrice_confusion": matrice,
        "etiquettes": etiquettes,
        "rapport": rapport,
        "predictions": etiquettes_predites
    }
