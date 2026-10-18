```markdown
# Projet 1 : Reconnaissance Faciale Intelligente des Étudiants 🎓

Ce projet s'inscrit dans le cadre du module **ANI-2083 - Modèle de la SAT et IA dans les Arts Technologiques** à l'École Nationale Supérieure Polytechnique de Yaoundé (ENSPY)[cite: 1].

---

## 📋 Table des matières
1. [Présentation & Objectifs](#-présentation--objectifs)
2. [Architecture du Projet](#-architecture-du-projet)
3. [Répartition des Rôles](#-répartition-des-rôles)
4. [Prérequis & Installation Détaillée](#-prérequis--installation-détaillée)
5. [Téléchargement des Modèles ONNX](#-téléchargement-des-modèles-onnx)
6. [Structure de la Base de Données (Dataset)](#-structure-de-la-base-de-données-dataset)
7. [Lancement & Utilisation](#-lancement--utilisation)
8. [Résolution des Problèmes Courants](#-résolution-des-problèmes-courants)
9. [Livrables attendus](#-livrables-attendus)

---

## 🎯 Présentation & Objectifs
L'objectif est de développer une application de vision par ordinateur en Python permettant de :
* Détecter les visages en temps réel à partir du flux d'une webcam ou d'une image fixe à l'aide du modèle **YuNet**[cite: 1].
* Extraire les empreintes faciales (embeddings) et identifier l'étudiant correspondant à l'aide du modèle **SFace**[cite: 1].
* Afficher un rectangle de sélection (*bounding box*), le nom de la personne reconnue ainsi que le score de confiance[cite: 1].
* Renvoyer **`Inconnu` / `Unknown`** si la personne n'est pas enregistrée dans la base ou si le score de similarité est inférieur au seuil de confiance[cite: 1].

---

## 📂 Architecture du Projet
```text
projet1_reco_faciale/
│
├── dataset/                    # Base de données d'images des 5 étudiants[cite: 1]
│   ├── étudiant_1/
│   ├── étudiant_2/
│   └── ...
├── models/                     # Modèles pré-entraînés au format ONNX
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── notebooks/                  # Notebooks Jupyter pour l'entraînement et l'évaluation[cite: 1]
├── result/                     # Graphiques, matrice de confusion et captures[cite: 1]
├── src/                        # Code source principal
│   └── main.py                 # Script d'exécution et boucle temps réel
├── test/                       # Images de test pour personnes inconnues/absentes[cite: 1]
├── LICENSE                     # Licence du projet
├── requirements.txt            # Dépendances Python[cite: 1]
└── README.md                   # Documentation du projet[cite: 1]

```

---

## 👥 Répartition des Rôles

* **Bias** : Dataset et préparation des données (collecte de 20 à 40 photos par étudiant, gestion de la qualité et des conditions d'éclairage).


* **Elisabeth** : Détection des visages (modèle YuNet/OpenCV, extraction des coordonnées et points clés faciaux).


* **Aroon** : Reconnaissance faciale (modèle SFace, extraction des embeddings, comparaison cosinus).


* **Aristide** : Entraînement et évaluation (notebooks, matrice de confusion, métriques de performance, tests de personnes inconnues).


* **Noumssi** : Intégration globale, architecture système, coordination Git/GitHub, tests finaux et rédaction de la documentation.



---

## 🛠️ Prérequis & Installation Détaillée

### 1. Prérequis Système

* **Python** : Version 3.9 ou plus récente (vérifiez avec `python --version` ou `python3 --version`).
* **Pip** : Gestionnaire de paquets Python (`pip --version`).
* **Webcam** fonctionnelle et connectée à votre machine.

### 2. Cloner le Projet

Ouvrez un terminal et clonez le dépôt Git :

```bash
git clone <URL_DU_DEPOT_GITHUB>
cd projet1_reco_faciale

```

### 3. Création et Activation d'un Environnement Virtuel (Recommandé)

Pour éviter tout conflit de dépendances, créez un environnement virtuel :

* **Sous Windows (PowerShell / CMD) :**
```powershell
python -m venv venv
.\venv\Scripts\activate

```


* **Sous Linux / macOS :**
```bash
python3 -m venv venv
source venv/bin/activate

```



### 4. Installation des Dépendances

Installez l'ensemble des bibliothèques nécessaires avec le fichier `requirements.txt` :

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

Si le fichier `requirements.txt` n'est pas encore rempli, installez directement les paquets suivants :

```bash
pip install opencv-python numpy matplotlib jupyter

```

---

## 📥 Téléchargement des Modèles ONNX

L'application s'appuie sur deux modèles légers d'OpenCV Zoo. Vous devez placer les deux fichiers `.onnx` dans le dossier `models/` situé à la racine du projet :

1. **Détecteur de visages (YuNet)** :
* Fichier : `face_detection_yunet_2023mar.onnx`
* Téléchargement : Récupérer depuis le dépôt officiel [OpenCV Model Zoo (YuNet)](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).


2. **Reconnaisseur de visages (SFace)** :
* Fichier : `face_recognition_sface_2021dec.onnx`
* Téléchargement : Récupérer depuis le dépôt officiel [OpenCV Model Zoo (SFace)](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).



*(Note : Ces fichiers sont au format binaire et ne doivent pas être ouverts manuellement dans un éditeur de texte).*

---

## 📁 Structure de la Base de Données (Dataset)

Le dossier `dataset/` doit être structuré de la manière suivante pour permettre un chargement automatique par `main.py` :

```text
dataset/
├── Jean_Dupont/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── Marie_Claire/
│   ├── img1.jpg
│   └── ...

```

Chaque dossier correspond au nom complet de l'étudiant et contient 20 à 40 images variées (angles, expressions, éclairages).

---

## 🚀 Lancement & Utilisation

Une fois l'environnement activé, les modèles placés dans `models/` et les images dans `dataset/` :

1. Naviguez dans le dossier `src/` :
```bash
cd src

```


2. Exécutez le script d'intégration principal :
```bash
python main.py

```


3. **Interactions :**
* Une fenêtre intitulée `Projet 1 - Reconnaissance Faciale Intelligente (ENSPY)` s'ouvrira.
* L'application détecte en temps réel votre visage, affiche un cadre vert si vous êtes reconnu (avec votre nom et le score), ou un cadre rouge marqué `Inconnu`.
* Appuyez sur la touche **`q`** de votre clavier pour fermer l'application.



---

## ❓ Résolution des Problèmes Courants

### 1. `Erreur : Les modèles ONNX sont introuvables`

* **Cause :** Les fichiers `.onnx` ne sont pas placés dans le dossier `models/` ou le chemin relatif est incorrect.
* **Solution :** Vérifiez que le dossier `models/` se trouve à la racine du projet et contient exactement les fichiers `face_detection_yunet_2023mar.onnx` et `face_recognition_sface_2021dec.onnx`.

### 2. `cv2.error: OpenCV DNN GemmLayerImpl::getMemoryShapes Invalid dimension`

* **Cause :** Incompatibilité entre les dimensions de l'image capturée par la webcam et la taille configurée dans YuNet.
* **Solution :** Ce problème est résolu dans `main.py` grâce à l'appel `detector.setInputSize((w, h))` appliqué de manière dynamique à chaque itération de la boucle vidéo.

### 3. La webcam ne s'ouvre pas

* **Cause :** L'index de la caméra est incorrect ou une autre application utilise la webcam.
* **Solution :** Dans `main.py`, modifiez `cv2.VideoCapture(0)` en `cv2.VideoCapture(1)` ou vérifiez les autorisations de caméra de votre système d'exploitation.

---

## 📦 Livrables du Projet

Conformément aux exigences du projet :

* [x] Code source complet et modulaire (`src/main.py`).
* [x] Dataset structuré des étudiants (`dataset/`).


* [x] Notebook d'évaluation et matrice de confusion (`notebooks/`).


* [x] Fichier des dépendances (`requirements.txt`).


* [x] Ce guide complet d'installation et d'utilisation (`README.md`).


* [x] Rapport de projet (3 à 5 pages) et captures d'écran de démonstration.



```

```
