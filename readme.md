## Dataset et préparation des données

### Contenu du dataset

Le dataset contient des photos des cinq membres du groupe (aristide, aroon, bias, elisabeth, noumssi), prises avec des téléphones personnels, plus un dossier `inconnu` de 11 photos de personnes extérieures au groupe. Toutes les personnes ont donné leur accord pour un usage strictement pédagogique.

Les photos varient par l'orientation du visage (face, tourné à gauche ou à droite), l'expression (neutre, sourire, sérieux), la distance à l'appareil et l'éclairage (intérieur, fenêtre, extérieur).

| Personne  | Photos valides | Entraînement | Test |
|-----------|----------------|--------------|------|
| aristide  | 22             | 17           | 5    |
| aroon     | 21             | 16           | 5    |
| bias      | 21             | 16           | 5    |
| elisabeth | 20             | 16           | 4    |
| noumssi   | 26             | 20           | 6    |
| inconnu   | 11             | 0            | 11   |
| **Total** | **121**        | **85**       | **36** |

### Organisation des dossiers

```text
data/
├── raw/                  <- photos brutes, un dossier par personne
│   ├── aristide/
│   ├── aroon/
│   ├── bias/
│   ├── elisabeth/
│   ├── noumssi/
│   └── inconnu/          <- personnes absentes de la base (test uniquement)
└── split/                <- généré automatiquement par le script
    ├── train/            <- 5 personnes, jamais "inconnu"
    └── test/             <- 5 personnes + "inconnu"
```

Le nom de chaque dossier sert d'étiquette de la personne dans le modèle et dans l'affichage final : ne pas le renommer.

### Préparation automatique (`src/prepare_dataset.py`)

Le script traite chaque photo de `data/raw/` :

1. lecture et réduction à 800 pixels maximum sur le plus grand côté ;
2. détection des visages avec YuNet (seuil de confiance 0,8) ;
3. rejet de toute photo qui ne contient pas exactement un visage (aucun visage, visage trop petit ou flou, plusieurs visages) ;
4. renommage normalisé (`prenom_001.jpg`, `prenom_002.jpg`, ...) ;
5. séparation aléatoire 80 % entraînement / 20 % test, avec une graine fixe (42) pour que le résultat soit reproductible.

Les photos du dossier `inconnu` vont uniquement dans `data/split/test/inconnu` et ne servent jamais à l'entraînement. Le script affiche pour chaque personne le nombre de photos gardées et rejetées, avec la raison de chaque rejet.

### Régénérer le dataset

Prérequis : environnement virtuel activé, dépendances installées (`pip install -r requirements.txt`) et modèle YuNet présent dans `models/face_detection_yunet_2023mar.onnx`.

Supprimer l'ancien résultat pour ne pas mélanger d'anciennes photos :

```powershell
Remove-Item -Recurse -Force data\split
```

Lancer la préparation :

```powershell
python src\prepare_dataset.py
```

### Règles à respecter

- Une seule personne par photo, sans filtre.
- Ne jamais utiliser le dossier `inconnu` pour l'entraînement.
- Les photos sont des données personnelles : ne pas les diffuser en dehors du projet.

### Limites

Le dataset est petit (20 à 26 images par personne) et provient de peu de sessions de prise de vue. Les résultats peuvent donc être sensibles à des conditions d'éclairage ou à des angles absents de la base. Les photos d'une même personne sont parfois proches les unes des autres, ce qui peut rendre le test plus facile qu'en conditions réelles.