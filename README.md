# Systèmes Intelligents de Détection et Classification de Panneaux Routiers

Pipeline de vision par ordinateur pour la détection et la classification automatique des panneaux routiers, utilisant et comparant des modèles de pointe comme YOLO, Faster R-CNN et DETR.

---

## Features

- Détection et classification de panneaux en temps réel.
- Comparaison des modèles selon précision, rapidité d’inférence et robustesse.
- Prétraitement et augmentation d’images pour améliorer la performance.
- Visualisation des résultats directement sur les images.
- Compatible avec les notebooks pour expérimentation et visualisation.

---

## Installation

Installez les packages nécessaires avec pip :

```bash
pip install torch torchvision opencv-python Pillow numpy matplotlib
```
#Dataset

Le dataset n’est pas inclus pour des raisons de taille.
Nous utilisons le même dataset pour la classification et la détection des panneaux routiers.

### Modifications apportées pour detection 

- Création de fichiers `labels/` correspondant à chaque image pour YOLO.
- Réorganisation des images en dossiers `train/`, `val/` et `test/`.
- Création du fichier `data.yaml` pour définir les chemins et les classes.
- Les fichiers `.cache` sont générés automatiquement et **ne sont pas inclus**.

### Organisation finale attendue

traffic_sign_detection/
└─ data/
├─ images/
│ ├─ train/
│ ├─ val/
│ └─ test/
├─ labels/
│ ├─ train/
│ ├─ val/
│ └─ test/
└─ data.yaml # fichier YAML pour YOLO

