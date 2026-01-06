# Reconnaissance d'émotions en temps réel

Détection d'émotions par analyse des landmarks faciaux (dlib) et classification MLP.

## Installation

```bash
pip install opencv-python dlib numpy scikit-learn pandas joblib
```

**Télécharger le modèle dlib:**

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Utilisation

**Entraînement:**

```bash
jupyter notebook emotion_trainer.ipynb
```

**Détection temps réel:**

```bash
python face_landmark_detection.py
```

Quitter avec `q`.

## Fichiers

```
├── emotion_trainer.ipynb               # Entraînement du modèle
├── face_landmark_detection.py          # Détection webcam
├── shape_predictor_68_face_landmarks.dat
└── emotion_mlp_model.joblib
```

## Référence

Basé sur: https://dlib.net/face_landmark_detection.py.html
