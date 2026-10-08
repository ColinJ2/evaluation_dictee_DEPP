# Règles sur les données

- **Données sensibles** : écritures d'élèves mineurs. Les microdonnées **ne quittent
  JAMAIS le SSP Cloud**. Aucune image ni donnée élève dans le dépôt Git (voir
  `.gitignore`) ; le dossier `data/` est ignoré.
- Les « .png » fournis sont en réalité des **TIFF bi-level 1 bit** (G4, 1594×2044) :
  toujours normaliser le format en entrée (`data/loaders.py`).
- La binarisation 1 bit détruit les nuances de gris (accents difficiles à lire) : demander
  à la DEPP si des scans en niveaux de gris existent en amont.
- Les annotations actuelles sont des CODES, pas des transcriptions mot à mot : fine-tuner
  un HTR suppose de produire un ground truth de transcription séparé.
