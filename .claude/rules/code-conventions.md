---
paths:
  - "src/**/*.py"
  - "scripts/**/*.py"
  - "tests/**/*.py"
  - "website/**/*.py"
---

# Conventions de code

- **Python ≥ 3.11**, dépendances gérées avec **uv** exclusivement.
- **ruff** (lint + format) : `uv run ruff check` et `uv run ruff format` avant chaque commit.
- **Typage** : annoter toutes les fonctions publiques. `mypy` en CI (non bloquant au début).
- **Docstrings en français**, style court (résumé d'une ligne + Args/Returns si utile).
- **Pas de chemin en dur** : tout passe par `configs/*.yaml` et `src/evaluation_dictee/config.py`.
- **Pas de secret dans le code** : identifiants S3, tokens → variables d'environnement
  (`.env`, jamais commité ; modèle : `.env.example`).
- **Reproductibilité** : une expérience = un fichier de config versionné + un run tracé
  dans Langfuse. Fixer les graines aléatoires.
  - Traces : **session** (un lancement ; `session_id` = `config.name` + horodatage) →
    **trace** (une copie, avec entrée/sortie et score d'accord) → **génération** (un appel
    LLM).
  - Paramètres du run propagés en tags/metadata (dont le tag `run:<nom>` qui regroupe les
    lancements d'une même config) ; métriques agrégées en Scores + metadata
    (voir `utils/tracking.py`).
  - Le fine-tuning HTR sur GPU reste suivi via MLflow (hors périmètre Langfuse).
- **Notebooks** : exploration uniquement ; toute logique réutilisable va dans `src/`.
- **Commits** : messages clairs en français, format `type: description`
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
