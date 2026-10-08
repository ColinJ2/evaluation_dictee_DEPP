# evaluation_dictee

Évaluer automatiquement la production écrite manuscrite d'élèves (dictée et rédaction,
CE1 → 6e) avec des modèles d'IA open weight, pour assister la correction experte de la
DEPP, en mesurant rigoureusement la fiabilité avant tout usage.

Les règles détaillées sont dans `.claude/rules/` (chargées automatiquement) :

| Fichier                          | Contenu                                                        |
|----------------------------------|----------------------------------------------------------------|
| `rules/methodologie.md`          | grille de codage, ratures, méthodes A/B/C/D, confiance, métriques, copies écartées |
| `rules/donnees.md`               | données sensibles, format des scans, annotations               |
| `rules/code-conventions.md`      | Python/uv/ruff, typage, docstrings, configs, Langfuse, commits |
| `rules/runs-longs.md`            | lanceur, checkpointing, reprise, bras d'expérience, export     |
| `rules/permissions.md`          | garde-fous : push, écritures S3, fichiers interdits, lien avec `settings.json` |

Décisions historisées : `docs/decisions.md`. Démarrage rapide : `README.md`.

## Contexte

- **Commanditaire** : DEPP (Ministère de l'Éducation nationale), bureau B2.
- **Maîtrise d'œuvre** : SSP Lab (INSEE). **Partenaire** : TEKLIA (HTR, Arkindex).
- **Infrastructure** : SSP Cloud (Onyxia, GPU H100, vLLM via `llm.lab`, S3 MinIO, MLflow,
  Langfuse, Argo Workflows).
- La DEPP corrige depuis 2015 la production d'écrit sur échantillon, 100 % manuellement
  (application AGATE, 5 à 10 min/copie). On teste si l'IA peut fiabiliser et accélérer.

### Deux corpus

1. **Dictée CM2** (1987→2026) : texte de référence CONNU, ~10 000 imagettes (2015 + 2021),
   83 items par copie (mots + ponctuation).
2. **Production écrite libre** (nov. 2024) : PAS de référence, 15 027 élèves, 5 à 13
   critères selon le niveau.

**On commence par la dictée** : la référence connue permet une évaluation exacte
(CER/WER, comparaison item par item).

## Tâche et approche (résumé)

Chaque mot/ponctuation reçoit un code ; la cible principale est la grille simplifiée
**1 / erreur / 0** (ratures : état final). Quatre méthodes sont comparées (A HTR+LLM,
B VLM+LLM, **C end-to-end privilégiée**, D fine-tuning) avec un score de confiance par
item. Détails et justifications : `rules/methodologie.md`.

## Structure du dépôt

```
evaluation_dictee/
├── .claude/
│   ├── CLAUDE.md             ← ce fichier (≤ 200 lignes)
│   ├── rules/                ← règles du projet, un thème par fichier
│   ├── settings.json         ← permissions partagées (ask/deny)
│   └── skills/launch/        ← skill /launch (lancer/suivre/arrêter un run complet)
├── .opencode/command/        ← équivalent OpenCode du skill launch
├── .github/workflows/        ← CI (ci.yml) et site (site.yml)
├── README.md                 ← démarrage rapide
├── pyproject.toml, uv.lock   ← dépendances + config outils
├── .env.example              ← variables d'environnement attendues
├── configs/                  ← YAML d'expériences (un par run)
├── src/evaluation_dictee/
│   ├── config.py             ← chargement/validation des configs (Pydantic)
│   ├── data/                 ← chargement images, parsing grille, I/O S3
│   ├── models/               ← interfaces modèles (VLM via vLLM, HTR, base)
│   ├── pipeline/             ← orchestration des méthodes A/B/C/D
│   ├── evaluation/           ← métriques, calibration, comparaison annotateurs
│   └── utils/                ← logging, Langfuse, helpers
├── scripts/                  ← points d'entrée CLI (run_benchmark, export_predictions…)
├── launchers/                ← lanceurs shell des runs longs (launch_eval.sh)
├── tests/                    ← tests pytest
├── notebooks/                ← exploration (pas de logique réutilisable)
├── website/                  ← site Quarto (résultats)
├── docs/                     ← note de cadrage, grille de codage, décisions
└── logs/                     ← logs des runs
```

## Commandes

```bash
uv sync                                   # environnement (groupe dev inclus)
uv run ruff check src tests               # lint
uv run ruff format src tests              # formatage
uv run pytest                             # tests
uv run mypy src                           # typage

# Un passage dans le terminal (échantillon réduit)
uv run scripts/run_benchmark.py --config configs/scoring/dictee_REFERENCE.yaml

# Échantillon complet (~30 h) : TOUJOURS via le lanceur (voir rules/runs-longs.md)
launchers/launch_eval.sh [--status | --stop]

# Site Quarto (les pages recalculent leurs figures au rendu)
uv sync --extra website && uv run quarto render website
```

## Pour un⋅e débutant⋅e

- Lire `README.md`, puis `docs/note_de_cadrage.docx`.
- Point d'entrée le plus simple : `scripts/run_benchmark.py` (config → données → modèle →
  métriques). Suivre le fil.
- En cas de doute méthodologique, la réponse est dans `.claude/rules/` ou
  `docs/decisions.md`.

## État d'avancement

- [x] Note de cadrage (docs/)
- [x] Test pilote manuel sur 3 copies (kappa 0,785 → 0,879 grille simplifiée)
- [x] Phase 0 : accès S3, dataset d'évaluation (200-300 copies gold), MLflow/Langfuse
- [ ] Phase 1 : benchmark HTR sur la dictée
- [x] Phase 2 : benchmark scoring (méthodes A/B/C)
- [ ] Phase 3 : passage à l'échelle ou fine-tuning
