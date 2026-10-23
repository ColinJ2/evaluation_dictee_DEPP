---
name: launch
description: Lance, surveille ou arrête une évaluation COMPLÈTE de la dictée (2015 : ~3 470 copies, 2021 : 6 793 copies ; des heures) sur le SSP Cloud. À utiliser dès que l'utilisateur demande de lancer un run, un benchmark, une évaluation sur tout l'échantillon, ou de savoir où en est un run en cours. Le run survit à la fermeture de la session.
---

# Lancer une évaluation complète de la dictée

Toute la logique est dans **`launchers/launch_eval.sh`** — un script bash autonome, sans
dépendance à un assistant. Ton rôle ici est de l'appeler avec les bons arguments et
d'expliquer ce qu'il répond. **N'improvise pas de `nohup` à la main** : le script
règle des pièges que la ligne de commande manuelle oublie systématiquement (verrou,
orphelins, log écrasé, export sous un mauvais nom).

## Les trois commandes

```bash
C=configs/scoring/dictee2021_end2end.yaml
launchers/launch_eval.sh --config $C --prompt base   # lancer
launchers/launch_eval.sh --config $C --status        # où en est le run : copies, débit, ETA
launchers/launch_eval.sh --config $C --stop          # arrêter proprement
```

`--config` (et `--model-name` si le modèle est surchargé) doit être **répété sur les
trois commandes**, sinon `--status`/`--stop` viseraient un autre run (sans `--config`,
c'est `dictee_end2end.yaml`, la dictée 2015). `--prompt` ne sert qu'au lancement.

## Nommage obligatoire d'un run publié (décision D12)

Un run complet est exporté pour le site, qui lit **année, approche et prompt dans le
nom du run**. Le champ `name` du YAML doit donc suivre
`dictee<AAAA>_<end2end|two_stage>_<prompt>` (prompt sans tiret bas, `base` = prompt
par défaut), et le lanceur reçoit `--prompt <prompt>` : il **refuse de partir** si
`--prompt` manque ou ne figure pas dans le nom. L'export va par défaut dans
`s3://projet-production-ecrits-depp/prediction_depp` (dossier lu par le site) ; ne
passer `--dest-prefix` que pour viser ailleurs.

| Config                                          | `name`                          | `--prompt` | Copies |
|-------------------------------------------------|---------------------------------|------------|--------|
| `configs/scoring/dictee_end2end.yaml`           | `dictee2015_end2end_base`       | `base`     | ~3 470 |
| `configs/scoring/dictee_two_stage.yaml`         | `dictee2015_two_stage_base`     | `base`     | ~3 470 |
| `configs/scoring/dictee2021_end2end.yaml`       | `dictee2021_end2end_base`       | `base`     | 6 793  |
| `configs/scoring/dictee2021_two_stage.yaml`     | `dictee2021_two_stage_base`     | `base`     | 6 793  |
| `configs/scoring/dictee2021_end2end_majponct.yaml`   | `dictee2021_end2end_majponct`   | `majponct` | 6 793  |
| `configs/scoring/dictee2021_two_stage_majponct.yaml` | `dictee2021_two_stage_majponct` | `majponct` | 6 793  |

**Nouveau prompt** : copier la config de base de l'année et de l'approche, changer
`name` (ex. `dictee2021_end2end_monprompt`) et les options du bloc `prompt:`, puis
lancer avec `--prompt monprompt`. Rien à coder côté site : le prompt apparaît dans
les menus au prochain rendu. Les configs `dictee_end2end_<bras>.yaml` (cot,
comptage…) sont des bras sur échantillon, hors de cette convention : `--no-export`
ou export vers `predictions/experimentations/` (D9).

`launchers/launch_eval.sh --help` liste tout le reste (`--limit`, `--passes`,
`--no-export`, `--dest-prefix`, `--skip-checks`, `--foreground`).

## Ce que fait le script, pour pouvoir l'expliquer

1. **Vérifications avant 30 h de calcul** : `uv` présent, config valide, `data.limit`
   bien à `null` (sinon il refuse : un run partiel se ferait passer pour complet),
   aucun run concurrent sur le même fichier de sortie (process + verrou `flock`),
   secrets lisibles, modèle joignable, CSV des labels lisible sur S3.
2. **Lancement détaché** (`setsid` + `nohup`) : le run survit à la fermeture de
   l'onglet, à une déconnexion réseau, à une mise en veille.
3. **Vérification du démarrage** : il attend la ligne `N copies au total, M à traiter
   (K déjà faites)` et l'affiche. C'est le contrôle qui compte — `0 déjà faites` alors
   qu'un checkpoint existe signalerait que le run vise le mauvais fichier.
4. **Relance des échecs** : jusqu'à `--passes` passages (3 par défaut). Le pipeline
   reprend seul les copies listées dans `failed_copies.txt`. La chaîne s'arrête dès
   qu'il n'y a plus d'échec, ou dès que le reliquat cesse de diminuer.
5. **Export S3 final** vers `prediction_depp/` sous le nom exact du run, avec un garde-fou : il refuse
   d'écraser un fichier S3 **plus gros** que le fichier local (protection contre
   l'écrasement d'un run complet par un run partiel).

## À savoir pour répondre à l'utilisateur

- **Durée** : ~3 470 copies (2015) ou 6 793 (2021), ~600-800 copies/h à concurrence 12. Compter des heures ; `--status` donne un débit mesuré et
  une heure de fin estimée, bien plus fiable qu'une estimation a priori.
- **Reprise** : après un crash ou un `--stop`, relancer **exactement la même commande**.
  Les copies déjà écrites sont sautées, rien n'est perdu sauf la copie en cours.
- **Fichiers** : tout est suffixé par le(s) modèle(s) — `<name>_<modèle>_predictions.jsonl`.
  Deux modèles n'écrasent jamais le même fichier, et peuvent tourner en parallèle.
- **Logs** : `logs/<run>_<horodatage>.log`, un par lancement, jamais écrasé ;
  `logs/<run>.latest.log` pointe sur le dernier.
- **Jamais** `kill` sur un PID : `uv run …` crée un wrapper **et** un `python3`, et
  tuer le seul wrapper laisse l'enfant écrire en orphelin. `--stop` s'en charge
  correctement (chaîne d'abord, puis `pkill -f` sur le motif exact du run).
- **Données sensibles** : ne jamais afficher ni copier d'images ou de transcriptions
  d'élèves hors du SSP Cloud, et rien de tout cela dans Git.

## Après le run

Les prédictions sont sur S3 (`prediction_depp/`, export automatique) et en local
dans `data/processed/`. Pour les voir sur le site : `uv run quarto render
website/resultats.qmd` (nommer la page : avec `freeze: auto`, un rendu global ne
recalcule pas une page dont seules les données ont changé). L'analyse se fait dans `notebooks/03_analyse_resultats.ipynb`
(métriques) et `notebooks/04_diagnostic.ipynb` (copies les plus en désaccord).
