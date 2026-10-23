---
description: Lance, surveille ou arrête une évaluation complète de la dictée (2015 ou 2021)
agent: build
---

# Évaluation complète de la dictée

Arguments reçus : `$ARGUMENTS`

Toute la logique est dans `launchers/launch_eval.sh`, un script bash autonome à la
racine du dépôt. Ne réécris pas de `nohup` à la main : le script règle des pièges que
la ligne de commande manuelle oublie (verrou sur le fichier de sortie, process
orphelins, log écrasé, export S3 sous un mauvais nom).

## Ce que tu dois faire

1. Lis `$ARGUMENTS` pour déterminer l'intention :
   - vide, « lance », « démarre », « run complet » → **lancer** ;
   - « où en est », « avancement », « status », « statut » → **surveiller** ;
   - « arrête », « stop », « tue » → **arrêter**.
2. Exécute la commande correspondante depuis la racine du dépôt :

```bash
C=configs/scoring/dictee2021_end2end.yaml
launchers/launch_eval.sh --config $C --prompt base   # lancer
launchers/launch_eval.sh --config $C --status        # avancement : copies, débit, heure de fin
launchers/launch_eval.sh --config $C --stop          # arrêt propre (chaîne + benchmark)
```

3. Choisis la config selon l'année, l'approche et le prompt demandés (tableau
   ci-dessous), et **répète `--config` (et `--model-name` s'il est surchargé) sur
   `--status` et `--stop`**, sinon ces commandes viseraient un autre run.

### Nommage obligatoire d'un run publié (décision D12)

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

`launchers/launch_eval.sh --help` liste les autres options (`--limit`, `--passes`,
`--no-export`, `--dest-prefix`, `--skip-checks`, `--foreground`).

4. Restitue la sortie du script à l'utilisateur : elle contient déjà le résultat des
   vérifications, le chemin du log et les commandes de suivi. N'invente rien.

## Points à connaître pour répondre

- Le run est détaché (`setsid` + `nohup`) : il **survit** à la fermeture de l'onglet,
  à une déconnexion ou à une mise en veille. ~3 470 copies (2015) ou 6 793 (2021),
  plusieurs heures.
- Le script refuse de partir si un run tourne déjà sur le même fichier de sortie, ou
  si la config contient un `data.limit` (un run partiel se ferait passer pour complet).
- Après une interruption, relancer **exactement la même commande** : les copies déjà
  écrites sont sautées (checkpoint dans `data/processed/<run>_predictions.jsonl`).
- Les copies en échec sont relancées automatiquement (3 passes par défaut), puis le
  fichier est exporté vers S3 (`prediction_depp/`) sous le nom exact du run.
- N'utilise **jamais** `kill` sur un PID : `uv run` crée un wrapper *et* un `python3` ;
  tuer le wrapper laisse l'enfant écrire en orphelin. `--stop` fait ça correctement.
- Données d'élèves mineurs : aucune image ni transcription ne sort du SSP Cloud, et
  rien de tout cela ne va dans Git.
