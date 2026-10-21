# Règles pour les runs longs et les exports

## Lancement

Un benchmark complet (3469 copies × ~30 s) dure ~30 h. **Jamais directement dans le
terminal du navigateur** (plantage réseau, veille ou fermeture d'onglet = process tué).
Point d'entrée unique : `launchers/launch_eval.sh` (ou le skill `/launch`).

```bash
launchers/launch_eval.sh            # lancer l'échantillon complet
launchers/launch_eval.sh --status   # copies, débit mesuré, heure de fin
launchers/launch_eval.sh --stop     # arrêt propre, sans process orphelin
```

Il vérifie l'environnement, détache le run (`setsid` + `nohup`), écrit un log horodaté par
lancement, relance les copies en échec et exporte vers S3. Détails : `--help`,
`launchers/README.md`.

- Les options qui identifient le run (`--config`, `--model-name`) doivent être **répétées
  sur les trois commandes** : le nom d'un run est `<name>_<modèle(s)>`, sans elles
  `--status` interrogerait un autre run.
- **Un seul run par fichier de sortie** : deux runs qui appendent le même JSONL dupliquent
  les copies et faussent les métriques (verrou `flock` sur `<sortie>.lock`).
- **Ne jamais arrêter un run par son PID** : `uv run` crée un wrapper ET un
  `python3 scripts/run_benchmark.py` ; tuer le wrapper laisse l'enfant orphelin écrire.
  Utiliser `--stop`, ou à la main `pkill -f "run_benchmark.py --config <la config>"`.
- Onyxia : `tmux` indisponible (vscode-python) → `screen` pour un run *interactif*, le
  lanceur pour tout le reste.

## Checkpointing et reprise

- Chaque copie est écrite immédiatement (`flush + fsync`) : un crash ne perd que la copie
  en cours ; relancer la même commande **reprend** (copies déjà dans
  `<run>_<modele>_predictions.jsonl` sautées).
- Les copies en exception API sont loggées dans `<run>_<modele>_failed_copies.txt` et
  retentées au lancement suivant.
- Une copie dont **aucun** item n'a pu être parsé (code `?` partout) n'est PAS considérée
  comme faite : `pipeline/purge.preparer_reprise` la retire du JSONL (sauvegarde `.bak`)
  et le run la refait. Sans ce garde-fou, la reprise fige les échecs (84 copies tout en
  `?` dans le run two_stage du 5 août 2026).
- Repartir de zéro : supprimer `<run>_<modele>_predictions.jsonl` ou changer `config.name`.

## Bras d'expérience et publication sur le site

- Un bras testé sur échantillon (chain-of-thought, comptage, exemples de fautes…) s'exporte
  dans `predictions/experimentations/`, **jamais** dans `predictions/` (sinon faux modèle
  sur le site ; décision D9, `scripts/export_predictions.py` route selon l'effectif réel).
- Publier un prompt sur le site : run sur le corpus complet, `name` de config au
  format `dictee<AAAA>_<end2end|two_stage>_<prompt>` (prompt sans tiret bas, `base` =
  prompt par défaut), export avec `--prompt <prompt>` vers `prediction_depp/` (dossier
  lu par le site). Le site lit tout dans le nom ; un nom non conforme est ignoré
  (décision D12, `src/evaluation_dictee/run_names.py`).
