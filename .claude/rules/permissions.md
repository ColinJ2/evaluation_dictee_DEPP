# Permissions et actions sensibles

Garde-fous techniques dans `.claude/settings.json` (partagé). Préférences personnelles :
`.claude/settings.local.json` (non commité). Vérifier l'état avec `/permissions`.

## Règles de conduite

- **Jamais de `git push` sans demande explicite**, et jamais de push forcé.
- **Aucune écriture S3 sans demande explicite** : exports de prédictions, upload,
  suppression, sync (CLI `aws`/`mc`, `scripts/export_predictions.py`, export du lanceur).
- **Ne jamais lire `.env`** ni le contenu de `data/` (données d'élèves mineurs).
- Pas d'actions destructrices (`rm -rf`, `git reset --hard`, `git clean`) sans accord.
- Ne pas arrêter ni relancer un run long sans demande (voir `runs-longs.md`).

## Ce que fait `settings.json`

| Niveau  | Cibles                                                                     |
|---------|----------------------------------------------------------------------------|
| `ask`   | `git push`, `git reset --hard`, `git clean`, `git branch -D`, `aws s3 cp/mv/rm/sync`, `mc cp/mv/rm/mirror`, `export_predictions.py`, `launch_eval.sh`, `pkill` |
| `deny`  | lecture de `.env`, lecture/écriture de `data/`, `rm -rf`, `git push --force/-f` |

`deny` l'emporte sur `ask`, qui l'emporte sur `allow`.

## Limites

Les règles reposent sur des motifs de commande : un accès S3 depuis un script Python
non listé n'est pas bloqué. La vraie protection est côté infrastructure : identifiants S3
en lecture seule pour l'assistant.
