"""Reconstitue `<run>_transcriptions.json` pour un run two_stage déjà écrit.

`run_benchmark` produit ce fichier en fin de run. Un run antérieur à cet ajout, ou
interrompu avant la fin, ne l'a pas : ce script le tire du JSONL de prédictions, où la
transcription de l'étape 1 figure déjà (champ `raw_transcription`), sans rien
réévaluer auprès du modèle.

Usage :
    uv run scripts/backfill_transcriptions.py --run-name dictee_two_stage_qwen3-6-35b-moe
    uv run scripts/backfill_transcriptions.py --tous   # tous les runs de data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_dictee.evaluation.report import load_predictions
from evaluation_dictee.pipeline.benchmark import SUFFIXE_TRANSCRIPTIONS, ecrire_transcriptions
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Écrit le fichier des transcriptions d'un run, ou de tous les runs du dossier."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", help="Nom du run (préfixe du fichier de prédictions).")
    parser.add_argument(
        "--tous",
        action="store_true",
        help="Traite tous les runs de --dossier (les runs end_to_end sont sautés).",
    )
    parser.add_argument("--dossier", default="data/processed", help="Dossier des prédictions.")
    args = parser.parse_args()
    if not args.run_name and not args.tous:
        parser.error("préciser --run-name ou --tous")

    dossier = Path(args.dossier)
    if args.tous:
        runs = [
            p.name.removesuffix("_predictions.jsonl")
            for p in sorted(dossier.glob("*_predictions.jsonl"))
        ]
    else:
        runs = [args.run_name]

    for run in runs:
        chemin = dossier / f"{run}_predictions.jsonl"
        if not chemin.exists():
            logger.warning("Introuvable, sauté : %s", chemin)
            continue
        destination = dossier / f"{run}{SUFFIXE_TRANSCRIPTIONS}"
        n = ecrire_transcriptions(load_predictions(chemin).to_dict("records"), destination)
        if n:
            logger.info("%s : %d transcription(s) -> %s", run, n, destination)
        else:
            logger.info("%s : aucune transcription d'étape 1 (run end_to_end ?), rien écrit.", run)


if __name__ == "__main__":
    main()
