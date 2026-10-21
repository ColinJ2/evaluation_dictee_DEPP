"""Copie les prédictions de corpus complet sous la convention de nommage du site.

Le site Quarto ne lit que les runs nommés `dictee<AAAA>_<approche>_<prompt>_<modele>`
(voir `evaluation_dictee/run_names.py`). Les runs produits avant cette convention
portent un nom sans année ni prompt (`dictee_end2end_<modele>`, 2015) ou sans prompt
(`dictee2021_end2end_<modele>`) : ce script les COPIE sous leur nouveau nom dans le
dossier lu par le site. Les originaux ne sont ni déplacés ni supprimés.

Seuls les runs sur corpus complet sont concernés (racine de `predictions/` et de
`prediction_depp/`) : les bras d'expérience de `predictions/experimentations/` ne
sont pas publiés sur le site (décision D9).

Tous les fichiers d'un run sont copiés (prédictions, copies écartées, transcriptions).
Une copie dont la cible existe déjà est sautée : relancer le script est sans risque.

Usage :
    uv run scripts/renommer_predictions.py                # simulation : copies numérotées
    uv run scripts/renommer_predictions.py --appliquer 1  # copie la n° 1, et elle seule
"""

from __future__ import annotations

import argparse
from pathlib import PurePosixPath

import fsspec

from evaluation_dictee.run_names import PROMPT_BASE, nom_base
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

BUCKET = "projet-production-ecrits-depp"
DESTINATION = f"{BUCKET}/prediction_depp"

#: (dossier source, ancien `name` du run, année, approche) — prompt de base pour tous.
ANCIENS_RUNS: list[tuple[str, str, int, str]] = [
    (f"{BUCKET}/predictions", "dictee_end2end", 2015, "end2end"),
    (f"{BUCKET}/predictions", "dictee_two_stage", 2015, "two_stage"),
    (f"{BUCKET}/prediction_depp", "dictee2021_end2end", 2021, "end2end"),
    (f"{BUCKET}/prediction_depp", "dictee2021_two_stage", 2021, "two_stage"),
]


def copies_prevues(fs: fsspec.AbstractFileSystem) -> list[tuple[str, str]]:
    """Liste toutes les copies (source, cible), dans un ordre stable.

    Les cibles déjà présentes restent dans la liste, pour que le numéro d'un fichier
    ne change pas d'une exécution à l'autre.

    Args:
        fs: système de fichiers S3.

    Returns:
        Les couples (chemin source, chemin cible), sans préfixe `s3://`.
    """
    copies = []
    for dossier, ancien, annee, approche in ANCIENS_RUNS:
        nouveau = nom_base(annee, approche, PROMPT_BASE)
        for chemin in sorted(fs.ls(dossier, detail=False)):
            nom = PurePosixPath(chemin).name
            if not nom.startswith(f"{ancien}_") or fs.isdir(chemin):
                continue
            # Un fichier déjà conforme (ex. `dictee2021_end2end_base_…`, qui commence
            # aussi par `dictee2021_end2end_`) n'est pas un ancien run.
            if nom.startswith(f"{nouveau}_"):
                continue
            reste = nom[len(ancien) + 1 :]
            copies.append((chemin, f"{DESTINATION}/{nouveau}_{reste}"))
    return copies


def main() -> None:
    """Affiche les copies prévues, puis les effectue si `--appliquer` est passé."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--appliquer",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Copie UN seul fichier : le numéro N de la simulation. Sans cette option : "
            "simulation seule. Une copie à la fois, pour valider chaque fichier."
        ),
    )
    args = parser.parse_args()

    fs = fsspec.filesystem("s3")
    copies = copies_prevues(fs)
    existants = set(fs.ls(DESTINATION, detail=False))
    if args.appliquer is None:
        for numero, (source, cible) in enumerate(copies, 1):
            etat = "déjà copié" if cible in existants else f"{fs.size(source) / 1e6:.0f} Mo"
            print(f"[{numero}] {source}\n    → {cible}  ({etat})")
        restants = sum(cible not in existants for _, cible in copies)
        logger.info("Simulation : %d fichier(s) restant(s) à copier.", restants)
        return
    if not 1 <= args.appliquer <= len(copies):
        parser.error(f"--appliquer attend un numéro entre 1 et {len(copies)}.")
    source, cible = copies[args.appliquer - 1]
    if cible in existants:
        logger.info("Déjà copié, rien à faire : s3://%s", cible)
        return
    fs.copy(source, cible)
    fs.invalidate_cache()
    if fs.size(source) != fs.size(cible):
        raise SystemExit(f"Taille différente après copie : {cible}")
    logger.info("OK — copié et vérifié (même taille) : s3://%s", cible)


if __name__ == "__main__":
    main()
