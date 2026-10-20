"""Génère le rapport HTML de comparaison des bras d'expérience.

Un « bras » est une variante de prompt testée contre la référence (chain-of-thought,
comptage des items, exemples de fautes…). Ce rapport dit si la variante fait mieux que
la référence, sur le même modèle et les mêmes copies, et pourquoi — ce que ni Langfuse
(qui mesure des runs isolés) ni le site (qui publie ce qui est retenu) ne font.

Usage :
    uv run scripts/rapport_bras.py
    uv run scripts/rapport_bras.py --modeles qwen3-8-27b gemma4-26b-moe
    uv run scripts/rapport_bras.py --bras référence=dictee_end2end exemples=dictee_end2end_exemples
    uv run scripts/rapport_bras.py --lister                  # quels modèles pour quels bras ?
    uv run scripts/rapport_bras.py --modeles tous            # tous les modèles trouvés
    uv run scripts/rapport_bras.py --dossier autre/dossier --local-seulement
    uv run scripts/rapport_bras.py --prefixe-s3 s3://bucket/autre/predictions

Où sont cherchées les prédictions, pour chaque couple (bras, modèle) : d'abord
`--dossier`, puis `<préfixe S3>/`, puis `<préfixe S3>/experimentations/`. Le préfixe S3
vient de `--prefixe-s3`, sinon de la variable `S3_PREDICTIONS_PREFIX`, sinon de la valeur
par défaut de `config.Secrets`. Le premier fichier trouvé l'emporte.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_dictee.evaluation.arm_report import (
    BRAS_PAR_DEFAUT,
    construire_rapport,
    inventaire_runs,
)
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

MODELES_PAR_DEFAUT = ["qwen3-6-35b-moe", "qwen3-8-27b", "gemma4-26b-moe"]


def main() -> None:
    """Construit le rapport et l'écrit sur disque."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--modeles",
        nargs="+",
        default=MODELES_PAR_DEFAUT,
        help=(
            "Modèles à comparer, ou `tous` pour chaque modèle trouvé sur au moins un bras "
            "(voir --lister). [défaut : les trois modèles testés]"
        ),
    )
    parser.add_argument(
        "--bras",
        nargs="+",
        default=None,
        metavar="LIBELLÉ=PRÉFIXE",
        help=(
            "Bras à inclure, sous la forme `libellé=préfixe_du_run`. "
            f"Défaut : {', '.join(f'{k}={v}' for k, v in BRAS_PAR_DEFAUT.items())}"
        ),
    )
    parser.add_argument(
        "--reference",
        default="référence",
        help="Libellé du bras servant d'étalon aux écarts. [défaut : référence]",
    )
    parser.add_argument(
        "--dossier", default="data/processed", help="Dossier local des prédictions, lu en premier."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--prefixe-s3",
        default=None,
        help="Racine distante des prédictions. [défaut : S3_PREDICTIONS_PREFIX]",
    )
    source.add_argument(
        "--local-seulement",
        action="store_true",
        help="Ne lire que --dossier, sans aucune recherche sur S3.",
    )
    parser.add_argument(
        "--lister",
        action="store_true",
        help="Afficher les modèles disponibles pour chaque bras, et où, puis s'arrêter.",
    )
    parser.add_argument(
        "--sortie",
        default="data/processed/rapport_bras.html",
        help="Fichier HTML produit. Sous data/, donc hors de Git.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=400,
        help="Tirages du bootstrap des intervalles de confiance. [défaut : 400]",
    )
    args = parser.parse_args()

    bras = None
    if args.bras:
        bras = dict(paire.split("=", 1) for paire in args.bras)
    prefixe_s3 = "" if args.local_seulement else args.prefixe_s3

    modeles = args.modeles
    if args.lister or modeles == ["tous"]:
        inventaire = inventaire_runs(bras or BRAS_PAR_DEFAUT, args.dossier, prefixe_s3)
        if args.lister:
            for libelle, trouves in inventaire.items():
                print(f"{libelle} :")
                for modele, emplacement in sorted(trouves.items()):
                    print(f"    {modele:<28} {emplacement}")
                if not trouves:
                    print("    (aucun run trouvé)")
            return
        modeles = sorted({m for trouves in inventaire.values() for m in trouves})

    html, rapport = construire_rapport(
        modeles=modeles,
        bras=bras,
        dossier=args.dossier,
        reference=args.reference,
        n_boot=args.n_boot,
        prefixe_s3=prefixe_s3,
    )
    sortie = Path(args.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(html, encoding="utf-8")

    logger.info(
        "%d série(s) sur %d copies communes : %s",
        len(rapport.series),
        rapport.n_copies,
        ", ".join(sorted({s.bras for s in rapport.series})),
    )
    logger.info("Rapport écrit : %s", sortie)


if __name__ == "__main__":
    main()
