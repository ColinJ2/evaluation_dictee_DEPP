"""Compare deux runs portant sur des copies DIFFÉRENTES (ex. dictée 2015 vs 2021).

`rapport_bras.py` compare des variantes sur les MÊMES copies (écarts appariés). Ici les
échantillons sont indépendants : les écarts et leurs intervalles viennent d'un bootstrap
par copie mené séparément sur chaque run. Comme le kappa dépend de la prévalence des
fautes, le script donne aussi les métriques par niveau d'élève (tiers de taux de faute
expert, bornes communes aux deux runs) : un écart global peut venir des élèves, pas du
modèle.

Sortie : agrégats seulement (aucune transcription ni identifiant de copie).

Usage :
    uv run scripts/comparer_runs.py --runs 2015=dictee2015_end2end_alea \
        2021=dictee2021_end2end_alea
    uv run scripts/comparer_runs.py --runs 2015=dictee_end2end 2021=dictee2021_end2end \
        --modele gemma4-26b-moe
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_dictee.data import reference
from evaluation_dictee.data.grid import load_grid
from evaluation_dictee.evaluation.report import filtrer_evaluables, load_predictions

CODES = [reference.SIMPLE_CORRECT, reference.SIMPLE_ERREUR, reference.SIMPLE_ABSENT]
OK, ERR, ABS = 0, 1, 2  # positions de CODES dans les matrices de confusion

METRIQUES = {
    "accord": "Accord brut",
    "kappa": "Kappa de Cohen",
    "kappa_mots": "Kappa mots",
    "kappa_ponct": "Kappa ponctuation",
    "sous_detection": "Fautes non détectées",
    "precision_faute": "Précision sur « erreur »",
    "fausses_fautes": "Correct codé erreur",
    "absents_manques": "Absents non détectés",
    "biais_pts": "Biais taux de faute (pts)",
    "prev_faute_expert": "Taux de faute (expert)",
    "prev_absent_expert": "Taux d'absents (expert)",
}
POURCENTS = {
    "accord",
    "sous_detection",
    "precision_faute",
    "fausses_fautes",
    "absents_manques",
    "prev_faute_expert",
    "prev_absent_expert",
}


@dataclass
class Run:
    """Prédictions évaluables d'un run et leurs matrices de confusion par copie."""

    libelle: str
    df: pd.DataFrame
    copies: np.ndarray
    # (n_copies, 3, 3) : toutes natures, mots seuls, ponctuation seule.
    conf: dict[str, np.ndarray]


def _confusions(df: pd.DataFrame, copies: np.ndarray) -> np.ndarray:
    """Empile une matrice de confusion 3×3 (codes simplifiés) par copie."""
    pos = {c: i for i, c in enumerate(copies)}
    idx = {c: i for i, c in enumerate(CODES)}
    sub = df[df["y_true"].isin(idx) & df["y_pred"].isin(idx)]
    out = np.zeros((len(copies), 3, 3))
    np.add.at(
        out,
        (
            sub["copy_id"].map(pos).to_numpy(),
            sub["y_true"].map(idx).to_numpy(),
            sub["y_pred"].map(idx).to_numpy(),
        ),
        1,
    )
    return out


def _kappa(c: np.ndarray) -> float:
    n = c.sum()
    if n == 0:
        return float("nan")
    p_o = np.trace(c) / n
    p_e = float((c.sum(axis=1) * c.sum(axis=0)).sum()) / n**2
    return float("nan") if p_e == 1 else float((p_o - p_e) / (1 - p_e))


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def metriques(conf: dict[str, np.ndarray], choix: np.ndarray | None = None) -> dict[str, float]:
    """Métriques d'un run (ou d'un tirage bootstrap `choix` de ses copies)."""
    tot = {k: (v if choix is None else v[choix]).sum(axis=0) for k, v in conf.items()}
    c = tot["tout"]
    n = c.sum()
    lignes, colonnes = c.sum(axis=1), c.sum(axis=0)
    return {
        "accord": _ratio(np.trace(c), n),
        "kappa": _kappa(c),
        "kappa_mots": _kappa(tot["mot"]),
        "kappa_ponct": _kappa(tot["ponctuation"]),
        "sous_detection": _ratio(c[ERR, OK], lignes[ERR]),
        "precision_faute": _ratio(c[ERR, ERR], colonnes[ERR]),
        "fausses_fautes": _ratio(c[OK, ERR], lignes[OK]),
        "absents_manques": _ratio(c[ABS, OK] + c[ABS, ERR], lignes[ABS]),
        "biais_pts": _ratio(colonnes[ERR] - lignes[ERR], n) * 100,
        "prev_faute_expert": _ratio(lignes[ERR], n),
        "prev_absent_expert": _ratio(lignes[ABS], n),
    }


def bootstrap(run: Run, n_boot: int, seed: int) -> pd.DataFrame:
    """Distribution bootstrap (copies rééchantillonnées) de toutes les métriques."""
    rng = np.random.default_rng(seed)
    n = len(run.copies)
    return pd.DataFrame([metriques(run.conf, rng.integers(0, n, n)) for _ in range(n_boot)])


def charger(libelle: str, chemin: Path, types: dict[str, str]) -> Run:
    """Charge un JSONL, applique les exclusions D8 et prépare les matrices par copie."""
    brut = load_predictions(chemin)
    df, retirees = filtrer_evaluables(brut)
    df = df.assign(
        y_true=df["y_true"].astype(str),
        y_pred=df["y_pred"].astype(str),
        nature=df["item_id"].map(types),
    )
    copies = np.array(sorted(df["copy_id"].unique()))
    conf = {"tout": _confusions(df, copies)}
    for nature in ("mot", "ponctuation"):
        conf[nature] = _confusions(df[df["nature"] == nature], copies)
    hors_grille = int((~df["y_pred"].isin(CODES)).sum())
    print(
        f"- **{libelle}** : {brut['copy_id'].nunique()} copies lues, {len(copies)} retenues "
        f"({len(df)} items) ; copies écartées : {retirees or 'aucune'} ; "
        f"prédictions hors grille (ex. « ? ») : {hors_grille}"
    )
    return Run(libelle, df, copies, conf)


def _fmt(cle: str, x: float) -> str:
    if np.isnan(x):
        return "—"
    if cle in POURCENTS:
        return f"{x * 100:.1f} %"
    return f"{x:+.1f}" if cle == "biais_pts" else f"{x:.3f}"


def _fmt_ecart(cle: str, x: float) -> str:
    if np.isnan(x):
        return "—"
    if cle in POURCENTS:
        return f"{x * 100:+.1f} pt"
    return f"{x:+.1f} pt" if cle == "biais_pts" else f"{x:+.3f}"


def section_globale(a: Run, b: Run, n_boot: int) -> None:
    """Tableau principal : chaque métrique, son IC 95 %, et l'écart b − a avec son IC."""
    ma, mb = metriques(a.conf), metriques(b.conf)
    ba, bb = bootstrap(a, n_boot, 42), bootstrap(b, n_boot, 43)
    diff = bb - ba
    print(f"\n## Métriques globales (IC 95 % bootstrap par copie, {n_boot} tirages)\n")
    print(
        f"| Métrique | {a.libelle} | {b.libelle} "
        f"| Écart {b.libelle} − {a.libelle} | IC 95 % écart |"
    )
    print("|---|---|---|---|---|")
    for cle, nom in METRIQUES.items():
        lo_a, hi_a = ba[cle].quantile([0.025, 0.975])
        lo_b, hi_b = bb[cle].quantile([0.025, 0.975])
        lo_d, hi_d = diff[cle].quantile([0.025, 0.975])
        signif = " **" if (lo_d > 0 or hi_d < 0) else ""
        print(
            f"| {nom} | {_fmt(cle, ma[cle])} [{_fmt(cle, lo_a)} ; {_fmt(cle, hi_a)}] "
            f"| {_fmt(cle, mb[cle])} [{_fmt(cle, lo_b)} ; {_fmt(cle, hi_b)}] "
            f"| {_fmt_ecart(cle, mb[cle] - ma[cle])}{signif} "
            f"| [{_fmt_ecart(cle, lo_d)} ; {_fmt_ecart(cle, hi_d)}] |"
        )
    print("\n`**` : l'intervalle de l'écart exclut 0.")


def section_confusion(a: Run, b: Run) -> None:
    """Matrices de confusion normalisées par ligne (code expert), côte à côte."""
    print("\n## Matrices de confusion (% de la ligne expert)\n")
    print(
        f"| Expert ↓ / Modèle → | {a.libelle} : 1 / err / 0 | {b.libelle} : 1 / err / 0 "
        f"| n expert {a.libelle} / {b.libelle} |"
    )
    print("|---|---|---|---|")
    ca, cb = a.conf["tout"].sum(axis=0), b.conf["tout"].sum(axis=0)
    for i, nom in enumerate(["1 (correct)", "err (faute)", "0 (absent)"]):
        la = " / ".join(f"{x:.1f}" for x in ca[i] / max(ca[i].sum(), 1) * 100)
        lb = " / ".join(f"{x:.1f}" for x in cb[i] / max(cb[i].sum(), 1) * 100)
        print(f"| {nom} | {la} | {lb} | {int(ca[i].sum())} / {int(cb[i].sum())} |")


def section_niveaux(a: Run, b: Run) -> None:
    """Métriques par tiers de niveau d'élève, bornes calculées sur les deux runs réunis."""
    taux = {
        r.libelle: r.conf["tout"][:, ERR, :].sum(axis=1) / r.conf["tout"].sum(axis=(1, 2))
        for r in (a, b)
    }
    bornes = np.quantile(np.concatenate(list(taux.values())), [1 / 3, 2 / 3])
    noms = [
        f"bons (< {bornes[0] * 100:.0f} % de fautes)",
        f"moyens ({bornes[0] * 100:.0f}–{bornes[1] * 100:.0f} %)",
        f"faibles (> {bornes[1] * 100:.0f} %)",
    ]
    print("\n## Par niveau d'élève (tiers du taux de faute expert, bornes communes)\n")
    print(
        "Contrôle de composition : si les écarts globaux disparaissent à niveau égal, "
        "ils viennent des élèves et non du modèle.\n"
    )
    print(
        f"| Niveau | Copies {a.libelle} / {b.libelle} | Accord | Kappa "
        "| Fautes non détectées | Correct codé erreur |"
    )
    print("|---|---|---|---|---|---|")
    for k, nom in enumerate(noms):
        cellules = []
        effectifs = []
        for r in (a, b):
            strate = np.digitize(taux[r.libelle], bornes) == k
            effectifs.append(str(int(strate.sum())))
            cellules.append(metriques({n: v[strate] for n, v in r.conf.items()}))
        ligne = [nom, " / ".join(effectifs)]
        for cle in ("accord", "kappa", "sous_detection", "fausses_fautes"):
            ligne.append(" → ".join(_fmt(cle, m[cle]) for m in cellules))
        print("| " + " | ".join(ligne) + " |")


def section_items(a: Run, b: Run, grille: dict[str, tuple[str, str]], top: int) -> None:
    """Items dont l'accord change le plus d'une année à l'autre."""

    def par_item(r: Run) -> pd.DataFrame:
        indic = pd.DataFrame(
            {
                "accord": r.df["y_true"] == r.df["y_pred"],
                "faute": r.df["y_true"] == reference.SIMPLE_ERREUR,
            }
        )
        return indic.groupby(r.df["item_id"]).mean()

    ia, ib = par_item(a), par_item(b)
    comp = ia.join(ib, lsuffix="_a", rsuffix="_b", how="inner")
    comp["ecart"] = comp["accord_b"] - comp["accord_a"]
    ordre = {item: i for i, item in enumerate(grille)}
    print(
        f"\n## Items qui bougent le plus (accord {b.libelle} − {a.libelle} ; "
        f"~200 copies par item : un écart < 10 pts est du bruit)\n"
    )
    print(
        f"| Rang | Item | Nature | Accord {a.libelle} → {b.libelle} | Écart "
        f"| Taux de faute expert {a.libelle} → {b.libelle} |"
    )
    print("|---|---|---|---|---|---|")
    extremes = pd.concat([comp.nsmallest(top, "ecart"), comp.nlargest(top, "ecart")])
    for item, row in extremes.sort_values("ecart").iterrows():
        attendu, nature = grille.get(item, ("?", "?"))
        print(
            f"| {ordre.get(item, -1) + 1} | {attendu} | {nature} "
            f"| {row.accord_a * 100:.0f} % → {row.accord_b * 100:.0f} % "
            f"| {row.ecart * 100:+.0f} pt "
            f"| {row.faute_a * 100:.0f} % → {row.faute_b * 100:.0f} % |"
        )
    # Corrélation : l'écart d'accord suit-il l'écart de taux de faute de l'item ?
    corr = np.corrcoef(comp["ecart"], comp["faute_b"] - comp["faute_a"])[0, 1]
    print(
        f"\nCorrélation, sur les {len(comp)} items, entre écart d'accord et écart de taux de "
        f"faute : {corr:+.2f}"
    )


def section_confiance(a: Run, b: Run) -> None:
    """Disponibilité des scores de confiance (la courbe de renvoi en dépend)."""
    print("\n## Scores de confiance\n")
    print("| Run | Items avec confiance | Moyenne | Médiane | Part ≤ 0,1 |")
    print("|---|---|---|---|---|")
    for r in (a, b):
        conf = pd.to_numeric(r.df.get("confidence"), errors="coerce")
        dispo = conf.notna().mean() if conf is not None else 0.0
        c = conf.dropna() if conf is not None else pd.Series(dtype=float)
        print(
            f"| {r.libelle} | {dispo * 100:.1f} % | {c.mean():.3f} | {c.median():.3f} "
            f"| {(c <= 0.1).mean() * 100:.1f} % |"
            if len(c)
            else f"| {r.libelle} | 0 % | — | — | — |"
        )


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        nargs=2,
        required=True,
        metavar="LIBELLÉ=RUN",
        help="Deux runs, sous la forme `libellé=config.name` (le modèle est ajouté).",
    )
    parser.add_argument("--modele", default="qwen3-6-35b-moe", help="Suffixe modèle du run.")
    parser.add_argument("--dossier", default="data/processed", help="Dossier des prédictions.")
    parser.add_argument("--grille", default="configs/grille_dictee_2015.json")
    parser.add_argument("--n-boot", type=int, default=2000, help="Tirages bootstrap.")
    parser.add_argument("--top-items", type=int, default=8, help="Items extrêmes par sens.")
    args = parser.parse_args()

    grille_obj = load_grid(args.grille)
    grille = {it.item_id: (it.attendu, it.type) for it in grille_obj.items}
    types = {k: v[1] for k, v in grille.items()}

    print("# Comparaison de runs sur échantillons indépendants\n")
    runs = []
    for paire in args.runs:
        libelle, nom = paire.split("=", 1)
        runs.append(
            charger(libelle, Path(args.dossier) / f"{nom}_{args.modele}_predictions.jsonl", types)
        )
    a, b = runs

    section_globale(a, b, args.n_boot)
    section_niveaux(a, b)
    section_confusion(a, b)
    section_items(a, b, grille, args.top_items)
    section_confiance(a, b)


if __name__ == "__main__":
    main()
