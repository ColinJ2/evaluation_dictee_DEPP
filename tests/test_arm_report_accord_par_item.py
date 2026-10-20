"""Tests de la section « accord par item, bras par bras » et de l'inventaire des runs.

Données entièrement synthétiques : des codes tirés au hasard sur les items de la grille
2015, aucune copie réelle.
"""

import json
import random
from pathlib import Path

import pytest

from evaluation_dictee.data.grid import load_grid
from evaluation_dictee.evaluation.arm_report import (
    construire_rapport,
    ecart_apparie,
    inventaire_runs,
)

GRILLE = "configs/grille_dictee_2015.json"


def _ecrire_run(chemin: Path, taux_ok: float, n_copies: int = 30, seed: int = 0) -> None:
    alea = random.Random(seed)
    items = load_grid(GRILLE).items
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as f:
        for c in range(n_copies):
            for it in items:
                y_true = "1" if alea.random() < 0.8 else "9"
                y_pred = y_true if alea.random() < taux_ok else ("9" if y_true == "1" else "1")
                ligne = {"copy_id": f"c{c}", "item_id": it.item_id, "y_true": y_true}
                f.write(json.dumps({**ligne, "y_pred": y_pred}) + "\n")


def test_ecart_apparie_identique_vaut_zero() -> None:
    verdicts = {"a": True, "b": False, "c": True}
    assert ecart_apparie(verdicts, verdicts) == (0.0, 0.0, 0.0)


def test_ecart_apparie_signe_et_intervalle() -> None:
    reference = {f"c{i}": False for i in range(40)}
    bras = {f"c{i}": i < 30 for i in range(40)}  # 30 copies gagnées sur 40
    ecart, bas, haut = ecart_apparie(bras, reference)
    assert ecart == pytest.approx(0.75)
    assert 0 < bas < ecart < haut


def test_ecart_apparie_trop_peu_de_copies() -> None:
    assert ecart_apparie({"a": True}, {"a": False}) is None


def test_inventaire_attribue_au_prefixe_le_plus_long(tmp_path: Path) -> None:
    """`dictee_end2end_cot_m` est un run du bras CoT, pas le modèle `cot_m` (D9)."""
    for nom in [
        "dictee_end2end_modele-a_predictions.jsonl",
        "dictee_end2end_cot_modele-a_predictions.jsonl",
        "dictee_end2end_modele-a_htr_predictions.jsonl",
    ]:
        (tmp_path / nom).write_text("{}\n", encoding="utf-8")

    trouves = inventaire_runs(
        {"référence": "dictee_end2end", "CoT": "dictee_end2end_cot"}, tmp_path, prefixe_s3=""
    )
    assert set(trouves["référence"]) == {"modele-a"}
    assert set(trouves["CoT"]) == {"modele-a"}


def test_rapport_contient_la_section_par_modele(tmp_path: Path) -> None:
    _ecrire_run(tmp_path / "dictee_end2end_modele-a_predictions.jsonl", 0.85, seed=1)
    _ecrire_run(tmp_path / "dictee_end2end_exemples_modele-a_predictions.jsonl", 0.95, seed=2)

    html, rapport = construire_rapport(
        modeles=["modele-a"],
        bras={"référence": "dictee_end2end", "exemples": "dictee_end2end_exemples"},
        dossier=tmp_path,
        grille=GRILLE,
        n_boot=20,
        prefixe_s3="",
    )
    assert rapport.n_copies == 30
    assert "3. Accord par item, bras par bras" in html
    assert "Accord par item — modele-a" in html
    assert "Écart d&#x27;accord à « référence », par item — modele-a" in html
