"""Tests du module de comparaison multi-modèles."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from evaluation_dictee.evaluation.multi_model import (
    agreement_per_item,
    confidence_score,
    disagreement_type_summary,
    load_multi_runs,
    referral_curve_multi,
    referral_curve_with_ci,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def multi_runs_data(tmp_path: Path) -> tuple[Path, list[str]]:
    """Crée 3 runs factices : 2 copies, 4 items chacune, 3 modèles."""
    # Copie A : les 3 modèles s'accordent partout (haute confiance)
    # Copie B : divergence sur i2 et i3 (faible confiance)
    truths_a = {"i1": "1", "i2": "9", "i3": "1", "i4": "0"}
    preds_a = {
        "run1": {"i1": "1", "i2": "9", "i3": "1", "i4": "0"},  # parfait
        "run2": {"i1": "1", "i2": "9", "i3": "1", "i4": "0"},  # parfait
        "run3": {"i1": "1", "i2": "9", "i3": "1", "i4": "0"},  # parfait
    }
    truths_b = {"i1": "1", "i2": "9", "i3": "1", "i4": "9"}
    preds_b = {
        "run1": {"i1": "1", "i2": "9", "i3": "9", "i4": "1"},  # 2 désaccords
        "run2": {"i1": "1", "i2": "1", "i3": "1", "i4": "1"},  # 2 désaccords
        "run3": {"i1": "1", "i2": "9", "i3": "1", "i4": "9"},  # parfait
    }

    for run in ["run1", "run2", "run3"]:
        records = []
        for it, truth in truths_a.items():
            records.append(
                {
                    "copy_id": "A",
                    "item_id": it,
                    "y_true": truth,
                    "y_pred": preds_a[run][it],
                    "confidence": 0.9,
                }
            )
        for it, truth in truths_b.items():
            records.append(
                {
                    "copy_id": "B",
                    "item_id": it,
                    "y_true": truth,
                    "y_pred": preds_b[run][it],
                    "confidence": 0.8,
                }
            )
        _write_jsonl(tmp_path / f"{run}_predictions.jsonl", records)

    return tmp_path, ["run1", "run2", "run3"]


def test_load_multi_runs_joins_correctly(multi_runs_data) -> None:
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    assert len(df) == 8  # 2 copies × 4 items
    for run in runs:
        assert f"y_pred__{run}" in df.columns
        assert f"conf__{run}" in df.columns
    assert "y_true" in df.columns


def test_load_missing_run_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_multi_runs(["inexistant"], output_dir=tmp_path)


def test_agreement_per_item(multi_runs_data) -> None:
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    agree = agreement_per_item(df)

    # Copie A : unanimité partout (les 3 modèles s'accordent)
    a_rows = agree[agree["copy_id"] == "A"]
    assert (a_rows["unanimite"]).all()
    assert (a_rows["n_accord_modeles"] == 3).all()

    # Copie B : sur i1 les 3 s'accordent, sur les autres il y a du désaccord
    b_rows = agree[agree["copy_id"] == "B"]
    i1 = b_rows[b_rows["item_id"] == "i1"].iloc[0]
    assert i1["unanimite"] is True or i1["unanimite"]  # unanimité sur i1
    # Sur i4 : run1 dit "1", run2 dit "1", run3 dit "9" → modale = "1", 2 accords
    i4 = b_rows[b_rows["item_id"] == "i4"].iloc[0]
    assert i4["n_accord_modeles"] == 2
    assert i4["modal_pred"] == "1"


def test_confidence_score_ordering(multi_runs_data) -> None:
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    agree = agreement_per_item(df)
    conf = confidence_score(agree)

    # Copie A doit avoir un score plus élevé que B (unanimité vs divergence)
    assert conf.loc["A", "score_confiance"] > conf.loc["B", "score_confiance"]
    # A a 100 % d'unanimité, B moins
    assert conf.loc["A", "pct_unanime"] == 100.0


def test_referral_curve_monotonicity(multi_runs_data) -> None:
    """Plus le seuil monte, plus on renvoie de copies (monotonie)."""
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    agree = agreement_per_item(df)
    conf = confidence_score(agree)
    curve = referral_curve_multi(agree, conf, reference_run="run1")

    pct = curve["pct_copies_renvoyees"].values
    assert all(pct[i] <= pct[i + 1] for i in range(len(pct) - 1)), f"Non monotone : {pct}"


def test_disagreement_type_summary(multi_runs_data) -> None:
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    agree = agreement_per_item(df)
    summary = disagreement_type_summary(agree)

    assert set(summary.index).issubset({1, 2, 3})
    assert abs(summary["pct_items"].sum() - 100.0) < 1e-6


def test_referral_curve_with_ci_has_bounds(multi_runs_data) -> None:
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    agree = agreement_per_item(df)
    conf = confidence_score(agree)
    curve = referral_curve_with_ci(agree, conf, reference_run="run1")
    assert "accord_lo" in curve.columns
    assert "accord_hi" in curve.columns
    # Les bornes encadrent la valeur estimée (là où elle est définie)
    ok = curve.dropna(subset=["pct_accord_retenues"])
    assert (ok["accord_lo"] <= ok["pct_accord_retenues"]).all()
    assert (ok["pct_accord_retenues"] <= ok["accord_hi"]).all()


def test_single_run_fallback(tmp_path: Path) -> None:
    """N=1 : le score de confiance est constant, la structure marche quand même."""
    records = [
        {"copy_id": "A", "item_id": "i1", "y_true": "1", "y_pred": "1", "confidence": 0.9},
        {"copy_id": "A", "item_id": "i2", "y_true": "9", "y_pred": "9", "confidence": 0.9},
    ]
    _write_jsonl(tmp_path / "solo_predictions.jsonl", records)
    df = load_multi_runs(["solo"], output_dir=tmp_path)
    agree = agreement_per_item(df)
    conf = confidence_score(agree)
    # Avec un seul modèle, tous les items sont "unanimes" par construction
    assert conf.loc["A", "score_confiance"] == 100.0


def _joint(preds: dict[str, list[str]], y_true: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"copy_id": [f"c{i}" for i in range(len(y_true))], "y_true": y_true})
    for run, codes in preds.items():
        df[f"y_pred__{run}"] = codes
    return df


def test_pairwise_agreement_symetrique() -> None:
    from evaluation_dictee.evaluation.multi_model import pairwise_agreement

    df = _joint({"a": ["1", "9", "1", "0"], "b": ["1", "9", "9", "0"]}, ["1", "9", "1", "0"])
    accord, kappa = pairwise_agreement(df)
    assert accord.loc["a", "b"] == accord.loc["b", "a"] == 0.75
    assert accord.loc["a", "a"] == 1.0
    assert kappa.loc["a", "b"] == pytest.approx(kappa.loc["b", "a"])
    assert 0 < kappa.loc["a", "b"] < 1


def test_fleiss_kappa_egal_a_un_si_accord_parfait() -> None:
    from evaluation_dictee.evaluation.multi_model import fleiss_kappa

    codes = ["1", "9", "0", "1"]
    df = _joint({"a": codes, "b": codes, "c": codes}, codes)
    assert fleiss_kappa(df) == pytest.approx(1.0)


def test_fleiss_kappa_deux_runs_proche_de_cohen() -> None:
    # À deux codeurs, Fleiss vaut le kappa de Scott : proche de Cohen quand les
    # marges des deux runs se ressemblent.
    from sklearn.metrics import cohen_kappa_score

    from evaluation_dictee.evaluation.multi_model import fleiss_kappa

    a = ["1"] * 60 + ["9"] * 30 + ["0"] * 10
    b = ["1"] * 55 + ["9"] * 5 + ["9"] * 28 + ["1"] * 2 + ["0"] * 10
    df = _joint({"a": a, "b": b}, a)
    assert fleiss_kappa(df) == pytest.approx(cohen_kappa_score(a, b), abs=0.01)


def test_consensus_levels_distingue_egalite() -> None:
    from evaluation_dictee.evaluation.multi_model import consensus_levels

    df = _joint(
        {
            "a": ["1", "1", "1", "1"],
            "b": ["1", "1", "9", "9"],
            "c": ["1", "1", "1", "0"],
            "d": ["1", "9", "9", "0"],
        },
        ["1", "1", "9", "1"],
    )
    niveaux = consensus_levels(agreement_per_item(df)).set_index(["n_accord_modeles", "egalite"])
    # Item 1 unanime, item 2 trois contre un, item 3 deux contre deux, item 4 2-1-1.
    assert niveaux.loc[(4, False), "n_items"] == 1
    assert niveaux.loc[(3, False), "accord_majoritaire"] == 1.0
    assert math.isnan(niveaux.loc[(2, True), "accord_majoritaire"])
    assert niveaux.loc[(2, True), "accord__b"] == 1.0
    assert niveaux.loc[(2, False), "n_items"] == 1
    assert niveaux["pct_items"].sum() == pytest.approx(100)
