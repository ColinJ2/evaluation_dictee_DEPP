"""Tests des outils statistiques et du rapport par item."""

import math

import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from evaluation_dictee.evaluation.calibration import (
    expected_calibration_error,
    reliability_bins,
)
from evaluation_dictee.evaluation.report import (
    disagreement_decomposition,
    per_copy_metrics,
    per_item_metrics,
)
from evaluation_dictee.evaluation.statistics import (
    cluster_bootstrap,
    design_effect,
    kappa_interval,
    kappa_interval_clustered,
    wilson_interval,
)


# ── Wilson ────────────────────────────────────────────────────────────────────
def test_wilson_proportion_mediane() -> None:
    ci = wilson_interval(50, 100)
    assert ci.estimate == 0.5
    assert ci.lower < 0.5 < ci.upper
    assert 0.39 < ci.lower < 0.41  # valeur connue ≈ 0.404


def test_wilson_bornes_dans_0_1() -> None:
    ci = wilson_interval(100, 100)
    assert ci.upper <= 1.0
    ci0 = wilson_interval(0, 100)
    assert ci0.lower >= 0.0


def test_wilson_n_nul_leve_erreur() -> None:
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


# ── Kappa et son IC analytique ────────────────────────────────────────────────
def test_kappa_interval_encadre_estimation() -> None:
    y_true = ["1"] * 60 + ["9"] * 40
    y_pred = ["1"] * 55 + ["9"] * 5 + ["9"] * 35 + ["1"] * 5
    ci = kappa_interval(y_true, y_pred)
    assert 0 < ci.estimate < 1
    assert ci.lower < ci.estimate < ci.upper


def test_kappa_interval_erreurs_rares_non_degenere() -> None:
    """Régression : l'ancienne formule (terme diagonal non élevé au carré, indices
    hors diagonale croisés) rendait un IC de largeur NULLE quand les erreurs sont
    rares — le cas de la plupart des items de la dictée. Bornes de référence :
    bootstrap de 1 000 rééchantillonnages, [0,196 ; 0,396].
    """
    y_true = ["1"] * 3300 + ["9"] * 100
    y_pred = ["1"] * 3290 + ["9"] * 10 + ["1"] * 80 + ["9"] * 20
    ci = kappa_interval(y_true, y_pred)
    assert ci.lower == pytest.approx(0.196, abs=0.01)
    assert ci.upper == pytest.approx(0.396, abs=0.01)


def test_kappa_interval_trois_codes_proche_du_bootstrap() -> None:
    """Trois catégories (1/9/0) : bornes de référence par bootstrap, [0,578 ; 0,665]."""
    y_true = ["1"] * 500 + ["9"] * 300 + ["0"] * 200
    y_pred = (
        ["1"] * 420
        + ["9"] * 60
        + ["0"] * 20
        + ["9"] * 200
        + ["1"] * 80
        + ["0"] * 20
        + ["0"] * 150
        + ["1"] * 30
        + ["9"] * 20
    )
    ci = kappa_interval(y_true, y_pred)
    assert ci.lower == pytest.approx(0.578, abs=0.01)
    assert ci.upper == pytest.approx(0.665, abs=0.01)


def test_kappa_interval_accord_parfait_interval_serre() -> None:
    y = ["1"] * 50 + ["9"] * 50
    ci = kappa_interval(y, y)
    assert ci.estimate == pytest.approx(1.0)
    assert ci.upper - ci.lower < 1e-9


def test_kappa_interval_une_seule_categorie_degenere() -> None:
    # p_e = 1 : la variance n'est pas définie et sklearn renvoie un kappa NaN.
    ci = kappa_interval(["1"] * 50, ["1"] * 50)
    assert math.isnan(ci.estimate)
    assert math.isnan(ci.lower)
    assert math.isnan(ci.upper)


def test_kappa_interval_longueurs_incoherentes() -> None:
    with pytest.raises(ValueError):
        kappa_interval(["1", "9"], ["1"])


def test_kappa_interval_series_vides() -> None:
    with pytest.raises(ValueError):
        kappa_interval([], [])


# ── Design effect ─────────────────────────────────────────────────────────────
def test_design_effect_grappes_homogenes_est_maximal() -> None:
    # Deux copies parfaitement homogènes : toute la variance est inter-copies,
    # l'ICC vaut 1 et le design effect atteint la taille de grappe.
    grappes = ["A"] * 10 + ["B"] * 10
    indicatrice = [0] * 10 + [1] * 10
    assert design_effect(pd.Series(indicatrice), pd.Series(grappes)) == pytest.approx(
        10.0, rel=1e-6
    )


def test_design_effect_sans_variance_vaut_un() -> None:
    grappes = ["A"] * 5 + ["B"] * 5
    assert design_effect(pd.Series([1] * 10), pd.Series(grappes)) == 1.0


def test_design_effect_minore_a_un() -> None:
    # Grappes volontairement mélangées : l'ICC estimé est négatif, on borne à 1.
    grappes = ["A", "A", "B", "B", "C", "C"]
    assert design_effect(pd.Series([0, 1, 0, 1, 0, 1]), pd.Series(grappes)) == 1.0


def test_design_effect_propre_a_chaque_indicatrice() -> None:
    # Deux indicatrices sur les MÊMES grappes n'ont pas le même design effect :
    # la première est parfaitement groupée, la seconde alterne dans chaque copie.
    grappes = pd.Series(["A"] * 4 + ["B"] * 4)
    groupee = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    alternee = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    assert design_effect(groupee, grappes) > design_effect(alternee, grappes)
    assert design_effect(alternee, grappes) == 1.0


def test_design_effect_longueurs_incoherentes() -> None:
    with pytest.raises(ValueError):
        design_effect(pd.Series([0, 1]), pd.Series(["A"]))


# ── Wilson corrigé du design effect ───────────────────────────────────────────
def test_wilson_deff_elargit_intervalle() -> None:
    brut = wilson_interval(500, 1000)
    corrige = wilson_interval(500, 1000, deff=4.0)
    assert corrige.estimate == brut.estimate  # la valeur ponctuelle ne change pas
    largeur_brute = brut.upper - brut.lower
    largeur_corrigee = corrige.upper - corrige.lower
    # deff = 4 → intervalle élargi d'un facteur ≈ √4 = 2.
    assert largeur_corrigee / largeur_brute == pytest.approx(2.0, rel=0.02)


def test_wilson_deff_neutre_a_un() -> None:
    assert wilson_interval(300, 1000, deff=1.0) == wilson_interval(300, 1000)


def test_wilson_deff_invalide() -> None:
    with pytest.raises(ValueError):
        wilson_interval(500, 1000, deff=0.5)


# ── Kappa par bootstrap de grappes ────────────────────────────────────────────
def _df_kappa_groupe() -> pd.DataFrame:
    """20 copies de 10 items : la moitié bien codées, l'autre systématiquement ratée."""
    lignes = []
    for c in range(20):
        for i in range(10):
            vrai = "1" if i % 2 else "9"
            predit = vrai if c % 2 == 0 else ("9" if vrai == "1" else "1")
            lignes.append({"copy_id": f"c{c}", "y_true": vrai, "y_pred": predit})
    return pd.DataFrame(lignes)


def test_kappa_clustered_encadre_estimation() -> None:
    df = _df_kappa_groupe()
    ci = kappa_interval_clustered(df["y_true"], df["y_pred"], df["copy_id"], n_boot=200)
    assert ci.lower <= ci.estimate <= ci.upper


def test_kappa_clustered_plus_large_que_delta_method() -> None:
    # Les items étant fortement corrélés dans chaque copie, le bootstrap par
    # grappes doit produire un intervalle plus large que la delta method, qui
    # suppose l'indépendance.
    df = _df_kappa_groupe()
    grappes = kappa_interval_clustered(df["y_true"], df["y_pred"], df["copy_id"], n_boot=400)
    naif = kappa_interval(df["y_true"], df["y_pred"])
    assert (grappes.upper - grappes.lower) > (naif.upper - naif.lower)


def test_kappa_clustered_reproductible() -> None:
    df = _df_kappa_groupe()
    args = (df["y_true"], df["y_pred"], df["copy_id"])
    a = kappa_interval_clustered(*args, n_boot=100, seed=7)
    b = kappa_interval_clustered(*args, n_boot=100, seed=7)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_kappa_clustered_longueurs_incoherentes() -> None:
    with pytest.raises(ValueError):
        kappa_interval_clustered(["1", "9"], ["1", "9"], ["c1"])


# ── Bootstrap par grappes ─────────────────────────────────────────────────────
def _df_jouet() -> pd.DataFrame:
    rows = []
    for copie in ["c1", "c2", "c3", "c4"]:
        for i in range(10):
            ok = i < 9  # 90 % d'accord
            rows.append(
                {
                    "copy_id": copie,
                    "item_id": f"it{i}",
                    "y_true": "1",
                    "y_pred": "1" if ok else "9",
                    "confidence": 0.9 if ok else 0.4,
                }
            )
    return pd.DataFrame(rows)


def test_cluster_bootstrap_contient_estimation() -> None:
    df = _df_jouet()
    ci = cluster_bootstrap(df, lambda d: float((d["y_true"] == d["y_pred"]).mean()), n_boot=200)
    assert ci.lower <= ci.estimate <= ci.upper
    assert ci.estimate == 0.9


def test_cluster_bootstrap_reproductible() -> None:
    df = _df_jouet()
    fn = lambda d: float((d["y_true"] == d["y_pred"]).mean())  # noqa: E731
    ci1 = cluster_bootstrap(df, fn, n_boot=100, seed=7)
    ci2 = cluster_bootstrap(df, fn, n_boot=100, seed=7)
    assert (ci1.lower, ci1.upper) == (ci2.lower, ci2.upper)


# ── Rapport par item / copie ──────────────────────────────────────────────────
def test_per_item_metrics_colonnes() -> None:
    stats = per_item_metrics(_df_jouet())
    for col in ["accord", "accord_lo", "accord_hi", "kappa", "kappa_lo", "kappa_hi"]:
        assert col in stats.columns
    assert (stats["accord_lo"] <= stats["accord"]).all()
    assert (stats["accord"] <= stats["accord_hi"]).all()
    definis = stats.dropna(subset=["kappa"])
    assert (definis["kappa_lo"] <= definis["kappa"]).all()
    assert (definis["kappa"] <= definis["kappa_hi"]).all()


def test_kappa_par_item_et_son_intervalle() -> None:
    """Kappa d'un item = celui de ses seules observations, IC qui l'encadre."""
    y_true = ["1"] * 60 + ["9"] * 40
    y_pred = ["1"] * 55 + ["9"] * 5 + ["9"] * 30 + ["1"] * 10
    df = pd.DataFrame(
        {
            "copy_id": [f"c{i}" for i in range(100)],
            "item_id": ["i1"] * 100,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    stats = per_item_metrics(df)
    attendu = cohen_kappa_score(y_true, y_pred)
    assert stats.loc["i1", "kappa"] == pytest.approx(attendu)
    assert stats.loc["i1", "kappa_lo"] < attendu < stats.loc["i1", "kappa_hi"]


def test_sur_correction_et_sur_detection() -> None:
    df = pd.DataFrame(
        {
            "copy_id": ["c"] * 4,
            "item_id": ["i1", "i1", "i1", "i1"],
            # expert: erreur, modèle: correct → sur-correction (1 cas)
            # expert: correct, modèle: erreur → sur-détection (1 cas)
            "y_true": ["9", "1", "1", "9"],
            "y_pred": ["1", "9", "1", "9"],
            "confidence": [0.8] * 4,
        }
    )
    stats = per_item_metrics(df)
    assert stats.loc["i1", "n_sur_correction"] == 1
    assert stats.loc["i1", "n_sur_detection"] == 1
    # 2 erreurs expert, 1 retrouvée → rappel 0.5 ; 2 erreurs modèle, 1 confirmée → précision 0.5
    assert stats.loc["i1", "rappel_erreur"] == 0.5
    assert stats.loc["i1", "precision_erreur"] == 0.5
    # Taux : 1 sur-correction parmi 2 fautes expert, 1 sur-détection parmi 2 corrects.
    assert stats.loc["i1", "taux_sur_correction"] == 0.5
    assert stats.loc["i1", "taux_sur_detection"] == 0.5
    assert (
        stats.loc["i1", "taux_sur_correction_lo"] < 0.5 < stats.loc["i1", "taux_sur_correction_hi"]
    )


def test_taux_sur_correction_non_defini_sans_faute_expert() -> None:
    df = pd.DataFrame(
        {
            "copy_id": ["a", "b"],
            "item_id": ["i1", "i1"],
            "y_true": ["1", "1"],
            "y_pred": ["1", "9"],
        }
    )
    stats = per_item_metrics(df)
    assert math.isnan(stats.loc["i1", "taux_sur_correction"])
    assert stats.loc["i1", "taux_sur_detection"] == 0.5


def test_biais_net_compense_les_deux_erreurs() -> None:
    # 2 sur-détections, 1 sur-correction sur 10 copies : biais net = +0,1.
    df = pd.DataFrame(
        {
            "copy_id": [f"c{i}" for i in range(10)],
            "item_id": ["i1"] * 10,
            "y_true": ["1", "1", "9"] + ["1"] * 4 + ["9"] * 3,
            "y_pred": ["9", "9", "1"] + ["1"] * 4 + ["9"] * 3,
        }
    )
    stats = per_item_metrics(df)
    assert stats.loc["i1", "biais_net"] == pytest.approx(0.1)
    assert stats.loc["i1", "biais_net_lo"] < 0.1 < stats.loc["i1", "biais_net_hi"]


def test_per_copy_metrics() -> None:
    stats = per_copy_metrics(_df_jouet())
    assert len(stats) == 4
    assert (stats["accord"] == 0.9).all()


def test_disagreement_decomposition() -> None:
    deco = disagreement_decomposition(_df_jouet())
    assert deco["n"].sum() == 4  # un désaccord par copie
    assert deco["pct_desaccords"].sum() == pytest.approx(100.0)


# ── Calibration ───────────────────────────────────────────────────────────────
def test_ece_modele_parfaitement_calibre() -> None:
    # confiance 1.0 et toujours juste → ECE = 0
    ece = expected_calibration_error(["1"] * 10, ["1"] * 10, [1.0] * 10)
    assert ece == pytest.approx(0.0)


def test_ece_modele_surconfiant() -> None:
    # confiance 0.95 mais 50 % d'accord seulement → ECE ≈ 0.45
    y_true = ["1"] * 5 + ["9"] * 5
    y_pred = ["1"] * 10
    ece = expected_calibration_error(y_true, y_pred, [0.95] * 10)
    assert ece == pytest.approx(0.45, abs=0.01)


def test_reliability_bins_couvrent_les_donnees() -> None:
    bins = reliability_bins(["1", "1"], ["1", "9"], [0.2, 0.99], n_bins=10)
    assert sum(b.n for b in bins) == 2
