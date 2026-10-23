"""Tests du catalogue des runs du site : année, approche, prompt et modèle lus dans le nom.

Remplace l'ancien mécanisme de promotion manuelle (`MODELES_PROMUS`, décision D10) :
le prompt est désormais un axe à part entière, lu dans le nom du run (décision D12).
Les pièges qui avaient motivé D9/D10 — un bras `dictee_end2end_cot_<modele>` lu
comme un modèle fantôme `cot_<modele>` — doivent rester impossibles.
"""

import importlib
import sys
from pathlib import Path

import pytest

WEBSITE_DIR = Path(__file__).resolve().parents[1] / "website"


@pytest.fixture
def analyse(monkeypatch, tmp_path):
    """Importe `_analyse` hors ligne, avec une liste de runs exportés contrôlée."""
    # Dossier local vide : l'import ne doit pas interroger S3.
    monkeypatch.setenv("S3_PREDICTIONS_PREFIX", str(tmp_path))
    monkeypatch.delenv("RESULTATS_MODELES", raising=False)
    monkeypatch.delenv("RESULTATS_ANNEE_REFERENCE", raising=False)
    sys.path.insert(0, str(WEBSITE_DIR))
    try:
        mod = importlib.import_module("_analyse")
        importlib.reload(mod)

        def _figer(noms: list[str]) -> None:
            """Fige les runs exportés et recalcule ce qui en découle au chargement."""
            monkeypatch.setattr(mod, "_EXPORTES", sorted(noms))
            mod.CATALOGUE = mod.catalogue()
            mod.ANNEES = sorted({cle[0] for cle in mod.CATALOGUE})
            mod.PROMPTS = mod._trier_prompts({cle[1] for cle in mod.CATALOGUE})
            mod.MODELES = sorted({cle[3] for cle in mod.CATALOGUE})
            mod.CONFIG_REFERENCE = mod._config_reference()

        mod._figer_pour_test = _figer
        yield mod
    finally:
        sys.path.remove(str(WEBSITE_DIR))
        sys.modules.pop("_analyse", None)


EXPORTES = [
    "dictee2015_end2end_base_gemma4-26b-moe",
    "dictee2015_two_stage_base_gemma4-26b-moe",
    "dictee2015_end2end_base_qwen3-6-35b-moe",
    "dictee2015_two_stage_base_qwen3-6-35b-moe",
    "dictee2015_end2end_exemples_qwen3-6-35b-moe",
    "dictee2021_end2end_base_qwen3-6-35b-moe",
    "dictee2021_two_stage_base_qwen3-6-35b-moe_gemma4-26b-moe",
    # Anciens noms et bras d'expérience : ignorés, jamais lus comme des modèles.
    "dictee_end2end_qwen3-6-35b-moe",
    "dictee_end2end_cot_qwen3-8-27b",
    "dictee2021_end2end_qwen3-6-35b-moe",
]


def test_catalogue_lit_les_quatre_axes(analyse) -> None:
    analyse._figer_pour_test(EXPORTES)
    assert analyse.ANNEES == [2015, 2021]
    assert analyse.PROMPTS == ["base", "exemples"]
    assert analyse.MODELES == ["gemma4-26b-moe", "qwen3-6-35b-moe"]
    assert analyse.CATALOGUE[(2015, "exemples", "end-to-end", "qwen3-6-35b-moe")] == (
        "dictee2015_end2end_exemples_qwen3-6-35b-moe"
    )
    # Two-stage à deux modèles : rangé sous le modèle de l'étape 1.
    assert analyse.CATALOGUE[(2021, "base", "two-stage", "qwen3-6-35b-moe")] == (
        "dictee2021_two_stage_base_qwen3-6-35b-moe_gemma4-26b-moe"
    )


def test_noms_non_conformes_ignores(analyse) -> None:
    analyse._figer_pour_test(EXPORTES)
    noms = set(analyse.CATALOGUE.values())
    assert "dictee_end2end_cot_qwen3-8-27b" not in noms
    assert "qwen3-8-27b" not in analyse.MODELES
    analyse.note_chargement()
    assert any("dictee_end2end_cot_qwen3-8-27b" in note for note in analyse.NOTES)


def test_vues_annees_et_prompts(analyse) -> None:
    analyse._figer_pour_test(EXPORTES)
    cles = [v.cle for v in analyse.vues_possibles()]
    assert "annees|qwen3-6-35b-moe|base" in cles
    assert "annees|qwen3-6-35b-moe|exemples" in cles
    assert "annees|gemma4-26b-moe|base" in cles
    # Deux prompts exportés pour qwen3-6 en 2015 seulement.
    assert "prompts|qwen3-6-35b-moe|2015|base|exemples" in cles
    assert not any(c.startswith("prompts|qwen3-6-35b-moe|2021") for c in cles)
    assert not any(c.startswith("prompts|gemma4-26b-moe") for c in cles)


def test_vue_annees_quatre_series(analyse) -> None:
    analyse._figer_pour_test(EXPORTES)
    vue = analyse.Vue("annees", "qwen3-6-35b-moe", ((2015, "base"), (2021, "base")))
    assert vue.titre == "qwen3-6-35b-moe, prompt base, 2015 vs 2021"
    series = [vue.serie(a, c) for a in analyse.APPROCHES for c in vue.configs]
    assert series == [
        "end-to-end · 2015",
        "end-to-end · 2021",
        "two-stage · 2015",
        "two-stage · 2021",
    ]


def test_series_absentes(analyse) -> None:
    analyse._figer_pour_test(EXPORTES)
    vue = analyse.Vue("prompts", "qwen3-6-35b-moe", ((2015, "base"), (2015, "exemples")))
    run = analyse.Run(
        label="x",
        nom="x",
        approche="end-to-end",
        modele="m",
        modele_cle="m",
        couleur="#000",
        df=None,
        copies=None,
        items=None,
        serie="end-to-end · base",
    )
    assert analyse.series_absentes(vue, {"x": run}) == [
        "end-to-end · exemples",
        "two-stage · base",
        "two-stage · exemples",
    ]


def test_configuration_de_reference(analyse) -> None:
    """Dernière année où un modèle a les deux approches au prompt de base."""
    analyse._figer_pour_test(EXPORTES)
    assert analyse.CONFIG_REFERENCE == (2021, "base")
    analyse._figer_pour_test([n for n in EXPORTES if "2021_two_stage" not in n])
    assert analyse.CONFIG_REFERENCE == (2015, "base")


def test_configs_comparables(analyse) -> None:
    """La zone « deux modèles » ne propose que les configurations à deux modèles."""
    analyse._figer_pour_test(EXPORTES)
    assert analyse.configs_comparables() == [(2015, "base")]


def test_couleurs_par_approche_et_rang(analyse) -> None:
    assert analyse.couleur_run("end-to-end", 0) != analyse.couleur_run("end-to-end", 1)
    assert analyse.couleur_run("end-to-end", 0) != analyse.couleur_run("two-stage", 0)
