"""Tests de la convention de nommage des runs (année, approche, prompt, modèle)."""

from pathlib import Path

import pytest

from evaluation_dictee.config import load_config, run_output_name
from evaluation_dictee.run_names import NomRun, analyser_nom_run, nom_base


def test_nom_simple() -> None:
    assert analyser_nom_run("dictee2015_end2end_base_qwen3-6-35b-moe") == NomRun(
        annee=2015, approche="end2end", prompt="base", modeles=("qwen3-6-35b-moe",)
    )


def test_two_stage_deux_modeles() -> None:
    nom = analyser_nom_run("dictee2021_two_stage_comptage-exemples_qwen3-6-35b-moe_gemma4-26b-moe")
    assert nom is not None
    assert nom.approche == "two_stage"
    assert nom.prompt == "comptage-exemples"
    assert nom.modele == "qwen3-6-35b-moe"
    assert nom.modeles == ("qwen3-6-35b-moe", "gemma4-26b-moe")
    assert nom.base == "dictee2021_two_stage_comptage-exemples"


def test_modele_avec_point_conserve() -> None:
    nom = analyser_nom_run("dictee2015_end2end_base_qwen3.6-35b-moe")
    assert nom is not None and nom.modele == "qwen3.6-35b-moe"


@pytest.mark.parametrize(
    "nom",
    [
        # Anciens noms, sans année ni prompt : ignorés, jamais mal lus.
        "dictee_end2end_qwen3-6-35b-moe",
        "dictee_two_stage_gemma4-26b-moe",
        # Année sans prompt : le modèle serait lu comme un prompt, sans modèle.
        "dictee2021_end2end_qwen3-6-35b-moe",
        # Bras d'expérience à l'ancienne : prompt avec tiret bas.
        "dictee_end2end_comptage_exemples_qwen3-8-27b",
        "dictee2015_end2end_alea_",
        "dictee2015_htr_base_qwen3-6-35b-moe",
    ],
)
def test_noms_non_conformes_ignores(nom: str) -> None:
    assert analyser_nom_run(nom) is None


def test_nom_base() -> None:
    assert nom_base(2015, "end2end", "comptage-exemples") == "dictee2015_end2end_comptage-exemples"
    with pytest.raises(ValueError):
        nom_base(2015, "end2end", "comptage_exemples")
    with pytest.raises(ValueError):
        nom_base(2015, "end_to_end", "base")


@pytest.mark.parametrize(
    "chemin",
    [
        "configs/scoring/dictee_end2end.yaml",
        "configs/scoring/dictee_two_stage.yaml",
        "configs/scoring/dictee2021_end2end.yaml",
        "configs/scoring/dictee2021_two_stage.yaml",
    ],
)
def test_configs_corpus_complet_suivent_la_convention(chemin: str) -> None:
    """Les runs sur corpus complet alimentent le site : leur nom doit s'y décoder."""
    config = load_config(Path(chemin))
    nom = analyser_nom_run(run_output_name(config))
    assert nom is not None
    assert nom.prompt == "base"
    approche = {"end_to_end": "end2end", "two_stage": "two_stage"}[config.approach]
    assert nom.approche == approche
