"""Tests du signal de doute « lecture_alternative » (option report_doubt, prompt `doute`)."""

import json

from evaluation_dictee.config import ModelConfig, PromptConfig, load_config
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.data.loaders import Copy
from evaluation_dictee.models.vlm import VLMScorer, _items_json_schema
from evaluation_dictee.pipeline.alignment import best_realignment
from evaluation_dictee.pipeline.prompts import build_dictation_prompt
from evaluation_dictee.run_names import analyser_nom_run

ITEMS = [
    GridItem("i1", "Le", "mot", [], [], []),
    GridItem("i2", "soir", "mot", [], [], []),
]


def _texte(messages: list) -> str:
    return "\n\n".join(str(m["content"]) for m in messages)


def _proprietes(**options: bool) -> list[str]:
    schema = _items_json_schema(chain_of_thought=options.pop("cot", False), **options)
    return list(schema["properties"]["items"]["items"]["properties"])


def test_absent_par_defaut() -> None:
    assert "lecture_alternative" not in _texte(
        build_dictation_prompt("Le soir", ITEMS, PromptConfig())
    )
    assert "lecture_alternative" not in _proprietes()


def test_prompt_et_format_de_sortie() -> None:
    texte = _texte(build_dictation_prompt("Le soir", ITEMS, PromptConfig(report_doubt=True)))
    assert "18 - HÉSITATION RÉELLE ENTRE DEUX LECTURES" in texte
    # Garde-fou contre la surinterprétation, et bénéfice du doute sur s/r, n/m.
    assert "dans la très grande majorité des cas" in texte
    assert "« s »/« r », « n »/« m »" in texte and "code 1 : le doute" in texte
    assert '"lecture_alternative": ""' in texte


def test_schema_declare_l_alternative_avant_le_code() -> None:
    cles = _proprietes(report_doubt=True)
    assert cles.index("transcription") < cles.index("lecture_alternative") < cles.index("code")
    cles_cot = _proprietes(report_doubt=True, cot=True)
    assert cles_cot.index("lecture_alternative") < cles_cot.index("comparaison")


def test_alternative_survit_au_realignement() -> None:
    """Comme la comparaison, l'alternative suit son token quand la copie est décalée."""
    aligned = best_realignment(
        expected_words=["Le", "soir"],
        codes=["1", "1", "1"],
        transcriptions=["Le", "et", "soir"],
        confidences=[None, None, None],
        alternatives=["", "", "soin"],
    )
    assert [a.transcription for a in aligned] == ["Le", "soir"]
    assert [a.lecture_alternative for a in aligned] == ["", "soin"]


def test_alternative_lue_dans_la_reponse() -> None:
    scorer = VLMScorer(
        ModelConfig(name="x"),
        PromptConfig(report_doubt=True),
        "http://x",
        "k",
        ITEMS,
        scheme="simplifiee",
    )
    reponse = {
        "items": [
            {"item_id": "i1", "transcription": "Le", "lecture_alternative": "", "code": "1"},
            {"item_id": "i2", "transcription": "soir", "lecture_alternative": "sois", "code": "1"},
        ]
    }
    pred = scorer._parse_response(
        Copy("c.png", "c.png", ["1", "1"], ["i1", "i2"]), json.dumps(reponse)
    )
    assert [it.lecture_alternative for it in pred.items] == ["", "sois"]


def test_config_doute() -> None:
    cfg = load_config("configs/scoring/dictee2021_end2end_doute.yaml")
    nom = analyser_nom_run(f"{cfg.name}_qwen3-6-35b-moe")
    assert nom is not None and nom.prompt == "doute" and nom.annee == 2021
    petitsmots = load_config("configs/scoring/dictee2021_end2end_petitsmots.yaml")
    assert cfg.prompt.report_doubt and not petitsmots.prompt.report_doubt
    assert cfg.model_dump(exclude={"name", "prompt"}) == petitsmots.model_dump(
        exclude={"name", "prompt"}
    )
    assert cfg.prompt.model_dump(exclude={"report_doubt"}) == (
        petitsmots.prompt.model_dump(exclude={"report_doubt"})
    )
