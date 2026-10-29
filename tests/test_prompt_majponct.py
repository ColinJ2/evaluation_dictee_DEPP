"""Tests des consignes majuscules et ponctuation tolérante (prompt `majponct`)."""

from evaluation_dictee.config import PromptConfig, load_config
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.pipeline.prompts import (
    build_dictation_prompt,
    build_text_coding_prompt,
    build_transcription_prompt,
)
from evaluation_dictee.run_names import analyser_nom_run

OPTIONS = {"case_rules", "lenient_punctuation", "doubt_word_is_error"}

ITEMS = [
    GridItem("i1", "Le", "mot", [], [], []),
    GridItem("i2", "soir", "mot", [], [], []),
    GridItem("i3", ".", "ponctuation", [], [], []),
]


def _texte(messages: list) -> str:
    return "\n\n".join(str(m["content"]) for m in messages)


def _systeme(messages: list) -> str:
    return next(str(m["content"]) for m in messages if m["role"] == "system")


def test_absentes_par_defaut() -> None:
    for texte in (
        _texte(build_dictation_prompt("Le soir.", ITEMS, PromptConfig())),
        _texte(build_transcription_prompt()),
        _texte(build_text_coding_prompt("Le soir.", "le soir", ITEMS)),
    ):
        assert "MAJUSCULES" not in texte
        assert "BÉNÉFICE DU DOUTE" not in texte
        assert "transcris TOUT signe" not in texte
        assert "EN CAS DE DOUTE" not in texte


def test_end_to_end_ajoute_les_deux_consignes() -> None:
    cfg = PromptConfig(case_rules=True, lenient_punctuation=True, doubt_word_is_error=True)
    texte = _systeme(build_dictation_prompt("Le soir.", ITEMS, cfg))
    assert "16 - MOTS, EN CAS DE DOUTE" in texte
    assert "14 - MAJUSCULES" in texte
    assert "« À » pour « A » est correct" in texte
    assert "15 - PONCTUATION, AU BÉNÉFICE DU DOUTE" in texte
    assert "code 9 (erreur) UNIQUEMENT si un signe est écrit et que tu es SÛR" in texte
    assert "code 0 (absent) UNIQUEMENT si tu es SÛR" in texte


def test_ponctuation_tolerante_remplace_la_consigne_e3() -> None:
    """Les deux règles se contredisent sur le code 9 : la tolérante l'emporte."""
    cfg = PromptConfig(check_punctuation_presence=True, lenient_punctuation=True)
    texte = _texte(build_dictation_prompt("Le soir.", ITEMS, cfg))
    assert "CAS PARTICULIER DE LA PONCTUATION" not in texte
    assert "BÉNÉFICE DU DOUTE" in texte


def test_two_stage_etape1_conserve_casse_et_signes() -> None:
    texte = _systeme(
        build_transcription_prompt(
            case_rules=True, lenient_punctuation=True, doubt_word_is_error=True
        )
    )
    assert "transcris celle qui s'écarte de l'orthographe correcte" in texte
    assert "Respecte EXACTEMENT les majuscules" in texte
    assert "transcris TOUT signe de ponctuation visible" in texte
    # La consigne des ratures reste en place.
    assert "version FINALE" in texte


def test_two_stage_etape2_ajoute_les_regles_au_systeme() -> None:
    messages = build_text_coding_prompt(
        "Le soir.",
        "le soir",
        ITEMS,
        case_rules=True,
        lenient_punctuation=True,
        doubt_word_is_error=True,
    )
    systeme = _systeme(messages)
    assert "5 - MAJUSCULES" in systeme
    assert "6 - PONCTUATION" in systeme
    assert "si un AUTRE signe y figure à sa place, code 9" in systeme
    assert "7 - MOTS, EN CAS DE DOUTE" in systeme
    # Rien n'est ajouté au message user (transcription et items).
    user = next(str(m["content"]) for m in messages if m["role"] == "user")
    assert "MAJUSCULES" not in user


def test_configs_majponct() -> None:
    for app in ("end2end", "two_stage"):
        cfg = load_config(f"configs/scoring/dictee2021_{app}_majponct.yaml")
        assert cfg.prompt.case_rules and cfg.prompt.lenient_punctuation
        assert cfg.prompt.doubt_word_is_error
        nom = analyser_nom_run(f"{cfg.name}_qwen3-6-35b-moe")
        assert nom is not None and nom.prompt == "majponct" and nom.annee == 2021
        base = load_config(f"configs/scoring/dictee2021_{app}.yaml")
        # Seules les deux options et le nom diffèrent du prompt `base`.
        assert cfg.model_dump(exclude={"name", "prompt"}) == base.model_dump(
            exclude={"name", "prompt"}
        )
        assert cfg.prompt.model_dump(exclude=OPTIONS) == (base.prompt.model_dump(exclude=OPTIONS))


def test_petits_mots_absents_par_defaut_et_ajoutes_sur_option() -> None:
    assert "PETITS MOTS" not in _texte(build_dictation_prompt("Le soir.", ITEMS, PromptConfig()))
    cfg = PromptConfig(case_rules=True, doubt_word_is_error=True, small_words_rules=True)
    texte = _systeme(build_dictation_prompt("Le soir.", ITEMS, cfg))
    assert "17 - PETITS MOTS" in texte
    assert texte.index("16 - MOTS, EN CAS DE DOUTE") < texte.index("17 - PETITS MOTS")
    assert "DOUBLE ERREUR" in texte
    assert "un même trait ne compte qu'UNE fois" in texte


def test_petits_mots_two_stage() -> None:
    etape1 = _systeme(build_transcription_prompt(small_words_rules=True))
    assert "8 - PETITS MOTS ET LETTRES PROCHES" in etape1
    etape2 = _systeme(
        build_text_coding_prompt("Le soir.", "le soir", ITEMS, small_words_rules=True)
    )
    assert "8 - PETITS MOTS" in etape2 and "DOUBLE ERREUR" in etape2


def test_configs_petitsmots() -> None:
    for app in ("end2end", "two_stage"):
        cfg = load_config(f"configs/scoring/dictee2021_{app}_petitsmots.yaml")
        nom = analyser_nom_run(f"{cfg.name}_qwen3-6-35b-moe")
        assert nom is not None and nom.prompt == "petitsmots" and nom.annee == 2021
        majponct = load_config(f"configs/scoring/dictee2021_{app}_majponct.yaml")
        # Seule l'option petits mots (et le nom) diffère de `majponct`.
        assert cfg.prompt.small_words_rules and not majponct.prompt.small_words_rules
        assert cfg.model_dump(exclude={"name", "prompt"}) == majponct.model_dump(
            exclude={"name", "prompt"}
        )
        assert cfg.prompt.model_dump(exclude={"small_words_rules"}) == (
            majponct.prompt.model_dump(exclude={"small_words_rules"})
        )
