"""Convention de nommage des runs de scoring : année, approche, prompt, modèle(s).

Le nom d'un run (champ `name` du YAML, puis préfixe des fichiers de sortie posé par
`config.run_output_name`) porte TOUT ce qui l'identifie, pour que le site et les
scripts le retrouvent sans table de correspondance ni déduction fragile :

    dictee<AAAA>_<end2end|two_stage>_<prompt>_<modele>[_<modele_etape2>]

    dictee2015_end2end_base_qwen3-6-35b-moe
    dictee2021_two_stage_base_qwen3-6-35b-moe_gemma4-26b-moe

Le prompt est un identifiant court SANS tiret bas (`base`, `exemples`,
`comptage-exemples`…) : c'est ce qui le sépare sans ambiguïté du modèle, dont le
nom ne contient pas non plus de tiret bas (cf. `_slugify_model_name`). Un nom qui
ne suit pas la convention (anciens runs `dictee_end2end_<modele>`, bras
d'expérience) est simplement ignoré par `analyser_nom_run`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Approches reconnues, telles qu'écrites dans les noms de run.
APPROCHES_NOM: tuple[str, ...] = ("end2end", "two_stage")

#: Prompt de référence : celui des configs `dictee<AAAA>_<approche>_base`.
PROMPT_BASE = "base"

_MOTIF = re.compile(
    r"^dictee(?P<annee>\d{4})_(?P<approche>end2end|two_stage)_"
    r"(?P<prompt>[a-z0-9][a-z0-9-]*)_(?P<modeles>[^_].*)$"
)


@dataclass(frozen=True)
class NomRun:
    """Composantes d'un nom de run conforme à la convention."""

    annee: int
    approche: str
    prompt: str
    #: Modèle de l'étape 1, puis éventuellement celui de l'étape 2 (two_stage).
    modeles: tuple[str, ...]

    @property
    def modele(self) -> str:
        """Modèle de l'étape 1 : celui qui identifie le run dans les comparaisons."""
        return self.modeles[0]

    @property
    def base(self) -> str:
        """Nom du run sans le suffixe de modèle (= champ `name` du YAML)."""
        return nom_base(self.annee, self.approche, self.prompt)


def nom_base(annee: int, approche: str, prompt: str) -> str:
    """Construit le champ `name` d'un run conforme à la convention.

    Args:
        annee: année de la dictée (ex. 2015).
        approche: `end2end` ou `two_stage`.
        prompt: identifiant du prompt, sans tiret bas.

    Returns:
        Le nom `dictee<annee>_<approche>_<prompt>`.

    Raises:
        ValueError: si l'approche est inconnue ou si le prompt contient un tiret bas.
    """
    if approche not in APPROCHES_NOM:
        raise ValueError(f"Approche inconnue : {approche!r} (attendu : {APPROCHES_NOM})")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", prompt):
        raise ValueError(
            f"Prompt invalide : {prompt!r} (minuscules, chiffres et tirets, sans tiret bas)"
        )
    return f"dictee{annee}_{approche}_{prompt}"


def verifier_prompt(nom: str, prompt: str) -> NomRun:
    """Vérifie qu'un run suit la convention et porte le prompt déclaré.

    Sert à l'export vers le site : celui qui exporte nomme le prompt, et ce nom
    doit figurer dans le nom du run, faute de quoi le site ne saurait pas le classer.

    Args:
        nom: nom du run (suffixe de fichier ôté).
        prompt: prompt déclaré par l'utilisateur.

    Returns:
        Les composantes du nom.

    Raises:
        ValueError: si le nom ne suit pas la convention ou porte un autre prompt.
    """
    decompose = analyser_nom_run(nom)
    if decompose is None:
        raise ValueError(
            f"Le run `{nom}` ne suit pas la convention "
            "`dictee<AAAA>_<end2end|two_stage>_<prompt>_<modele>` : le site ne pourrait "
            f"pas le classer. Renommer le champ `name` du YAML, par exemple en "
            f"`dictee2015_end2end_{prompt}`, puis relancer le run (ou renommer ses fichiers)."
        )
    if decompose.prompt != prompt:
        raise ValueError(
            f"Prompt déclaré `{prompt}`, mais le run `{nom}` porte le prompt "
            f"`{decompose.prompt}`. Corriger --prompt, ou le champ `name` du YAML."
        )
    return decompose


def analyser_nom_run(nom: str) -> NomRun | None:
    """Décompose un nom de run (suffixe de fichier ôté) selon la convention.

    Args:
        nom: nom du run, ex. `dictee2015_end2end_base_qwen3-6-35b-moe`.

    Returns:
        Les composantes du nom, ou None s'il ne suit pas la convention.
    """
    trouve = _MOTIF.match(nom)
    if trouve is None:
        return None
    modeles = tuple(trouve["modeles"].split("_"))
    if any(not m for m in modeles):
        return None
    return NomRun(
        annee=int(trouve["annee"]),
        approche=trouve["approche"],
        prompt=trouve["prompt"],
        modeles=modeles,
    )
