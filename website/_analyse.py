"""Chargement et calculs partagés par les pages « Résultats » et « Écarts ».

Le site **relit les prédictions déjà exportées** (S3 par défaut, ou un dossier
local via `S3_PREDICTIONS_PREFIX`) et recalcule chiffres et figures au rendu :
aucun pipeline n'est réexécuté, aucune figure n'est figée. Toute défaillance
(paquet non importable, run non exporté, S3 injoignable) est capturée dans
`NOTES` et laisse la page rendable, avec des « — » à la place des chiffres.

Ce module est préfixé par `_` : Quarto ne le rend pas comme une page, il est
seulement importé par les `.qmd` (dont le répertoire de travail est `website/`).

Le contenu reprend l'analyse de `notebooks/03_analyse_resultats.ipynb`. Les
briques génériques (Wilson, IC du kappa, design effect, métriques par item et
par copie, comparaison multi-modèles) viennent du paquet `evaluation_dictee` ;
seule la mise en forme propre au site vit ici.
"""

from __future__ import annotations

import html
import math
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

# ── Import du paquet du projet ────────────────────────────────────────────────
# Le site peut être rendu hors `uv run` (Quarto lancé avec le Python système) :
# on ajoute alors `src/` au chemin d'import en remontant depuis le dossier courant.
REPO_ROOT: Path | None = None
for _parent in (Path.cwd(), *Path.cwd().parents):
    if (_parent / "src" / "evaluation_dictee").is_dir():
        REPO_ROOT = _parent
        sys.path.insert(0, str(_parent / "src"))
        break

#: Incidents rencontrés au chargement, affichés en fin de page par `bloc_notes()`.
NOTES: list[str] = []

try:
    from evaluation_dictee.data.grid import GridItem, load_grid
    from evaluation_dictee.evaluation.multi_model import (
        agreement_per_item,
        confidence_score,
        consensus_levels,
        disagreement_type_summary,
        fleiss_kappa,
        pairwise_agreement,
        referral_curve_with_ci,
    )
    from evaluation_dictee.evaluation.report import (
        copies_by_disagreement,
        disagreement_decomposition,
        filtrer_evaluables,
        load_predictions,
        per_copy_metrics,
        per_item_metrics,
    )
    from evaluation_dictee.evaluation.statistics import (
        design_effect,
        kappa_interval_clustered,
        wilson_interval,
    )
    from evaluation_dictee.run_names import PROMPT_BASE, analyser_nom_run

    PAQUET_OK = True
except ImportError as exc:  # noqa: BLE001 — la page doit rester rendable
    PAQUET_OK = False
    PROMPT_BASE = "base"

    def analyser_nom_run(nom: str) -> None:  # type: ignore[misc]
        """Sans le paquet, aucun run n'est reconnu."""
        return None

    NOTES.append(
        f"Paquet `evaluation_dictee` non importable ({exc}) : aucun chiffre n'a pu "
        "être calculé. Rendre le site depuis l'environnement du projet "
        "(`uv sync --extra website` puis `uv run quarto render website`)."
    )

# ── Paramètres du site ────────────────────────────────────────────────────────
#: Racine des prédictions exportées. Pointer sur un dossier local pour un rendu
#: hors ligne : `export S3_PREDICTIONS_PREFIX=/chemin/vers/predictions`. Le site ne
#: lit que les runs nommés selon la convention de `evaluation_dictee/run_names.py`
#: (`dictee<AAAA>_<approche>_<prompt>_<modele>`), déposés dans `prediction_depp/`.
PREFIX = os.environ.get(
    "S3_PREDICTIONS_PREFIX", "s3://projet-production-ecrits-depp/prediction_depp"
)


def _liste_env(variable: str, defaut: list[str]) -> list[str]:
    """Lit une liste séparée par des virgules dans l'environnement."""
    brut = os.environ.get(variable, "")
    valeurs = [v.strip() for v in brut.split(",") if v.strip()]
    return valeurs or defaut


#: Approches comparées, dans l'ordre d'affichage : libellé → segment du nom de run.
#: Ce sont les deux MÉTHODES du projet (.claude/rules/methodologie.md) — end-to-end une
#: passe vs two-stage HTR+codage. Les variantes de prompt sont un AUTRE axe, lu dans
#: le segment <prompt> du nom (décision D12) : plus rien n'est déduit d'un préfixe,
#: ce qui supprime les confusions de la décision D9 (`dictee_end2end_cot_<modele>`
#: lu comme un faux modèle).
APPROCHES: dict[str, str] = {
    "end-to-end": "end2end",
    "two-stage": "two_stage",
}

#: Modèles retenus si la découverte automatique ne trouve rien (S3 injoignable,
#: rendu hors ligne sans prédictions) : la page reste rendable et nomme ce qu'elle
#: attend, au lieu d'afficher un sélecteur vide.
MODELES_DEFAUT: list[str] = ["qwen3-6-35b-moe"]

#: Approche de référence quand une analyse exige un run unique (classement des
#: pires copies, tri des items). Par défaut l'approche privilégiée du projet.
APPROCHE_REFERENCE = os.environ.get("RESULTATS_APPROCHE_REFERENCE", "end-to-end")

#: Prompt de référence des pages sans axe « prompt » (Écarts, Évaluation).
PROMPT_REFERENCE = os.environ.get("RESULTATS_PROMPT_REFERENCE", PROMPT_BASE)

#: Une configuration de corpus et de consigne : (année de la dictée, prompt).
Config = tuple[int, str]


def libelle_run(approche: str, modele: str) -> str:
    """Libellé d'un run croisant une approche et un modèle (clé des dictionnaires)."""
    return f"{approche} · {modele}"


def libelle_config(config: Config) -> str:
    """Libellé lisible d'une configuration, ex. `2015 · base`."""
    return f"{config[0]} · {config[1]}"


#: Chemin de la grille de codage (mots attendus, ordre de la dictée).
GRID_PATH = os.environ.get("RESULTATS_GRID_PATH", "configs/grille_dictee_2015.json")

#: Codes attendus dans la grille simplifiée. `y_true` contient une minorité de
#: codes hérités de la saisie AGATE (vides, « i », « v ») : on les compte comme
#: des erreurs (convention « erreur = code != 1 » du projet) mais on les signale.
CODES_VALIDES = frozenset({"1", "9", "0"})

#: Libellé lisible de chaque code.
LIBELLE_CODE = {"1": "correct", "9": "erreur", "0": "absent", "?": "non lu"}

# ── Copies vierges, écartées des classements d'écart ──────────────────────────
# Le critère est celui du pipeline, et lui seul. `pipeline/benchmark.py` mesure la
# densité d'encre de l'image (`data/loaders.ink_ratio`) et, sous
# `config.data.blank_ink_threshold` (2,5 % par défaut — seul le pré-imprimé
# marque), déclare la copie VIERGE : elle est codée « 0 » sur tous les items sans
# appel modèle, et chaque ligne du JSONL est estampillée `blank: true`. Le site se
# contente de relire ce marqueur : aucune heuristique concurrente ici, pour que
# site et pipeline ne puissent pas diverger dans leur définition.
#
# Un run produit AVANT ce garde-fou n'a pas la colonne : le site ne peut alors
# écarter aucune copie et le dit explicitement, plutôt que de deviner.
COLONNE_VIERGE = "blank"

#: Seuil d'encre du pipeline, repris pour l'affichage quand la source de données
#: ne le porte pas. La valeur qui s'applique au run est celle de son YAML
#: (`data.blank_ink_threshold`).
SEUIL_ENCRE_PIPELINE = 0.025

#: Colonne portant la densité d'encre mesurée, quand elle est disponible.
COLONNE_ENCRE = "ink_ratio"

# ── Palette ───────────────────────────────────────────────────────────────────
C_EXPERT = "#1f4e79"
C_MODELE = "#c44536"
C_OK, C_MOYEN, C_PB = "#2e7d32", "#ef6c00", "#c62828"
#: Une famille de couleurs par approche, une nuance par modèle. La couleur d'un
#: run ne dépend donc PAS de l'ordre de chargement : `end-to-end · gemma` garde
#: la même teinte dans la vue d'un modèle et dans une comparaison de paires, et
#: l'approche reste lisible d'un coup d'œil (rouge = end-to-end, bleu =
#: two-stage), le modèle n'étant que la nuance.
NUANCES_APPROCHE: dict[str, list[str]] = {
    "end-to-end": ["#c44536", "#e0897c", "#8c2f24", "#f0b8b0"],
    "two-stage": ["#0b7285", "#57a5b3", "#06424d", "#a5d2da"],
}

#: Palette de repli quand une approche n'est pas dans `NUANCES_APPROCHE`.
COULEURS_RUN = ["#c44536", "#0b7285", "#7b2d8b", "#946200"]
Z95 = 1.96


def couleur_run(approche: str, rang: int) -> str:
    """Couleur d'un run : teinte de l'approche, nuance selon son rang.

    Le rang est celui du modèle dans une comparaison de modèles, ou celui de la
    configuration (année, prompt) dans la vue d'un modèle : la teinte dit toujours
    l'approche, la nuance ce qui varie à l'intérieur de la vue.

    Args:
        approche: libellé d'approche (clé de `APPROCHES`).
        rang: rang du run dans ce qui varie à approche fixée.

    Returns:
        La couleur hexadécimale du run.
    """
    nuances = NUANCES_APPROCHE.get(approche, COULEURS_RUN)
    return nuances[rang % len(nuances)]


def init_matplotlib() -> None:
    """Applique le style de figures du projet (à appeler une fois par page)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9.5,
        }
    )


def grille_axes(n: int, largeur: float = 7.5, hauteur: float = 7.0, ncols: int = 2):
    """Grille de sous-graphiques à `n` panneaux, sur au plus `ncols` colonnes.

    Les figures « un panneau par run » deviennent illisibles alignées sur une
    seule ligne dès qu'on croise les approches et les modèles : au-delà de deux
    panneaux, on passe à la ligne plutôt que d'écraser chaque panneau.

    Args:
        n: nombre de panneaux utiles.
        largeur: largeur d'un panneau, en pouces.
        hauteur: hauteur d'un panneau, en pouces.
        ncols: nombre maximal de colonnes.

    Returns:
        Le couple (figure, liste des `n` axes), les axes en trop étant masqués.
    """
    import matplotlib.pyplot as plt

    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(largeur * ncols, hauteur * nrows), squeeze=False
    )
    plats = [ax for ligne in axes for ax in ligne]
    for ax in plats[n:]:
        ax.set_visible(False)
    return fig, plats[:n]


# ── Formatage ─────────────────────────────────────────────────────────────────
def _absent(x: object) -> bool:
    """Vrai si la valeur est manquante (None ou NaN) et doit s'afficher « — »."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x: object, d: int = 1) -> str:
    """Formate une proportion de [0, 1] en pourcentage, « — » si absente."""
    return "—" if _absent(x) else f"{float(x):.{d}%}".replace("%", " %")


def pts(x: object, d: int = 1) -> str:
    """Formate un pourcentage déjà exprimé en points (0-100), « — » si absent."""
    return "—" if _absent(x) else f"{float(x):.{d}f} %"


def num(x: object, d: int = 3) -> str:
    """Formate un nombre décimal, « — » si absent."""
    return "—" if _absent(x) else f"{float(x):.{d}f}"


def milliers(n: object) -> str:
    """Formate un entier avec une espace fine comme séparateur de milliers."""
    return "—" if _absent(n) else f"{int(n):,}".replace(",", " ")


def formateur_unite(unite: str):
    """Renvoie le formateur associé à une unité (« pct » pour %, « num » sinon)."""
    return pct if unite == "pct" else num


def ic(valeur: object, bas: object, haut: object, formateur=pct) -> str:
    """Formate « estimation [bas ; haut] », « — » si l'estimation est absente."""
    if _absent(valeur):
        return "—"
    if _absent(bas) or _absent(haut):
        return formateur(valeur)
    return f"{formateur(valeur)} <small>[{formateur(bas)} ; {formateur(haut)}]</small>"


def table_markdown(entetes: list[str], lignes: list[list[str]], legende: str = "") -> str:
    """Assemble un tableau Markdown (aligné à droite sauf la 1re colonne).

    Args:
        entetes: libellés de colonnes.
        lignes: valeurs, déjà formatées en chaînes.
        legende: légende Quarto, ajoutée sous le tableau si non vide.

    Returns:
        Le tableau au format Markdown, prêt à `print()` dans une cellule
        `output: asis`.
    """
    aligns = ["---"] + [":---:"] * (len(entetes) - 1)
    out = ["| " + " | ".join(entetes) + " |", "| " + " | ".join(aligns) + " |"]
    out += ["| " + " | ".join(ligne) + " |" for ligne in lignes]
    if legende:
        out += ["", f": {legende} {{.striped .hover}}"]
    return "\n".join(out)


# Ancre de la définition de chaque métrique sur la page « Évaluation & métriques ».
# Les tableaux de résultats n'affichent donc plus le sens de lecture ni la méthode
# d'intervalle : ils y renvoient, pour rester lisibles.
ANCRES_METRIQUES: dict[str, str] = {
    "Accord brut": "accord-brut",
    "Kappa de Cohen": "kappa",
    "Rappel des erreurs (sensibilité)": "rappel",
    "Précision sur les erreurs": "precision",
    "Taux de sur-correction": "sur-correction",
    "Taux de sur-détection": "sur-detection",
    "ECE (calibration)": "ece",
}


def lien_metrique(metrique: str, page: str = "evaluation.qmd") -> str:
    """Nom de métrique transformé en lien vers sa définition (inchangé si inconnue)."""
    ancre = ANCRES_METRIQUES.get(metrique)
    return f"[{metrique}]({page}#{ancre})" if ancre else metrique


def bloc_notes() -> str:
    """Callout replié listant les incidents de chargement (vide si aucun)."""
    if not NOTES:
        return ""
    lignes = [
        '::: {.callout-warning appearance="simple" collapse="true" '
        'title="Chiffres non calculés ou incomplets"}'
    ]
    lignes += [f"- {n}" for n in NOTES]
    lignes.append(":::")
    return "\n".join(lignes)


# ── Vues commutables : un menu déroulant, une vue par modèle ──────────────────
# La page « Résultats » rend TOUS les modèles au moment du rendu, chacun dans son
# `div`, et un menu déroulant en montre un seul (`website/js/vues-modeles.html`).
# Ce choix garde une seule page, une seule URL et un basculement instantané ; le
# coût est un rendu plus long, proportionnel au nombre de modèles exportés.
#
# Les fences utilisent SIX deux-points : le contenu d'une vue contient lui-même
# des `:::` (callouts, `panel-tabset`), et pandoc ne referme correctement une
# division imbriquée que si la fence extérieure est plus longue.
FENCE_VUE = ":" * 6

#: Dossier des figures des vues, relatif au dossier `website/`. Les figures d'une
#: vue ne peuvent pas être la sortie d'une cellule (il faut les intercaler dans le
#: `div` de chaque modèle) : elles sont donc écrites sur disque et insérées en
#: Markdown. Le dossier est ignoré par Git, comme `_freeze/`.
DOSSIER_FIGURES = Path("figures")


def _slug(texte: str) -> str:
    """Réduit un libellé à un nom de fichier sûr (minuscules, tirets)."""
    garde = [c if c.isalnum() else "-" for c in texte.lower()]
    return "-".join(filter(None, "".join(garde).split("-")))


def enregistrer_figure(fig, nom: str, legende: str = "") -> str:
    """Écrit une figure dans `figures/` et renvoie le Markdown qui l'insère.

    Args:
        fig: figure matplotlib, fermée après écriture.
        nom: identifiant de la figure (section + vue), transformé en nom de fichier.
        legende: légende affichée sous la figure.

    Returns:
        Le Markdown de l'image, à `print()` dans une cellule `output: asis`.
    """
    import matplotlib.pyplot as plt

    DOSSIER_FIGURES.mkdir(exist_ok=True)
    chemin = DOSSIER_FIGURES / f"{_slug(nom)}.png"
    fig.savefig(chemin, bbox_inches="tight")
    plt.close(fig)
    return f"![{legende}]({chemin.as_posix()})"


@contextmanager
def vue_modele(cle: str, actif: str) -> Iterator[None]:
    """Ouvre une vue « un modèle » : tout ce qui est imprimé dedans lui appartient.

    Args:
        cle: clé de la vue (`Vue.cle`), lue par le JavaScript des menus.
        actif: clé de la vue sélectionnée au chargement (visible sans JS).
    """
    classes = ".vue-modele" + (" .vue-active" if cle == actif else "")
    print(f'{FENCE_VUE} {{{classes} data-vue-modele="{html.escape(cle, quote=True)}"}}')
    yield
    print(FENCE_VUE)


def cle_paire(config: Config, paire: tuple[str, str]) -> str:
    """Clé d'une vue « paire de modèles » : configuration puis modèles."""
    return f"{config[0]}|{config[1]}|{paire[0]}|{paire[1]}"


@contextmanager
def vue_paire(cle: str, actif: str | None) -> Iterator[None]:
    """Ouvre la vue d'une paire de modèles comparés, à configuration fixée.

    Args:
        cle: clé de la vue (`cle_paire`).
        actif: clé de la vue sélectionnée au chargement de la page, ou None.
    """
    classes = ".vue-paire" + (" .vue-active" if cle == actif else "")
    print(f'{FENCE_VUE} {{{classes} data-vue-paire="{html.escape(cle, quote=True)}"}}')
    yield
    print(FENCE_VUE)


def _options(valeurs: list[str], selection: str, libelles: dict[str, str] | None = None) -> str:
    """Balises `<option>` d'un menu déroulant."""
    libelles = libelles or {}
    return "".join(
        f'<option value="{html.escape(v, quote=True)}"'
        f"{' selected' if v == selection else ''}>{html.escape(libelles.get(v, v))}</option>"
        for v in valeurs
    )


def _menu(identifiant: str, libelle: str, classe: str, options: str) -> str:
    """Un libellé et son menu déroulant."""
    return (
        f'<label for="{identifiant}">{libelle}</label>'
        f'<select id="{identifiant}" class="{classe}">{options}</select>'
    )


def selecteur_modele(modeles: list[str], selection: str, identifiant: str) -> str:
    """Menu déroulant simple choisissant un modèle (vues indexées par modèle).

    Sert aux pages à configuration fixée (ex. « Publications DEPP », 2015).

    Args:
        modeles: modèles proposés, dans l'ordre d'affichage.
        selection: modèle sélectionné au chargement.
        identifiant: identifiant HTML, unique dans la page.

    Returns:
        Le HTML du menu, à `print()` dans une cellule `output: asis`.
    """
    return (
        f'<div class="selecteur-vue" data-modeles="{html.escape("|".join(modeles), quote=True)}">'
        + _menu(identifiant, "Modèle affiché", "selecteur-modele", _options(modeles, selection))
        + "</div>"
    )


def selecteur_vues(vues: list[Vue], active: Vue) -> str:
    """Menus choisissant la vue « un modèle » : mode, modèle, année et prompt(s).

    Deux modes : comparer les **années** à prompt commun (une série par année et
    par approche), ou comparer **deux prompts** sur une même année. Le JavaScript
    (`js/vues-modeles.html`) compose la clé de la vue à partir des menus.

    Args:
        vues: vues rendues dans la page.
        active: vue sélectionnée au chargement.

    Returns:
        Le HTML des menus, à `print()` dans une cellule `output: asis`.
    """
    modeles = list(dict.fromkeys(v.modele for v in vues))
    annees = [str(a) for a in ANNEES]
    prompts_annees = list(dict.fromkeys(v.configs[0][1] for v in vues if v.mode == "annees"))
    prompts = list(PROMPTS)
    a_deux_prompts = any(v.mode == "prompts" for v in vues)
    configs_a = active.configs[0]
    configs_b = active.configs[-1]
    modes = {
        "annees": "les années, à prompt commun",
        "prompts": "deux prompts, sur une année" + ("" if a_deux_prompts else " (aucun exporté)"),
    }

    def _cache(mode: str) -> str:
        return "" if mode == active.mode else " hidden"

    attribut_prompts = html.escape("|".join(prompts), quote=True)
    return (
        f'<div class="selecteur-vue selecteur-config" data-prompts="{attribut_prompts}">'
        + _menu("selecteur-mode", "Comparer", "sel-mode", _options(list(modes), active.mode, modes))
        + _menu("selecteur-modele", "Modèle", "sel-modele", _options(modeles, active.modele))
        + f'<span class="sel-groupe" data-mode="annees"{_cache("annees")}>'
        + _menu(
            "selecteur-prompt",
            "Prompt",
            "sel-prompt",
            _options(prompts_annees or prompts, configs_a[1]),
        )
        + "</span>"
        + f'<span class="sel-groupe" data-mode="prompts"{_cache("prompts")}>'
        + _menu("selecteur-annee", "Année", "sel-annee", _options(annees, str(configs_a[0])))
        + _menu("selecteur-prompt-a", "Prompt A", "sel-prompt-a", _options(prompts, configs_a[1]))
        + _menu("selecteur-prompt-b", "Prompt B", "sel-prompt-b", _options(prompts, configs_b[1]))
        + "</span>"
        + "</div>"
    )


def selecteur_paire(
    configs: list[Config], modeles: list[str], config: Config, paire: tuple[str, str]
) -> str:
    """Menus choisissant la configuration (année · prompt) et les deux modèles comparés.

    Args:
        configs: configurations où au moins deux modèles sont exportés.
        modeles: modèles proposés, dans l'ordre d'affichage.
        config: configuration sélectionnée au chargement.
        paire: paire sélectionnée au chargement.

    Returns:
        Le HTML des menus, à `print()` dans une cellule `output: asis`.
    """
    valeurs = [f"{a}|{p}" for a, p in configs]
    libelles = {f"{a}|{p}": libelle_config((a, p)) for a, p in configs}
    return (
        f'<div class="selecteur-vue" data-modeles="{html.escape("|".join(modeles), quote=True)}">'
        + _menu(
            "selecteur-paire-config",
            "Année · prompt",
            "selecteur-paire-config",
            _options(valeurs, f"{config[0]}|{config[1]}", libelles),
        )
        + '<label for="selecteur-paire-a">Comparer</label>'
        '<select id="selecteur-paire-a" class="selecteur-paire" data-role="a">'
        f"{_options(modeles, paire[0])}</select>"
        '<label for="selecteur-paire-b">à</label>'
        '<select id="selecteur-paire-b" class="selecteur-paire" data-role="b">'
        f"{_options(modeles, paire[1])}</select>"
        "</div>"
    )


# ── Chargement ────────────────────────────────────────────────────────────────
@dataclass
class Run:
    """Un run de prédictions et ses agrégats, prêts à être tracés."""

    label: str
    nom: str
    approche: str
    modele: str
    #: Modèle DEMANDÉ, c'est-à-dire celui de l'étape 1 et la clé des menus
    #: déroulants. `modele` peut lui ajouter l'étape 2 (`m1 → m2`) : il sert à
    #: l'affichage, jamais au filtrage.
    modele_cle: str
    couleur: str
    df: pd.DataFrame
    copies: pd.DataFrame = field(repr=False)
    items: pd.DataFrame = field(repr=False)
    annee: int = 0
    prompt: str = ""
    #: Libellé court dans une vue, où seul ce qui varie est nommé (ex.
    #: `end-to-end · 2021`). Vide hors d'une vue : on retombe sur `approche`.
    serie: str = ""

    @property
    def n_copies(self) -> int:
        """Nombre de copies évaluées."""
        return int(self.df["copy_id"].nunique())

    @property
    def n_items(self) -> int:
        """Nombre de lignes item × copie."""
        return len(self.df)


#: Suffixes des fichiers exportés (cf. `utils/s3_export.py`). Le suffixe HTR se
#: termine par celui du scoring : ne jamais tester l'un sans écarter l'autre.
SUFFIXE_SCORING = "_predictions.jsonl"
SUFFIXE_HTR = "_htr_predictions.jsonl"


def _chemin(nom_run: str) -> str:
    """URI du JSONL de prédictions d'un run."""
    return PREFIX.rstrip("/") + "/" + nom_run + SUFFIXE_SCORING


#: Nom des runs de scoring exportés à côté de `PREFIX`, listés une seule fois.
_EXPORTES: list[str] | None = None


def _noms_exportes() -> list[str]:
    """Noms des runs de scoring exportés à côté de `PREFIX`, triés.

    On liste le RÉPERTOIRE, jamais un motif `<base>_*` : sur S3, un glob par
    préfixe fait mettre en cache par s3fs une vue *partielle* du répertoire,
    après quoi les autres fichiers deviennent invisibles — y compris pour
    `load_predictions`, qui échouerait alors sur un fichier bien présent.

    Ce listing est volontairement NON récursif : les bras d'expérience (testés sur un
    échantillon) sont exportés dans un sous-dossier `experimentations/`, invisible
    d'ici par construction (décision D9).

    Returns:
        Les noms de runs (suffixe de modèle compris, `_predictions.jsonl` ôté),
        hors runs HTR. Liste vide si le préfixe est injoignable.
    """
    global _EXPORTES
    if _EXPORTES is None:
        import fsspec

        try:
            fs, _, _ = fsspec.get_fs_token_paths(PREFIX)
            entrees = [str(e) for e in fs.ls(PREFIX.rstrip("/"), detail=False)]
        except Exception:  # noqa: BLE001 — la page doit rester rendable
            entrees = []
        _EXPORTES = sorted(
            Path(e).name.removesuffix(SUFFIXE_SCORING)
            for e in entrees
            if e.endswith(SUFFIXE_SCORING) and not e.endswith(SUFFIXE_HTR)
        )
    return _EXPORTES


#: Clé du catalogue : (année, prompt, libellé d'approche, modèle de l'étape 1).
CleRun = tuple[int, str, str, str]


def catalogue() -> dict[CleRun, str]:
    """Runs exportés, indexés par année, prompt, approche et modèle.

    Tout est lu dans le nom du run (convention de `run_names.py`) : un nom qui ne
    la suit pas — ancien run, bras d'expérience — est ignoré, jamais deviné. Un
    two-stage dont l'étape 2 utilise un autre modèle est rangé sous le modèle de
    l'étape 1 ; si deux runs se disputent la même clé, le premier (ordre
    alphabétique) est retenu.

    Returns:
        Le dictionnaire clé → nom du run exporté.
    """
    approche_de = {segment: libelle for libelle, segment in APPROCHES.items()}
    runs: dict[CleRun, str] = {}
    for nom in _noms_exportes():
        decompose = analyser_nom_run(nom)
        if decompose is None or decompose.approche not in approche_de:
            continue
        cle = (decompose.annee, decompose.prompt, approche_de[decompose.approche], decompose.modele)
        runs.setdefault(cle, nom)
    return runs


def _trier_prompts(prompts: set[str]) -> list[str]:
    """Prompts triés, le prompt de base en tête."""
    return sorted(prompts, key=lambda p: (p != PROMPT_BASE, p))


#: Runs exportés, lus une fois au chargement du module.
CATALOGUE: dict[CleRun, str] = catalogue()

#: Années, prompts et modèles présents dans le catalogue, dans l'ordre d'affichage.
ANNEES: list[int] = sorted({cle[0] for cle in CATALOGUE})
PROMPTS: list[str] = _trier_prompts({cle[1] for cle in CATALOGUE}) or [PROMPT_BASE]

#: Modèles comparés, dans l'ordre d'affichage. Un modèle non exporté est
#: simplement omis, avec une note — on ne substitue JAMAIS un autre modèle, ce
#: qui fausserait la comparaison. `RESULTATS_MODELES` force la liste ; sinon elle
#: vient de S3.
MODELES: list[str] = _liste_env(
    "RESULTATS_MODELES", sorted({cle[3] for cle in CATALOGUE}) or MODELES_DEFAUT
)


def configs_du_modele(modele: str) -> list[Config]:
    """Configurations (année, prompt) où le modèle a au moins un run exporté."""
    return sorted(
        {(a, p) for (a, p, _, m) in CATALOGUE if m == modele},
        key=lambda c: (c[0], PROMPTS.index(c[1]) if c[1] in PROMPTS else len(PROMPTS)),
    )


def approches_du_modele(modele: str, config: Config | None = None) -> list[str]:
    """Approches exportées pour un modèle, sur une configuration donnée.

    Args:
        modele: nom du modèle.
        config: (année, prompt). [défaut : la configuration de référence]

    Returns:
        Les libellés d'approche exportés, dans l'ordre d'`APPROCHES`.
    """
    annee, prompt = config or CONFIG_REFERENCE
    return [a for a in APPROCHES if (annee, prompt, a, modele) in CATALOGUE]


def _config_reference() -> Config:
    """Configuration des pages sans axe « année · prompt » (Écarts, Évaluation).

    L'année la plus récente où un modèle a les deux approches au prompt de
    référence ; à défaut, la plus récente tout court. `RESULTATS_ANNEE_REFERENCE`
    force l'année.
    """
    forcee = os.environ.get("RESULTATS_ANNEE_REFERENCE")
    if forcee:
        return int(forcee), PROMPT_REFERENCE
    completes = [
        a
        for a in ANNEES
        if any(all((a, PROMPT_REFERENCE, ap, m) in CATALOGUE for ap in APPROCHES) for m in MODELES)
    ]
    annees = completes or ANNEES
    return (annees[-1] if annees else 0), PROMPT_REFERENCE


CONFIG_REFERENCE: Config = _config_reference()

#: Modèles évalués sur les DEUX approches à la configuration de référence : ils
#: sont préférés comme sélection par défaut, pour que la page s'ouvre sur une
#: vue non tronquée.
MODELES_COMPLETS: list[str] = [
    modele for modele in MODELES if len(approches_du_modele(modele)) == len(APPROCHES)
]

#: Modèle utilisé par les pages qui n'ont pas d'axe « modèle » (page « Écarts »,
#: dont le détail copie par copie serait illisible multiplié par les modèles), et
#: sélection initiale des menus de la page « Résultats ».
MODELE_REFERENCE = os.environ.get(
    "RESULTATS_MODELE_REFERENCE", (MODELES_COMPLETS or MODELES or [""])[0]
)

#: Run servant de référence quand une analyse en exige un seul.
RUN_REFERENCE = os.environ.get(
    "RESULTATS_RUN_REFERENCE", libelle_run(APPROCHE_REFERENCE, MODELE_REFERENCE)
)


def paires_modeles(modeles: list[str] | None = None) -> list[tuple[str, str]]:
    """Paires de modèles comparables, sans doublon ni ordre inverse.

    Returns:
        Les couples (modèle A, modèle B) avec A avant B dans la liste : une paire
        par comparaison, l'ordre des deux menus déroulants étant normalisé par le
        JavaScript de la page.
    """
    liste = modeles if modeles is not None else MODELES
    return [(a, b) for i, a in enumerate(liste) for b in liste[i + 1 :]]


def _modele_du_run(df: pd.DataFrame, nom: str) -> str:
    """Modèle(s) d'un run, lu dans les prédictions ou, à défaut, dans son nom.

    Les runs récents estampillent chaque ligne d'un `model` (et d'un
    `model_stage2` en two_stage) : c'est la source la plus fiable, et la seule
    qui distingue les deux étapes. Les runs plus anciens ne portent pas ces
    colonnes — on retombe alors sur le suffixe du nom de fichier.

    Args:
        df: prédictions du run.
        nom: nom du run (préfixe des fichiers de sortie).

    Returns:
        Le modèle, sous la forme `<étape 1>` ou `<étape 1> → <étape 2>`.
    """
    if "model" in df.columns and df["model"].notna().any():
        etape1 = str(df["model"].dropna().iloc[0])
        etape2 = ""
        if "model_stage2" in df.columns and df["model_stage2"].notna().any():
            etape2 = str(df["model_stage2"].dropna().iloc[0])
        return etape1 if etape2 in ("", etape1) else f"{etape1} → {etape2}"
    decompose = analyser_nom_run(nom)
    return " → ".join(decompose.modeles) if decompose else "—"


#: Runs déjà lus, indexés comme le catalogue. La page « Résultats » affiche chaque
#: run dans plusieurs vues : sans ce cache, le même JSONL serait relu sur S3
#: autant de fois qu'il apparaît.
_CACHE_RUNS: dict[CleRun, Run | None] = {}

#: Copies écartées des métriques par run (motif -> nombre de copies), pour une note
#: unique en bas de page plutôt qu'une ligne par run.
_ECARTEES: dict[str, dict[str, int]] = {}


def charger_run(cle: CleRun) -> Run | None:
    """Charge un run du catalogue, en mémorisant lectures et échecs.

    Args:
        cle: (année, prompt, approche, modèle).

    Returns:
        Le run chargé (libellé `approche · modèle`, nuance du modèle), ou None
        s'il n'est pas exporté, vide ou illisible — l'incident est consigné.
    """
    if cle in _CACHE_RUNS:
        return _CACHE_RUNS[cle]
    _CACHE_RUNS[cle] = None
    nom = CATALOGUE.get(cle)
    if nom is None or not PAQUET_OK:
        return None
    try:
        df = load_predictions(_chemin(nom))
    except Exception as exc:  # noqa: BLE001 — la page doit rester rendable
        NOTES.append(f"`{nom}` illisible ({type(exc).__name__}) : run omis.")
        return None
    if df.empty:
        NOTES.append(f"`{nom}` est vide : run omis.")
        return None
    # Décision D8 : les copies vierges (auto-codées sans appel modèle) et les items
    # illisibles (code expert « i ») sortent des métriques. Les inclure surestimait
    # le kappa d'environ 0,02 sur l'échantillon de 500 copies.
    df, retirees = filtrer_evaluables(df)
    if retirees:
        _ECARTEES[nom] = retirees
    if df.empty:
        NOTES.append(f"`{nom}` ne contient aucune copie évaluable : run omis.")
        return None
    annee, prompt, approche, modele = cle
    _CACHE_RUNS[cle] = Run(
        label=libelle_run(approche, modele),
        nom=nom,
        approche=approche,
        modele=_modele_du_run(df, nom),
        modele_cle=modele,
        couleur=couleur_run(approche, MODELES.index(modele) if modele in MODELES else 0),
        df=df,
        copies=per_copy_metrics(df),
        items=per_item_metrics(df),
        annee=annee,
        prompt=prompt,
    )
    return _CACHE_RUNS[cle]


def charger_runs(modeles: list[str] | None = None, config: Config | None = None) -> dict[str, Run]:
    """Charge les runs croisant chaque approche et chaque modèle, à configuration fixée.

    Un modèle non exporté est **omis**, jamais remplacé par un autre : la page
    compare les modèles entre eux, une substitution silencieuse y attribuerait
    les chiffres d'un modèle à un autre.

    Args:
        modeles: modèles à charger. [défaut : `MODELES`, tous comparés]
        config: (année, prompt). [défaut : `CONFIG_REFERENCE`]

    Returns:
        Les runs chargés, indexés par libellé `approche · modèle`.
    """
    annee, prompt = config or CONFIG_REFERENCE
    runs: dict[str, Run] = {}
    for approche in APPROCHES:
        for modele in modeles if modeles is not None else MODELES:
            run = charger_run((annee, prompt, approche, modele))
            if run is not None:
                runs[run.label] = run
    return runs


@dataclass(frozen=True)
class Vue:
    """Une vue « un modèle » : un modèle, et les configurations qu'elle confronte.

    Chaque configuration (année, prompt) est déclinée sur les deux approches : une
    vue « années » 2015 vs 2021 affiche donc quatre séries.
    """

    #: `annees` (prompt commun, une configuration par année) ou `prompts` (une
    #: année, deux prompts).
    mode: str
    modele: str
    configs: tuple[Config, ...]

    @property
    def cle(self) -> str:
        """Clé de la vue, recomposée à l'identique par le JavaScript des menus."""
        if self.mode == "annees":
            return f"annees|{self.modele}|{self.configs[0][1]}"
        (annee, prompt_a), (_, prompt_b) = self.configs
        return f"prompts|{self.modele}|{annee}|{prompt_a}|{prompt_b}"

    @property
    def titre(self) -> str:
        """Titre lisible, repris dans les légendes."""
        if self.mode == "annees":
            annees = " vs ".join(str(a) for a, _ in self.configs)
            return f"{self.modele}, prompt {self.configs[0][1]}, {annees}"
        (annee, prompt_a), (_, prompt_b) = self.configs
        return f"{self.modele}, {annee}, prompt {prompt_a} vs {prompt_b}"

    def serie(self, approche: str, config: Config) -> str:
        """Libellé court d'une série : l'approche et ce qui varie dans la vue."""
        varie = str(config[0]) if self.mode == "annees" else config[1]
        return f"{approche} · {varie}"


def vues_possibles(modeles: list[str] | None = None) -> list[Vue]:
    """Vues « un modèle » ayant au moins un run exporté.

    Mode `annees` : pour chaque modèle et chaque prompt, toutes les années du
    catalogue. Mode `prompts` : pour chaque modèle et chaque année, chaque paire de
    prompts exportés (dans l'ordre de `PROMPTS`).

    Args:
        modeles: modèles à décliner. [défaut : `MODELES`]

    Returns:
        Les vues, mode `annees` d'abord.
    """
    vues: list[Vue] = []
    liste = modeles if modeles is not None else MODELES
    for modele in liste:
        configs = configs_du_modele(modele)
        for prompt in PROMPTS:
            if any(p == prompt for _, p in configs):
                vues.append(Vue("annees", modele, tuple((a, prompt) for a in ANNEES)))
    for modele in liste:
        configs = configs_du_modele(modele)
        for annee in ANNEES:
            prompts = [p for p in PROMPTS if (annee, p) in configs]
            for i, prompt_a in enumerate(prompts):
                for prompt_b in prompts[i + 1 :]:
                    vues.append(Vue("prompts", modele, ((annee, prompt_a), (annee, prompt_b))))
    return vues


def runs_de_vue(vue: Vue) -> dict[str, Run]:
    """Runs d'une vue, groupés par approche puis par configuration.

    Chaque run est une copie de celui du cache, renommée pour la vue : un libellé
    unique dans la page (`approche · modèle · année · prompt`), un libellé court
    (`serie`) et une couleur dont la teinte dit l'approche et la nuance la
    configuration.

    Args:
        vue: la vue à remplir.

    Returns:
        Les runs exportés de la vue, indexés par libellé unique.
    """
    runs: dict[str, Run] = {}
    for approche in APPROCHES:
        for rang, config in enumerate(vue.configs):
            run = charger_run((config[0], config[1], approche, vue.modele))
            if run is None:
                continue
            label = f"{run.label} · {libelle_config(config)}"
            runs[label] = replace(
                run,
                label=label,
                serie=vue.serie(approche, config),
                couleur=couleur_run(approche, rang),
            )
    return runs


def series_absentes(vue: Vue, runs: dict[str, Run]) -> list[str]:
    """Séries attendues dans une vue (configuration × approche) mais non exportées."""
    presentes = {r.serie for r in runs.values()}
    return [
        vue.serie(approche, config)
        for approche in APPROCHES
        for config in vue.configs
        if vue.serie(approche, config) not in presentes
    ]


def configs_comparables() -> list[Config]:
    """Configurations (année, prompt) où au moins deux modèles sont exportés."""
    return [
        (a, p)
        for a in ANNEES
        for p in PROMPTS
        if len({m for (aa, pp, _, m) in CATALOGUE if (aa, pp) == (a, p) and m in MODELES}) >= 2
    ]


def note_chargement() -> None:
    """Consigne dans `NOTES` ce que le lecteur doit savoir du chargement."""
    ignores = [n for n in _noms_exportes() if analyser_nom_run(n) is None]
    if ignores:
        NOTES.append(
            "Runs présents sous `" + PREFIX + "` mais ignorés, faute de suivre la convention "
            "`dictee<AAAA>_<approche>_<prompt>_<modele>` : "
            + ", ".join(f"`{n}`" for n in ignores)
            + "."
        )
    if not CATALOGUE:
        NOTES.append(
            "Aucun run n'a pu être chargé. Vérifier `S3_PREDICTIONS_PREFIX` et "
            "l'export (`uv run scripts/export_predictions.py --config … --prompt …`)."
        )
    if _ECARTEES:
        total: dict[str, int] = {}
        for motifs in _ECARTEES.values():
            for motif, n in motifs.items():
                total[motif] = max(total.get(motif, 0), n)
        detail = ", ".join(f"{n} {motif}(s)" for motif, n in sorted(total.items()))
        NOTES.append(
            f"Copies écartées des métriques : jusqu'à {detail} selon le run. Une copie "
            "**vierge** est codée « absent » sans appel modèle, et un item **illisible** "
            "(code expert « i ») n'a reçu aucun jugement auquel comparer le modèle : les "
            "compter lui imputerait un défaut de numérisation ou d'annotation. Les "
            "inclure surestimait le kappa d'environ 0,02. Voir la décision D8."
        )


def restreindre_corpus_commun(runs: dict[str, Run]) -> tuple[dict[str, Run], int]:
    """Restreint tous les runs aux copies qu'ils ont TOUS traitées.

    Comparer deux modèles sur des corpus différents (run inachevé, copies
    abandonnées à la transcription) confond l'effet du modèle avec celui de la
    composition de l'échantillon : les copies difficiles ne se répartissent pas
    au hasard. Cette restriction est la seule façon de lire un écart entre
    modèles comme un écart de qualité.

    Args:
        runs: runs chargés par `charger_runs`.

    Returns:
        Le couple (runs restreints aux copies communes, nombre de ces copies).
        Les runs sont renvoyés tels quels si le corpus est déjà commun.
    """
    if not runs:
        return {}, 0
    commun: set[str] = set.intersection(*(set(r.df["copy_id"].unique()) for r in runs.values()))
    if not commun:
        return {}, 0
    if all(r.n_copies == len(commun) for r in runs.values()):
        return runs, len(commun)
    restreints: dict[str, Run] = {}
    for label, r in runs.items():
        df = r.df[r.df["copy_id"].isin(commun)].reset_index(drop=True)
        restreints[label] = replace(
            r, df=df, copies=per_copy_metrics(df), items=per_item_metrics(df)
        )
    return restreints, len(commun)


def charger_grille() -> tuple[list[GridItem], dict[str, str], dict[str, int]]:
    """Charge la grille de codage : items ordonnés, mot attendu et rang par item.

    Returns:
        Le triplet (items de la grille, item_id → mot attendu, item_id → rang
        dans la dictée). Vide si la grille est introuvable (incident consigné).
    """
    if not PAQUET_OK or REPO_ROOT is None:
        return [], {}, {}
    try:
        grille = load_grid(str(REPO_ROOT / GRID_PATH))
    except Exception as exc:  # noqa: BLE001 — la page doit rester rendable
        NOTES.append(f"Grille `{GRID_PATH}` illisible ({type(exc).__name__}) : items non libellés.")
        return [], {}, {}
    mots = {it.item_id: it.attendu for it in grille.items}
    rangs = {it.item_id: i + 1 for i, it in enumerate(grille.items)}
    return grille.items, mots, rangs


def libelle_item(item_id: str, mots: dict[str, str], rangs: dict[str, int]) -> str:
    """Libellé d'item au format `NN · mot`, qui lève l'ambiguïté des mots répétés."""
    return f"{rangs.get(item_id, 0):02d} · {mots.get(item_id, item_id)}"


def note_codes_non_standard(runs: dict[str, Run]) -> str:
    """Phrase signalant les codes experts hors grille simplifiée (ou chaîne vide)."""
    if not runs:
        return ""
    ref = next(iter(runs.values())).df
    hors = ref[~ref["y_true"].isin(CODES_VALIDES)]
    if hors.empty:
        return ""
    detail = ", ".join(
        f"« {code or 'vide'} » ({milliers(n)})" for code, n in hors["y_true"].value_counts().items()
    )
    return (
        f"{milliers(len(hors))} items sur {milliers(len(ref))} "
        f"({len(hors) / len(ref):.2%}) portent un code expert hors grille simplifiée : "
        f"{detail}. Ils sont conservés et comptés comme des erreurs (convention "
        "« erreur = code ≠ 1 »), ce qui pèse de façon négligeable sur les agrégats."
    )


# ── Synthèse globale ──────────────────────────────────────────────────────────
def _proportion_groupee(indicatrice: pd.Series, grappes: pd.Series) -> tuple[object, float]:
    """Proportion et IC de Wilson corrigés du design effect propre à l'indicatrice.

    Args:
        indicatrice: indicatrice binaire dont on estime la moyenne.
        grappes: identifiant de copie de chaque observation.

    Returns:
        Le couple (intervalle de confiance, design effect appliqué). L'intervalle
        vaut None si l'indicatrice est vide.
    """
    if not len(indicatrice):
        return None, float("nan")
    deff = design_effect(indicatrice, grappes)
    return wilson_interval(int(indicatrice.sum()), len(indicatrice), deff=deff), deff


def synthese_globale(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Accord, kappa, rappel et précision sur l'erreur, avec IC à 95 % par grappes.

    Les 83 items d'une copie ne sont pas indépendants : les IC sont donc corrigés
    du **design effect de Kish**, calculé pour **chaque** métrique sur sa propre
    indicatrice (celle de l'accord et celle du rappel n'ont pas le même). Le kappa
    n'étant la moyenne d'aucune indicatrice, son IC vient d'un bootstrap par
    grappes, qui ne suppose rien sur la structure de corrélation.

    Args:
        df: prédictions à l'item (colonnes copy_id, y_true, y_pred).
        n_boot: nombre de tirages du bootstrap par grappes (kappa).

    Returns:
        Un DataFrame indexé par métrique : valeur, borne basse, borne haute,
        `unite` (« pct » pour une proportion de [0, 1], « num » pour le kappa),
        `deff` (design effect appliqué, NaN pour le kappa qui passe par le
        bootstrap), `methode` et `lecture` (sens de la métrique).
    """
    grappes = df["copy_id"]
    err_exp = df["y_true"] != "1"
    err_mod = df["y_pred"] != "1"

    accord, deff_accord = _proportion_groupee(df["y_true"] == df["y_pred"], grappes)
    # Rappel et précision se mesurent sur des SOUS-ENSEMBLES d'items : la grappe
    # doit être restreinte de la même façon, sinon le design effect est faussé.
    rappel, deff_rappel = _proportion_groupee(err_mod[err_exp], grappes[err_exp])
    precision, deff_precision = _proportion_groupee(err_exp[err_mod], grappes[err_mod])
    # Sur-correction : l'expert voit une erreur, le modèle non (biais VLM connu).
    surcorr, deff_surcorr = _proportion_groupee(~err_mod[err_exp], grappes[err_exp])
    surdet, deff_surdet = _proportion_groupee(err_mod[~err_exp], grappes[~err_exp])
    kappa = kappa_interval_clustered(df["y_true"], df["y_pred"], grappes, n_boot=n_boot)

    methode_wilson = "Wilson corrigé du design effect"
    lignes = [
        (
            "Accord brut",
            accord,
            "pct",
            deff_accord,
            methode_wilson,
            "part des items codés à l'identique : plus haut = mieux",
        ),
        (
            "Kappa de Cohen",
            kappa,
            "num",
            float("nan"),
            f"bootstrap par grappes ({n_boot} tirages)",
            "accord corrigé du hasard : plus haut = mieux",
        ),
        (
            "Rappel des erreurs (sensibilité)",
            rappel,
            "pct",
            deff_rappel,
            methode_wilson,
            "part des erreurs de l'élève retrouvées : plus haut = moins de fautes ratées",
        ),
        (
            "Précision sur les erreurs",
            precision,
            "pct",
            deff_precision,
            methode_wilson,
            "part des erreurs signalées qui en sont vraiment : plus haut = moins de "
            "fausses alertes",
        ),
        (
            "Taux de sur-correction",
            surcorr,
            "pct",
            deff_surcorr,
            methode_wilson,
            "erreurs de l'élève que le modèle valide à tort : plus **bas** = mieux",
        ),
        (
            "Taux de sur-détection",
            surdet,
            "pct",
            deff_surdet,
            methode_wilson,
            "items corrects que le modèle sanctionne à tort : plus **bas** = mieux",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "metrique": nom,
                "valeur": float("nan") if v is None else v.estimate,
                "bas": float("nan") if v is None else v.lower,
                "haut": float("nan") if v is None else v.upper,
                "unite": unite,
                "deff": deff,
                "methode": methode,
                "lecture": lecture,
            }
            for nom, v, unite, deff, methode, lecture in lignes
        ]
    ).set_index("metrique")


# ── Prévalence par item ───────────────────────────────────────────────────────
def prevalence_par_item(
    df: pd.DataFrame, mots: dict[str, str], rangs: dict[str, int]
) -> pd.DataFrame:
    """Prévalence d'erreur par item, expert vs modèle, avec IC de Wilson à 95 %.

    **Aucune correction de clustering ici, et c'est voulu** : pour un item donné il
    y a exactement *une* observation par copie, chacune venant d'un élève différent.
    Il n'y a donc pas de corrélation intra-copie à corriger et le design effect vaut
    1. L'appliquer élargirait les intervalles d'un facteur ≈ 3 sans justification.
    La correction concerne les métriques agrégées sur les 83 items d'une copie,
    dans `synthese_globale`.

    Args:
        df: prédictions à l'item.
        mots: item_id → mot attendu.
        rangs: item_id → rang dans la dictée.

    Returns:
        Un DataFrame indexé par item_id, trié par rang : prévalences d'erreur
        expert et modèle avec leurs IC, et l'écart modèle − expert en points de
        pourcentage.
    """
    lignes = []
    for item_id, grp in df.groupby("item_id"):
        n = len(grp)
        k_exp = int((grp["y_true"] != "1").sum())
        k_mod = int((grp["y_pred"] != "1").sum())
        ci_exp = wilson_interval(k_exp, n)
        ci_mod = wilson_interval(k_mod, n)
        lignes.append(
            {
                "item_id": item_id,
                "position": rangs.get(item_id, 0),
                "mot_attendu": mots.get(item_id, item_id),
                "pct_expert": k_exp / n * 100,
                "ic_expert_bas": ci_exp.lower * 100,
                "ic_expert_haut": ci_exp.upper * 100,
                "pct_modele": k_mod / n * 100,
                "ic_modele_bas": ci_mod.lower * 100,
                "ic_modele_haut": ci_mod.upper * 100,
                "ecart": (k_mod - k_exp) / n * 100,
            }
        )
    return pd.DataFrame(lignes).set_index("item_id").sort_values("position")


# ── Métriques selon la difficulté de la copie ─────────────────────────────────
def metriques_par_difficulte(
    df: pd.DataFrame, copies: pd.DataFrame, col_tri: str, n_quantiles: int = 8
) -> pd.DataFrame:
    """Accord, kappa, rappel et précision par quantile de difficulté de copie.

    Args:
        df: prédictions à l'item.
        copies: métriques par copie (index = copy_id), source de la difficulté.
        col_tri: colonne de `copies` définissant la difficulté (nb d'erreurs).
        n_quantiles: nombre de groupes de difficulté.

    Comme dans `synthese_globale`, les IC sont corrigés du design effect propre à
    chaque indicatrice : ces métriques agrègent les 83 items corrélés d'une copie.

    Returns:
        Un DataFrame, une ligne par quantile, trié par difficulté croissante :
        difficulté moyenne (`x`), effectif, accord et son IC Wilson, kappa,
        rappel et son IC, précision.
    """
    from sklearn.metrics import cohen_kappa_score

    try:
        groupes = pd.qcut(copies[col_tri], q=n_quantiles, duplicates="drop")
    except ValueError:  # trop peu de valeurs distinctes pour des quantiles
        groupes = pd.cut(copies[col_tri], bins=n_quantiles)

    copies = copies.assign(_groupe=groupes)
    milieu = copies.groupby("_groupe", observed=True)[col_tri].mean()

    lignes = []
    for etiquette, copies_grp in copies.groupby("_groupe", observed=True):
        sub = df[df["copy_id"].isin(copies_grp.index)]
        if len(sub) < 10:
            continue
        grappes = sub["copy_id"]
        err_exp = sub["y_true"] != "1"
        err_mod = sub["y_pred"] != "1"
        n_me = int(err_mod.sum())
        n_vp = int((err_exp & err_mod).sum())
        accord, _ = _proportion_groupee(sub["y_true"] == sub["y_pred"], grappes)
        rappel, _ = _proportion_groupee(err_mod[err_exp], grappes[err_exp])
        try:
            kappa = float(cohen_kappa_score(sub["y_true"], sub["y_pred"]))
        except ValueError:
            kappa = float("nan")
        lignes.append(
            {
                "x": milieu[etiquette],
                "n_copies": len(copies_grp),
                "accord": accord.estimate,
                "accord_bas": accord.lower,
                "accord_haut": accord.upper,
                "kappa": kappa,
                "rappel": rappel.estimate if rappel else float("nan"),
                "rappel_bas": rappel.lower if rappel else float("nan"),
                "rappel_haut": rappel.upper if rappel else float("nan"),
                "precision": n_vp / n_me if n_me else float("nan"),
            }
        )
    return pd.DataFrame(lignes).sort_values("x")


# ── Comparaison multi-runs ────────────────────────────────────────────────────
def joindre_runs(runs: dict[str, Run]) -> pd.DataFrame | None:
    """Joint les runs sur (copy_id, item_id) au format attendu par `multi_model`.

    Équivalent de `multi_model.load_multi_runs`, mais à partir des DataFrames
    déjà en mémoire : le site ne relit pas les JSONL une seconde fois.

    Args:
        runs: runs chargés par `charger_runs`.

    Returns:
        Le DataFrame joint (jointure interne : seuls les items présents dans
        tous les runs), avec une colonne `y_pred__<libellé>` par run. None si
        moins de deux runs sont disponibles ou si l'intersection est vide.
    """
    if len(runs) < 2:
        return None
    premier = next(iter(runs.values()))
    joint = premier.df[["copy_id", "item_id", "y_true"]].copy()
    for label, run in runs.items():
        cols = run.df[["copy_id", "item_id", "y_pred", "confidence"]].rename(
            columns={"y_pred": f"y_pred__{label}", "confidence": f"conf__{label}"}
        )
        joint = joint.merge(cols, on=["copy_id", "item_id"], how="inner")
    if joint.empty:
        NOTES.append("Aucun item commun aux runs comparés : la section multi-modèles est omise.")
        return None
    return joint


# ── Écarts par copie ──────────────────────────────────────────────────────────
def classement_ecarts(df: pd.DataFrame) -> pd.DataFrame:
    """Copies triées par taux de désaccord décroissant, enrichi du détail utile.

    Args:
        df: prédictions à l'item d'un run.

    Returns:
        Un DataFrame indexé par copy_id, trié des pires copies aux meilleures :
        effectif, nombre et part de désaccords, accord, nombre d'erreurs vues par
        l'expert et par le modèle, écart signé sur ce nombre, parts d'items codés
        « 0 » (mot absent) et hors grille par l'expert, et — si le run porte le
        marqueur du pipeline — la colonne booléenne `vierge`. À part égale de
        désaccord, les copies sont triées par identifiant, pour que le classement
        soit reproductible.
    """
    classement = copies_by_disagreement(df)
    copies = per_copy_metrics(df)
    out = classement.join(copies[["n_erreurs_expert", "n_erreurs_modele", "confiance_moyenne"]])
    out["ecart_n_erreurs"] = out["n_erreurs_modele"] - out["n_erreurs_expert"]
    # Colonnes descriptives : elles éclairent la lecture des fiches mais ne
    # servent PAS à écarter des copies (seul le marqueur du pipeline le fait).
    out["pct_absent_expert"] = (df["y_true"] == "0").groupby(df["copy_id"]).mean() * 100
    out["pct_hors_grille_expert"] = (~df["y_true"].isin(CODES_VALIDES)).groupby(
        df["copy_id"]
    ).mean() * 100
    if COLONNE_VIERGE in df.columns:
        # Le marqueur est identique sur toutes les lignes d'une copie ; `max`
        # ramène simplement le booléen au niveau copie.
        out["vierge"] = df[COLONNE_VIERGE].fillna(False).groupby(df["copy_id"]).max().astype(bool)
    # Tri stable sur un index déjà ordonné : à taux de désaccord égal, l'ordre
    # reste celui des identifiants de copie.
    return out.sort_index().sort_values("pct_desaccord", ascending=False, kind="mergesort")


def couverture_copies(runs: dict[str, Run]) -> pd.DataFrame:
    """Couverture de chaque run : copies évaluées et copies écartées du corpus.

    Une copie manquante n'est pas neutre : le pipeline two-stage écarte les copies
    qu'il ne parvient pas à transcrire à l'étape 1, alors que l'end-to-end produit
    un codage pour toutes. Comparer les runs à effectif égal exige donc de savoir
    combien de copies chacun a laissées de côté.

    Args:
        runs: runs chargés par `charger_runs`.

    Le corpus de référence est l'union des copies vues par les runs de la MÊME
    année : les dictées 2015 et 2021 sont deux corpus distincts.

    Returns:
        Un DataFrame indexé par libellé de run : nombre de copies évaluées,
        nombre et part de copies du corpus (union des runs de l'année) non couvertes.
    """
    corpus_annee: dict[int, set[str]] = {}
    for run in runs.values():
        corpus_annee.setdefault(run.annee, set()).update(run.df["copy_id"].unique())
    lignes = []
    for label, run in runs.items():
        corpus = corpus_annee[run.annee]
        vues = set(run.df["copy_id"].unique())
        lignes.append(
            {
                "run": label,
                "n_copies": len(vues),
                "n_ecartees": len(corpus - vues),
                "pct_ecartees": len(corpus - vues) / len(corpus) * 100 if corpus else 0.0,
            }
        )
    return pd.DataFrame(lignes).set_index("run")


def seuil_encre() -> float:
    """Seuil de copie vierge du projet, lu dans la config du paquet.

    Returns:
        La valeur par défaut de `DataConfig.blank_ink_threshold`, pour que la page
        ne puisse pas afficher un seuil périmé si celui du code change. Retombe sur
        `SEUIL_ENCRE_PIPELINE` si la config n'est pas importable.
    """
    try:
        from evaluation_dictee.config import DataConfig

        return float(DataConfig.model_fields["blank_ink_threshold"].default)
    except Exception:  # noqa: BLE001 — la page doit rester rendable
        return SEUIL_ENCRE_PIPELINE


def charger_densites_encre(runs: dict[str, Run]) -> tuple[pd.DataFrame | None, str]:
    """Densité d'encre par copie, et provenance de la mesure.

    Source unique : la colonne `ink_ratio` du JSONL, que le benchmark écrit en même
    temps qu'il applique le seuil de copie vierge — c'est donc exactement la mesure
    qui a décidé. La mesure vit dans le pipeline et nulle part ailleurs : il n'existe
    pas de fichier de densités produit à côté, qui pourrait diverger de lui.

    Args:
        runs: runs chargés par `charger_runs`.

    Returns:
        Le couple (DataFrame indexé par copy_id avec la colonne `ink_ratio`,
        libellé de provenance). Le DataFrame vaut None si aucun run ne porte la
        colonne ; l'incident est alors consigné dans `NOTES`.
    """
    for label, run in runs.items():
        if COLONNE_ENCRE in run.df.columns:
            serie = run.df.groupby("copy_id")[COLONNE_ENCRE].first().dropna()
            if len(serie):
                return serie.to_frame(COLONNE_ENCRE), f"prédictions du run {label}"

    NOTES.append(
        f"Aucun run ne porte la colonne `{COLONNE_ENCRE}` : distribution des densités "
        "d'encre omise. Ces runs sont antérieurs à la mesure de densité — relancer le "
        "benchmark et réexporter pour l'obtenir."
    )
    return None, ""


def marqueur_vierge_disponible(runs: dict[str, Run]) -> bool:
    """Vrai si tous les runs portent le marqueur de copie vierge du pipeline.

    Args:
        runs: runs chargés par `charger_runs`.

    Returns:
        True seulement si chaque run a la colonne `blank`. Les runs exportés avant
        l'ajout du filtre de densité d'encre ne l'ont pas : aucune copie ne peut
        alors être écartée, et la page doit le signaler au lieu de deviner.
    """
    return bool(runs) and all(COLONNE_VIERGE in run.df.columns for run in runs.values())


def separer_copies_vierges(classement: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les copies vierges du reste du classement, selon le pipeline.

    Le critère est le marqueur `blank` posé par `pipeline/benchmark.py` d'après la
    densité d'encre de l'image. En son absence (run antérieur au garde-fou), rien
    n'est écarté : mieux vaut un classement bruité et signalé comme tel qu'un
    filtrage improvisé qui divergerait du pipeline.

    Args:
        classement: sortie de `classement_ecarts`.

    Returns:
        Le couple (copies vierges, copies exploitables). L'ordre du classement est
        conservé de part et d'autre ; le premier terme est vide si le run ne porte
        pas le marqueur.
    """
    if "vierge" not in classement.columns:
        return classement.iloc[:0], classement
    vierge = classement["vierge"]
    return classement[vierge], classement[~vierge]


def nature_copie(ligne: pd.Series) -> str:
    """Qualifie une copie d'après ce que l'annotateur a pu y coder.

    Sert à annoter les fiches : une copie que l'expert n'a pas pu coder sur une
    large part de ses items s'interprète autrement qu'une copie ordinaire, même
    quand la densité d'encre ne la classe pas comme vierge (écriture illisible).

    Args:
        ligne: une ligne de `classement_ecarts`.

    Returns:
        Un libellé court, ou une chaîne vide pour une copie ordinaire.
    """
    if bool(ligne.get("vierge", False)):
        return "copie vierge selon le pipeline — densité d'encre sous le seuil"
    absent = float(ligne["pct_absent_expert"])
    hors = float(ligne["pct_hors_grille_expert"])
    total = absent + hors
    if total < 25:
        return ""
    motif = "illisibles" if hors > absent else "absents"
    return (
        f"copie largement incodable — l'expert code {total:.0f} % des items comme "
        f"{motif}, alors que la densité d'encre ne la classe pas vierge"
    )


def transcription_modele(sub: pd.DataFrame) -> tuple[str, str]:
    """Transcription du modèle pour une copie, et sa provenance.

    Args:
        sub: lignes de prédiction d'une seule copie, dans l'ordre de la dictée.

    Returns:
        Le couple (texte, provenance). La transcription brute de l'étape 1 (HTR)
        est prioritaire ; à défaut le texte est recollé item par item.
    """
    if "raw_transcription" in sub.columns:
        brutes = sub["raw_transcription"].dropna()
        brutes = brutes[brutes.astype(str).str.strip() != ""]
        if len(brutes):
            return str(brutes.iloc[0]).strip(), "étape 1 (HTR), lecture brute du modèle"
    if "transcription" in sub.columns:
        mots = [str(m) for m in sub["transcription"].fillna("").tolist() if str(m).strip()]
        if mots:
            return " ".join(mots), "recollée item par item"
    return "", "non fournie par ce run"


def decomposition_copie(sub: pd.DataFrame) -> str:
    """Résume les transitions expert → modèle d'une copie (« 1→9 : 12 ; … »)."""
    dis = sub[sub["y_true"] != sub["y_pred"]]
    if dis.empty:
        return "aucun désaccord"
    compte = dis.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False)
    return " ; ".join(f"{t or '∅'}→{p} : {n}" for (t, p), n in compte.items())


# ── Rendu HTML des codages ────────────────────────────────────────────────────
def _puce(mot: str, code: str, classe: str, infobulle: str, detail: str = "") -> str:
    """Puce HTML d'un item : mot attendu au-dessus, code en dessous."""
    sup = f"<span class='puce-detail'>{html.escape(detail)}</span>" if detail else ""
    return (
        f"<span class='puce {classe}' title='{html.escape(infobulle)}'>"
        f"<span class='puce-mot'>{html.escape(mot)}</span>"
        f"<span class='puce-code'>{html.escape(code)}</span>{sup}</span>"
    )


def codage_expert_html(
    items: list[GridItem], codes_expert: dict[str, str], codes_modele: dict[str, str]
) -> str:
    """Bandeau des codes de l'annotateur expert, nuancé par code.

    Args:
        items: items de la grille, dans l'ordre de la dictée.
        codes_expert: item_id → code expert.
        codes_modele: item_id → code modèle (sert à marquer les items divergents).

    Returns:
        Le fragment HTML du bandeau de puces.
    """
    puces = []
    for it in items:
        code = codes_expert.get(it.item_id, "?")
        divergent = codes_modele.get(it.item_id, "?") != code
        classe = f"puce-code-{code if code in CODES_VALIDES else 'autre'}"
        if divergent:
            classe += " puce-divergente"
        libelle = LIBELLE_CODE.get(code, f"code « {code or 'vide'} »")
        puces.append(_puce(it.attendu, code or "∅", classe, f"expert : {libelle}"))
    return f"<div class='bandeau-puces'>{''.join(puces)}</div>"


def codage_modele_html(
    items: list[GridItem],
    codes_expert: dict[str, str],
    codes_modele: dict[str, str],
    lectures: dict[str, str] | None = None,
) -> str:
    """Bandeau des codes du modèle : **vert** si accord avec l'expert, **rouge** sinon.

    Args:
        items: items de la grille, dans l'ordre de la dictée.
        codes_expert: item_id → code expert.
        codes_modele: item_id → code modèle.
        lectures: item_id → mot lu par le modèle ; affiché sous la puce quand il
            diffère du mot attendu (c'est là que se logent les erreurs de lecture).

    Returns:
        Le fragment HTML du bandeau de puces.
    """
    lectures = lectures or {}
    puces = []
    for it in items:
        code_m = codes_modele.get(it.item_id, "?")
        code_e = codes_expert.get(it.item_id, "?")
        accord = code_m == code_e
        classe = "puce-accord" if accord else "puce-desaccord"
        infobulle = (
            f"modèle : {LIBELLE_CODE.get(code_m, code_m)} — accord avec l'expert"
            if accord
            else f"modèle : {LIBELLE_CODE.get(code_m, code_m)} / "
            f"expert : {LIBELLE_CODE.get(code_e, code_e or 'vide')}"
        )
        lu = str(lectures.get(it.item_id, "") or "").strip()
        detail = ""
        if not accord:
            detail = f"exp. {code_e or '∅'}"
        if lu and lu != it.attendu:
            detail = f"{detail} · lu « {lu} »" if detail else f"lu « {lu} »"
        puces.append(_puce(it.attendu, code_m, classe, infobulle, detail))
    return f"<div class='bandeau-puces'>{''.join(puces)}</div>"


def legende_puces() -> str:
    """Légende des couleurs des bandeaux de puces."""
    return (
        "<div class='legende-puces'>"
        "<span class='puce puce-accord'><span class='puce-mot'>accord</span></span>"
        "<span class='puce puce-desaccord'><span class='puce-mot'>désaccord</span></span>"
        "<span class='puce puce-code-1'><span class='puce-mot'>1 correct</span></span>"
        "<span class='puce puce-code-9'><span class='puce-mot'>9 erreur</span></span>"
        "<span class='puce puce-code-0'><span class='puce-mot'>0 absent</span></span>"
        "</div>"
    )


# Fonctions du paquet ré-exportées telles quelles : les pages passent par ce
# module pour y accéder, afin qu'un import raté n'empêche pas le rendu (le bloc
# try/except ci-dessus est alors le seul point de défaillance).
__all__ = [
    "agreement_per_item",
    "confidence_score",
    "consensus_levels",
    "disagreement_decomposition",
    "disagreement_type_summary",
    "fleiss_kappa",
    "pairwise_agreement",
    "referral_curve_with_ci",
    "wilson_interval",
]
