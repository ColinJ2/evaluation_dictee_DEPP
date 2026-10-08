"""Chargement des imagettes et des labels experts (local ou S3).

Les fichiers DEPP 2015 sont des TIFF 1 bit à extension trompeuse ".png" ; ceux de 2021
sont de vrais PNG à palette, noir et blanc eux aussi. On les normalise systématiquement
en niveaux de gris, sans perte ni redimensionnement.
"""

from __future__ import annotations

import csv
import io
import random
from collections.abc import Iterator
from dataclasses import dataclass, field

import fsspec
import numpy as np
from PIL import Image


@dataclass
class Copy:
    """Une copie d'élève : son identifiant, son image et les codes experts."""

    copy_id: str
    image_path: str
    expert_codes: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)


def _join(base: str, name: str) -> str:
    """Concatène un dossier (local ou s3://) et un nom de fichier.

    Args:
        base: Dossier de base, avec ou sans slash final.
        name: Nom du fichier à ajouter.

    Returns:
        Le chemin complet `base/name` (un seul slash de séparation).
    """
    return base.rstrip("/") + "/" + name


def load_image(path: str) -> Image.Image:
    """Charge une image (locale ou S3) en niveaux de gris (gère les TIFF 1 bit déguisés en .png).

    Args:
        path: Chemin local ou S3 de l'image.

    Returns:
        L'image convertie en niveaux de gris (mode "L").
    """
    with fsspec.open(path, "rb") as f:
        img = Image.open(io.BytesIO(f.read()))
    return img.convert("L")


def ink_ratio(image: Image.Image) -> float:
    """Proportion de pixels sombres (encre) d'une image en niveaux de gris.

    Sert à repérer les formulaires vierges (l'élève n'a rien écrit) : seul le
    pré-imprimé (titre, lignes, cadre, pied de page) marque quelques pour cent.

    Args:
        image: Image PIL (convertie en "L" si besoin).

    Returns:
        Fraction de pixels d'intensité < 128, entre 0.0 et 1.0.
    """
    arr = np.asarray(image.convert("L"))
    return float((arr < 128).mean())


def load_labels(csv_path: str) -> dict[str, dict[str, str]]:
    """Charge les codes de l'annotateur depuis le CSV de correction (séparateur ';', local ou S3).

    Deux structures coexistent selon le millésime, reconnues à l'en-tête :
    - large (2015) : colonne d'index, colonne nom d'image, puis une colonne par item ;
    - long (2021) : colonnes `dictee;num_item;corrige`, une ligne par copie x item.

    Args:
        csv_path: Chemin local ou S3 du CSV de correction.

    Returns:
        Un dictionnaire `copy_id -> {item_id: code}`, valeurs nettoyées des espaces.
    """
    labels: dict[str, dict[str, str]] = {}
    with fsspec.open(csv_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = [col.strip() for col in next(reader)]
        if "num_item" in header:
            return _load_labels_long(reader, header)
        item_ids = header[2:]
        for row in reader:
            if not row:
                continue
            copy_id = row[1].strip()
            labels[copy_id] = {item: row[i + 2].strip() for i, item in enumerate(item_ids)}
    return labels


def _load_labels_long(reader: Iterator[list[str]], header: list[str]) -> dict[str, dict[str, str]]:
    """Pivote un CSV de correction au format long (une ligne par copie x item).

    Dans le fichier 2021, les lignes d'une copie ne suivent pas l'ordre de la dictée.
    Or cet ordre compte : il fixe celui des items dans le prompt et sert au
    ré-alignement. Les items sont donc triés par identifiant, de largeur fixe
    (`E5FFO1540101` + numéro sur deux chiffres), ce qui redonne l'ordre de la grille.

    Args:
        reader: Lecteur CSV positionné après l'en-tête.
        header: En-tête du fichier (doit contenir `dictee`, `num_item` et `corrige`).

    Returns:
        Un dictionnaire `copy_id -> {item_id: code}`, items dans l'ordre de la dictée.
    """
    i_copie, i_item, i_code = (header.index(col) for col in ("dictee", "num_item", "corrige"))
    par_copie: dict[str, dict[str, str]] = {}
    for row in reader:
        if not row:
            continue
        par_copie.setdefault(row[i_copie].strip(), {})[row[i_item].strip()] = row[i_code].strip()
    return {
        copy_id: {item: codes[item] for item in sorted(codes)}
        for copy_id, codes in par_copie.items()
    }


def _exists(path: str) -> bool:
    """Teste l'existence d'un fichier local ou S3.

    Args:
        path: Chemin local ou S3 à vérifier.

    Returns:
        True si le fichier existe, False sinon.
    """
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return bool(paths) and fs.exists(paths[0])


def load_dataset(
    images_dir: str,
    labels_csv: str,
    limit: int | None = None,
    seed: int | None = None,
) -> list[Copy]:
    """Construit la liste des copies à partir des images et du CSV de labels.

    Ne retient que les copies dont l'image existe réellement dans `images_dir`.

    Args:
        images_dir: Dossier (local ou S3) contenant les imagettes.
        labels_csv: Chemin du CSV des codes experts.
        limit: Nombre maximal de copies à charger (toutes si None).
        seed: Graine d'un tirage aléatoire des `limit` copies ; None = les premières
            par identifiant.

    Returns:
        La liste des copies trouvées, triées par identifiant.
    """
    labels = load_labels(labels_csv)

    copy_ids = sorted(labels)
    if seed is not None:
        # Mélange d'une liste triée : le tirage ne dépend que de la graine et du
        # corpus, et ses N premiers éléments ne changent pas quand `limit` augmente.
        random.Random(seed).shuffle(copy_ids)

    copies: list[Copy] = []
    for copy_id in copy_ids:
        image_path = _join(images_dir, copy_id)
        if not _exists(image_path):
            continue
        item_codes = labels[copy_id]
        copies.append(
            Copy(
                copy_id=copy_id,
                image_path=image_path,
                item_ids=list(item_codes.keys()),
                expert_codes=list(item_codes.values()),
            )
        )
        if limit is not None and len(copies) >= limit:
            break
    return sorted(copies, key=lambda c: c.copy_id)
