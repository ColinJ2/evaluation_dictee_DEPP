"""Tests du chargement des données : CSV large (2015) et long (2021), tirage des copies."""

from pathlib import Path

import pytest

from evaluation_dictee.data.loaders import load_dataset, load_labels


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[str, str]:
    """Corpus de 30 copies (images vides, CSV long), dont c07 sans image."""
    images = tmp_path / "images"
    images.mkdir()
    lignes = ["dictee;num_item;corrige"]
    for k in range(30):
        nom = f"c{k:02d}.png"
        if k != 7:
            (images / nom).touch()
        lignes.append(f"{nom};E5FFO154010101;1")
    csv = tmp_path / "labels.csv"
    csv.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return str(images), str(csv)


def _ids(images: str, csv: str, **kwargs) -> list[str]:
    return [c.copy_id for c in load_dataset(images_dir=images, labels_csv=csv, **kwargs)]


def test_sans_graine_les_premieres_copies(corpus: tuple[str, str]) -> None:
    """Comportement historique : les premières copies par identifiant, sans image sautée."""
    assert _ids(*corpus, limit=8) == [f"c{k:02d}.png" for k in (0, 1, 2, 3, 4, 5, 6, 8)]


def test_tirage_aleatoire_reproductible_et_emboite(corpus: tuple[str, str]) -> None:
    """Même graine, même tirage ; un `limit` plus grand prolonge l'échantillon."""
    petit = _ids(*corpus, limit=5, seed=42)
    grand = _ids(*corpus, limit=10, seed=42)

    assert petit == _ids(*corpus, limit=5, seed=42)
    assert petit != _ids(*corpus, limit=5)  # pas les premières copies
    assert set(petit) <= set(grand)
    assert petit == sorted(petit)
    assert "c07.png" not in grand  # une copie sans image n'est jamais retenue
    assert len(grand) == 10


def test_format_large_2015(tmp_path: Path) -> None:
    """Une ligne par copie, une colonne par item, dans l'ordre de l'en-tête."""
    csv = tmp_path / "large.csv"
    csv.write_text(
        ";dictee;E5FFO154010101;E5FFO154010102\n0;a.png;1;4\n1;b.png; 0 ;1\n",
        encoding="utf-8",
    )

    assert load_labels(str(csv)) == {
        "a.png": {"E5FFO154010101": "1", "E5FFO154010102": "4"},
        "b.png": {"E5FFO154010101": "0", "E5FFO154010102": "1"},
    }


def test_format_long_2021_remet_les_items_dans_l_ordre(tmp_path: Path) -> None:
    """Une ligne par copie x item, lignes désordonnées : items rangés comme la dictée."""
    csv = tmp_path / "long.csv"
    csv.write_text(
        "dictee;num_item;corrige\n"
        "a.png;E5FFO154010110;9\n"
        "a.png;E5FFO154010102;4\n"
        "b.png;E5FFO154010101;1\n"
        "a.png;E5FFO154010101;1\n"
        "\n"
        "b.png;E5FFO154010110; 0 \n"
        "b.png;E5FFO154010102;3\n",
        encoding="utf-8",
    )

    labels = load_labels(str(csv))

    assert labels == {
        "a.png": {"E5FFO154010101": "1", "E5FFO154010102": "4", "E5FFO154010110": "9"},
        "b.png": {"E5FFO154010101": "1", "E5FFO154010102": "3", "E5FFO154010110": "0"},
    }
    # L'ordre des items, et pas seulement leur contenu : il fixe l'ordre du prompt.
    assert list(labels["a.png"]) == ["E5FFO154010101", "E5FFO154010102", "E5FFO154010110"]
