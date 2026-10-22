import pytest

from src.plants.catalog import OTHER_OPTION, PLANTVILLAGE_LABELS, PlantCatalog
from src.plants.conditions import CONDITIONS


def test_catalog_has_14_species_and_38_classes(catalog):
    assert len(catalog.species) == 14
    assert sum(len(s.labels) for s in catalog.species.values()) == 38


def test_every_label_has_condition_info():
    assert set(CONDITIONS) == set(PLANTVILLAGE_LABELS)


@pytest.mark.parametrize("name,expected", [
    ("Tomato", "Tomato"), ("  tomato ", "Tomato"), ("TOMATOES", "Tomato"),
    ("maize", "Corn (maize)"), ("Corn (maize)", "Corn (maize)"),
    ("bell pepper", "Bell pepper"), ("Capsicum", "Bell pepper"),
    ("sour cherry", "Cherry (incl. sour cherry)"), ("Grapevine", "Grape"),
])
def test_supported_plant_names_match(catalog, name, expected):
    match = catalog.match(name)
    assert match.supported
    assert match.species.display_name == expected


@pytest.mark.parametrize("name", ["Peace lily", "Rose", "pepper", "peppermint", "Pumpkin", "Lemon", "", OTHER_OPTION])
def test_unsupported_or_ambiguous_names_do_not_match(catalog, name):
    assert not catalog.match(name).supported


def test_selector_lists_display_names_plus_other(catalog):
    opts = catalog.selector_options()
    assert opts[-1] == OTHER_OPTION
    assert "Tomato" in opts and "Bell pepper" in opts and len(opts) == 15


def test_coverage_warnings_for_limited_species(catalog):
    assert "healthy leaves only" in catalog.match("Blueberry").species.coverage_warning
    assert "never saw healthy" in catalog.match("Orange").species.coverage_warning
    assert "never saw healthy" in catalog.match("Squash").species.coverage_warning
    assert "only one disease" in catalog.match("Peach").species.coverage_warning
    assert catalog.match("Tomato").species.coverage_warning is None


def test_label_indices_belong_to_species(catalog):
    tomato = catalog.match("tomato").species
    assert all(catalog.class_names[i].startswith("Tomato___") for i in catalog.label_indices(tomato))
    assert len(catalog.label_indices(tomato)) == 10


def test_empty_catalog_rejected():
    with pytest.raises(ValueError):
        PlantCatalog([])
