from pathlib import Path

import pytest

from mri_radiomics.agent.memory import Store
from mri_radiomics.model import get_model

SAMPLES = Path(__file__).resolve().parent.parent / "assets" / "samples"


@pytest.fixture(scope="session")
def model():
    return get_model()


@pytest.fixture
def store():
    return Store()


@pytest.fixture
def case_ids(model, store):
    ids = {}
    for label in ("glioma", "meningioma", "pituitary", "notumor"):
        feats = model.featurize(SAMPLES / f"{label}.jpg")
        ids[label] = store.add_case(f"{label}.jpg", feats, model.predict(feats).as_dict())
    return ids
