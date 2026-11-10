from conftest import SAMPLES


def test_features_shape(model):
    feats = model.featurize(SAMPLES / "glioma.jpg")
    assert len(feats) == 46 and set(model.features) == set(feats)


def test_predict_samples(model):
    for label in ("glioma", "meningioma", "pituitary", "notumor"):
        pred = model.predict(model.featurize(SAMPLES / f"{label}.jpg"))
        assert pred.label == label
        assert abs(sum(pred.probabilities.values()) - 1) < 1e-3


def test_explain_and_similar(model):
    feats = model.featurize(SAMPLES / "pituitary.jpg")
    drivers = model.explain(feats, 3)
    assert len(drivers) == 3 and all("deviation_from_healthy_std" in d for d in drivers)
    neighbours = model.similar(feats, 5)
    assert len(neighbours) == 5 and neighbours[0]["distance"] <= neighbours[-1]["distance"]


def test_model_card_metrics(model):
    assert model.metrics["test_accuracy"] > 0.85
