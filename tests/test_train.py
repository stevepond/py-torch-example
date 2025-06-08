import pytest

torch = pytest.importorskip('torch')

from torch_app import train, model


def test_train_model_reduces_loss():
    dataset = train.generate_data(n_samples=50, input_dim=5)
    m = model.SimpleClassifier(5)
    loss = train.train_model(m, dataset, lr=0.1, epochs=3)
    assert loss >= 0


def test_tune_model_selects_best_lr():
    dataset = train.generate_data(n_samples=50, input_dim=5)
    lrs = [0.01, 0.05, 0.1]
    best_lr, best_loss = train.tune_model(lrs, dataset, epochs=2, input_dim=5)
    assert best_lr in lrs
    assert best_loss >= 0
