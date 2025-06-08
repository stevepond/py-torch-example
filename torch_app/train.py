import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from typing import Iterable, Tuple

from .model import SimpleClassifier


def generate_data(n_samples: int = 100, input_dim: int = 10) -> TensorDataset:
    """Generate random classification data."""
    x = torch.randn(n_samples, input_dim)
    true_w = torch.randn(input_dim, 1)
    logits = x @ true_w
    y = (logits > 0).float()
    return TensorDataset(x, y)


def train_model(
    model: nn.Module,
    dataset: TensorDataset,
    lr: float = 0.1,
    epochs: int = 5,
) -> float:
    """Train the model and return the final loss."""
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return loss.item()


def tune_model(
    lr_values: Iterable[float],
    dataset: TensorDataset,
    epochs: int = 5,
    input_dim: int = 10,
) -> Tuple[float, float]:
    """Tune learning rate and return (best_lr, best_loss)."""
    best_loss = float('inf')
    best_lr = None
    for lr in lr_values:
        model = SimpleClassifier(input_dim)
        final_loss = train_model(model, dataset, lr=lr, epochs=epochs)
        if final_loss < best_loss:
            best_loss = final_loss
            best_lr = lr
    return best_lr, best_loss


def main() -> None:
    dataset = generate_data()
    lrs = [0.01, 0.05, 0.1]
    best_lr, best_loss = tune_model(lrs, dataset)
    print(f"Best LR: {best_lr}, Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
