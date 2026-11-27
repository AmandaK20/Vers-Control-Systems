import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from preprocessing import load_and_prepare_base, get_task_splits, SEED

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_evaluate(df: pd.DataFrame, task: str, task_type: str, epochs: int = 10, batch_size: int = 512):
    X_train, X_test, y_train, y_test = get_task_splits(df, task=task, seed=SEED)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = TabularMLP(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if task_type == 'classification':
        pos_count = max(float(y_train.sum()), 1.0)
        neg_count = float(len(y_train) - pos_count)
        pos_weight = torch.tensor([min(neg_count / pos_count, 10.0)], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        raw_out = model(torch.tensor(X_test, device=DEVICE)).cpu().numpy()

    result = {
        'model': 'Neural Network (MLP)',
        'task': task,
        'task_type': task_type,
        'split_seed': SEED
    }

    if task_type == 'classification':
        probs = 1.0 / (1.0 + np.exp(-raw_out))
        preds = (probs >= 0.5).astype(int)
        result.update({
            'accuracy': round(accuracy_score(y_test, preds), 4),
            'precision': round(precision_score(y_test, preds, zero_division=0), 4),
            'recall': round(recall_score(y_test, preds, zero_division=0), 4),
            'f1_score': round(f1_score(y_test, preds, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y_test, probs), 4),
            'mae': np.nan, 'rmse': np.nan, 'r2': np.nan
        })
    else:
        result.update({
            'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan,
            'f1_score': np.nan, 'roc_auc': np.nan,
            'mae': round(mean_absolute_error(y_test, raw_out), 4),
            'rmse': round(np.sqrt(mean_squared_error(y_test, raw_out)), 4),
            'r2': round(r2_score(y_test, raw_out), 4)
        })

    print(f"[{task}] Done -> {result}")
    return result


def main():
    os.makedirs('results', exist_ok=True)
    df = load_and_prepare_base('data/DataCoSupplyChainDataset.csv')

    tasks = [
        ('fraud', 'classification'),
        ('late_delivery', 'classification'),
        ('sales', 'regression'),
        ('product_quantity', 'regression')
    ]

    results = [train_and_evaluate(df, task, task_type) for task, task_type in tasks]
    pd.DataFrame(results).to_csv('results/nn_results.csv', index=False)
    print("\nSaved all NN results to results/nn_results.csv")


if __name__ == '__main__':
    main()
