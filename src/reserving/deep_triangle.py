"""
Module Reserving — Deep Triangle (adaptation simplifiée de Kuo, 2019).

GRU partagé prédisant l'incrément de paiement suivant à partir des
incréments observés, normalisés par la prime acquise. Complète le
triangle de façon autorégressive à l'inférence.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TriangleSeqDataset(Dataset):
    """Paires (input, target) en téléforçage, restreintes aux cellules observées."""

    def __init__(self, X, M):
        self.X, self.M = X, M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, m = self.X[idx], self.M[idx]
        obs_len = int(m.sum())

        input_seq = np.zeros(9, dtype=np.float32)
        target_seq = np.zeros(9, dtype=np.float32)
        valid_mask = np.zeros(9, dtype=np.float32)

        if obs_len >= 2:
            input_seq[: obs_len - 1] = x[: obs_len - 1]
            target_seq[: obs_len - 1] = x[1:obs_len]
            valid_mask[: obs_len - 1] = 1.0

        return {
            "input": torch.tensor(input_seq).unsqueeze(-1),
            "target": torch.tensor(target_seq),
            "mask": torch.tensor(valid_mask),
        }


class DeepTriangleGRU(nn.Module):
    """GRU many-to-many, prédiction du prochain incrément à chaque pas."""

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.output(out).squeeze(-1)


def masked_mse_loss(pred, target, mask):
    sq_err = (pred - target) ** 2 * mask
    return sq_err.sum() / mask.sum().clamp(min=1)


def train_deep_triangle(model, dataset, n_epochs, optimizer, device, batch_size=64):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for batch in loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = masked_mse_loss(pred, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch+1}/{n_epochs} — loss: {sum(losses)/len(losses):.6f}")

    return model


def predict_future_increments(model, obs_seq, device):
    """
    Complète la séquence de façon autorégressive jusqu'au lag 10.
    Les prédictions sont clippées à 0 avant d'être réinjectées comme input
    du pas suivant, pour éviter la propagation de valeurs négatives
    incohérentes (un paiement cumulé ne peut pas décroître) sans pour
    autant contraindre l'architecture pendant l'entraînement.
    """
    model.eval()
    n_future = 10 - len(obs_seq)
    if n_future == 0:
        return np.array([])

    with torch.no_grad():
        x = torch.tensor(obs_seq, dtype=torch.float32, device=device).view(1, -1, 1)
        out, h = model.gru(x)
        next_pred = model.output(out[:, -1, :]).clamp(min=0.0)
        preds = [next_pred.item()]
        current_input = next_pred.view(1, 1, 1)

        for _ in range(n_future - 1):
            out_step, h = model.gru(current_input, h)
            next_pred = model.output(out_step[:, -1, :]).clamp(min=0.0)
            preds.append(next_pred.item())
            current_input = next_pred.view(1, 1, 1)

    return np.array(preds)