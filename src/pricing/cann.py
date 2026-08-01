"""
Module Pricing — Architecture CANN (Combined Actuarial Neural Network).

Implémente l'approche Wüthrich & Merz (2019) : le GLM Poisson sert de
composante fixe (skip connection), un réseau de neurones apprend le résidu
non capturé par le GLM à partir des variables brutes.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


CONTINUOUS_COLS = ["VehPower_norm", "VehAge_norm", "DrivAge_norm", "BonusMalus_norm", "Density_log"]
CATEGORICAL_COLS = ["VehBrand_code", "Region_code", "Area_code", "VehGas_code"]

# Cardinalités observées sur le portefeuille (nb de modalités par variable catégorielle)
CATEGORICAL_CARDINALITIES = {
    "VehBrand_code": 11,
    "Region_code": 21,
    "Area_code": 6,
    "VehGas_code": 2,
}


class FreMTPL2Dataset(Dataset):
    """Dataset PyTorch pour le CANN de fréquence.
    
    Optimisé: conversion tensor à la demande (lazy loading) pour accélérer l'initialisation.
    """

    def __init__(self, df):
        self.continuous = df[CONTINUOUS_COLS].values
        self.categorical = df[CATEGORICAL_COLS].values
        self.glm_log_pred = df["glm_log_pred"].values
        self.exposure = df["Exposure"].values
        self.claim_nb = df["ClaimNb"].values

    def __len__(self):
        return len(self.claim_nb)

    def __getitem__(self, idx):
        return {
            "continuous": torch.tensor(self.continuous[idx], dtype=torch.float32),
            "categorical": torch.tensor(self.categorical[idx], dtype=torch.long),
            "glm_log_pred": torch.tensor(self.glm_log_pred[idx], dtype=torch.float32),
            "exposure": torch.tensor(self.exposure[idx], dtype=torch.float32),
            "claim_nb": torch.tensor(self.claim_nb[idx], dtype=torch.float32),
        }


class CANNFrequencyNet(nn.Module):
    """
    Réseau résiduel du CANN de fréquence.

    Architecture : embeddings pour les variables catégorielles, concaténés
    aux variables continues, passés dans un MLP. La sortie du MLP est
    ajoutée à la prédiction du GLM (skip connection), conformément à
    Wüthrich & Merz (2019).

    Initialisation critique : la dernière couche est initialisée à zéro,
    de sorte qu'au début de l'entraînement, le CANN reproduit exactement
    les prédictions du GLM (le réseau n'a encore rien appris).
    """

    def __init__(self, n_continuous, categorical_cardinalities, embedding_dim=4, hidden_dim=32):
        super().__init__()

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(card, embedding_dim)
            for col, card in categorical_cardinalities.items()
        })
        self.categorical_cols = list(categorical_cardinalities.keys())

        n_embedding_out = embedding_dim * len(categorical_cardinalities)
        input_dim = n_continuous + n_embedding_out

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialisation à zéro de la dernière couche : le CANN démarre = GLM
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, continuous, categorical, glm_log_pred):
        embedded = [
            self.embeddings[col](categorical[:, i])
            for i, col in enumerate(self.categorical_cols)
        ]
        embedded = torch.cat(embedded, dim=1)

        x = torch.cat([continuous, embedded], dim=1)
        residual = self.mlp(x).squeeze(-1)

        # Skip connection : log(lambda_CANN) = log(lambda_GLM) + résidu appris
        log_lambda = glm_log_pred + residual
        return log_lambda


def poisson_deviance_loss(log_lambda, exposure, claim_nb):
    """
    Perte = déviance de Poisson (à minimiser), cohérente avec la métrique
    utilisée pour évaluer le GLM. mu = exposure * exp(log_lambda).

    log_lambda est borné avant l'exponentielle pour éviter toute explosion
    numérique (overflow) en début d'entraînement, quand le réseau résiduel
    n'est pas encore stabilisé.
    """
    log_lambda = torch.clamp(log_lambda, min=-20.0, max=5.0)
    mu = exposure * torch.exp(log_lambda)
    mu = torch.clamp(mu, min=1e-8, max=1e8)

    # Poisson deviance: 2 * (y * log(y/mu) - (y - mu))
    # Handle y=0 case separately for numerical stability
    # When y=0: deviance = 2 * mu
    # When y>0: deviance = 2 * (y * log(y/mu) - (y - mu))
    
    # Compute log(y/mu) safely
    log_ratio = torch.log(claim_nb + 1e-8) - torch.log(mu)
    
    # For y=0, the term y*log(y/mu) should be 0
    # Use where to handle this
    y_log_term = torch.where(
        claim_nb > 0,
        claim_nb * log_ratio,
        torch.zeros_like(claim_nb)
    )
    
    dev = 2 * (y_log_term - (claim_nb - mu))
    
    # Clamp to prevent extreme values
    dev = torch.clamp(dev, min=-1e6, max=1e6)
    
    return dev.mean()


def train_cann(model, train_dataset, valid_dataset, n_epochs=15, batch_size=4096, lr=1e-4, device="cpu"):
    """
    Boucle d'entraînement du CANN de fréquence.
    Sauvegarde et recharge les poids au meilleur point de validation
    (le modèle retourné n'est pas nécessairement celui de la dernière epoch).
    """
    from torch.utils.data import DataLoader
    import copy

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "valid_loss": [], "best_valid_loss": float("inf"), "best_epoch": 0}
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            glm_log_pred = batch["glm_log_pred"].to(device)
            exposure = batch["exposure"].to(device)
            claim_nb = batch["claim_nb"].to(device)

            optimizer.zero_grad()
            log_lambda = model(continuous, categorical, glm_log_pred)
            loss = poisson_deviance_loss(log_lambda, exposure, claim_nb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in valid_loader:
                continuous = batch["continuous"].to(device)
                categorical = batch["categorical"].to(device)
                glm_log_pred = batch["glm_log_pred"].to(device)
                exposure = batch["exposure"].to(device)
                claim_nb = batch["claim_nb"].to(device)

                log_lambda = model(continuous, categorical, glm_log_pred)
                loss = poisson_deviance_loss(log_lambda, exposure, claim_nb)
                valid_losses.append(loss.item())

        avg_train = sum(train_losses) / len(train_losses)
        avg_valid = sum(valid_losses) / len(valid_losses)
        history["train_loss"].append(avg_train)
        history["valid_loss"].append(avg_valid)

        if avg_valid < history["best_valid_loss"]:
            history["best_valid_loss"] = avg_valid
            history["best_epoch"] = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch+1}/{n_epochs} — train_loss: {avg_train:.4f} — valid_loss: {avg_valid:.4f}")

    print(f"\nMeilleur valid_loss : {history['best_valid_loss']:.4f} (epoch {history['best_epoch']})")
    model.load_state_dict(best_state)
    return model, history

class PairInteractionNet(nn.Module):
    """
    Réseau dédié à l'exploration d'une interaction entre DEUX variables
    continues, conformément à l'approche de Schelldorfer & Wüthrich (2019),
    Section 3.5. Le GLM sert de working weight fixe (offset), ce réseau
    n'apprend que le résidu multiplicatif propre à cette paire.
    """

    def __init__(self, hidden_dim=20):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, var1, var2, log_mu_glm):
        x = torch.stack([var1, var2], dim=1)
        residual = self.mlp(x).squeeze(-1)
        log_lambda = log_mu_glm + residual
        return log_lambda


class PairDataset(Dataset):
    """Dataset pour tester une interaction entre deux variables continues.
    
    Optimisé: conversion tensor à la demande (lazy loading) pour accélérer l'initialisation.
    """

    def __init__(self, df, var1_col, var2_col):
        self.var1 = df[var1_col].values
        self.var2 = df[var2_col].values
        self.log_mu_glm = df["log_mu_glm"].values
        self.exposure = df["Exposure"].values
        self.claim_nb = df["ClaimNb"].values

    def __len__(self):
        return len(self.claim_nb)

    def __getitem__(self, idx):
        return {
            "var1": torch.tensor(self.var1[idx], dtype=torch.float32),
            "var2": torch.tensor(self.var2[idx], dtype=torch.float32),
            "log_mu_glm": torch.tensor(self.log_mu_glm[idx], dtype=torch.float32),
            "exposure": torch.tensor(self.exposure[idx], dtype=torch.float32),
            "claim_nb": torch.tensor(self.claim_nb[idx], dtype=torch.float32),
        }



def poisson_deviance_loss_v2(log_lambda, claim_nb):
    """Variante où log_lambda est déjà en échelle mu = exposure * lambda (pas besoin de multiplier)."""
    log_lambda = torch.clamp(log_lambda, min=-20.0, max=10.0)
    mu = torch.exp(log_lambda)
    mu = torch.clamp(mu, min=1e-8)

    safe_claim_nb = torch.where(claim_nb > 0, claim_nb, torch.ones_like(claim_nb))
    log_term = torch.where(
        claim_nb > 0,
        claim_nb * torch.log(safe_claim_nb / mu),
        torch.zeros_like(claim_nb)
    )
    dev = 2 * (log_term - (claim_nb - mu))
    return dev.mean()


class GroupInteractionNet(nn.Module):
    """
    Réseau dédié à l'exploration d'une interaction entre un groupe de
    variables (continues + catégorielles), conformément à l'approche de
    Schelldorfer & Wüthrich (2019), Section 3.5. Ici : VehPower, VehAge,
    VehGas (continues/binaire) + VehBrand (catégorielle, via embedding).
    """

    def __init__(self, n_continuous, brand_cardinality, embedding_dim=2, hidden_dim=20):
        super().__init__()

        self.brand_embedding = nn.Embedding(brand_cardinality, embedding_dim)

        input_dim = n_continuous + embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, continuous, brand_code, log_mu_glm):
        brand_emb = self.brand_embedding(brand_code)
        x = torch.cat([continuous, brand_emb], dim=1)
        residual = self.mlp(x).squeeze(-1)
        log_lambda = log_mu_glm + residual
        return log_lambda





import copy

def train_group_interaction(model, train_loader, valid_loader, n_epochs, optimizer, device):
    """
    Entraîne un modèle d'interaction de groupe, en sauvegardant les poids
    au meilleur point de validation (le modèle final retourné n'est PAS
    nécessairement celui de la dernière epoch).
    """
    best_valid = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            continuous = batch["continuous"].to(device)
            brand_code = batch["brand_code"].to(device)
            log_mu_glm = batch["log_mu_glm"].to(device)
            claim_nb = batch["claim_nb"].to(device)

            optimizer.zero_grad()
            log_lambda = model(continuous, brand_code, log_mu_glm)
            loss = poisson_deviance_loss_v2(log_lambda, claim_nb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in valid_loader:
                continuous = batch["continuous"].to(device)
                brand_code = batch["brand_code"].to(device)
                log_mu_glm = batch["log_mu_glm"].to(device)
                claim_nb = batch["claim_nb"].to(device)

                log_lambda = model(continuous, brand_code, log_mu_glm)
                loss = poisson_deviance_loss_v2(log_lambda, claim_nb)
                valid_losses.append(loss.item())

        avg_valid = sum(valid_losses) / len(valid_losses)
        if avg_valid < best_valid:
            best_valid = avg_valid
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch+1}/{n_epochs} — valid_loss: {avg_valid:.4f}")

    print(f"\nMeilleur valid_loss : {best_valid:.4f} (epoch {best_epoch})")

    # Recharge les poids du meilleur point avant de retourner le modèle
    model.load_state_dict(best_state)
    return model, best_valid, best_epoch


class GroupDataset(Dataset):
    """Dataset pour tester une interaction entre un groupe de variables.
    
    Optimisé: conversion tensor à la demande (lazy loading) pour accélérer l'initialisation.
    """

    def __init__(self, df, continuous_cols, brand_col):
        self.continuous = df[continuous_cols].values
        self.brand_code = df[brand_col].values
        self.log_mu_glm = df["log_mu_glm"].values
        self.claim_nb = df["ClaimNb"].values

    def __len__(self):
        return len(self.claim_nb)

    def __getitem__(self, idx):
        return {
            "continuous": torch.tensor(self.continuous[idx], dtype=torch.float32),
            "brand_code": torch.tensor(self.brand_code[idx], dtype=torch.long),
            "log_mu_glm": torch.tensor(self.log_mu_glm[idx], dtype=torch.float32),
            "claim_nb": torch.tensor(self.claim_nb[idx], dtype=torch.float32),
        }




def get_shap_input_matrix(model, df, continuous_cols, brand_col, device):
    """
    Construit la matrice d'entrée pour SHAP : variables continues concaténées
    à l'embedding appris de VehBrand (pré-calculé, traité comme features
    numériques fixes puisque l'embedding lookup n'est pas différentiable
    par rapport à l'identifiant catégoriel brut).
    """
    continuous = torch.tensor(df[continuous_cols].values, dtype=torch.float32).to(device)
    brand_code = torch.tensor(df[brand_col].values, dtype=torch.long).to(device)

    with torch.no_grad():
        brand_emb = model.brand_embedding(brand_code)

    X = torch.cat([continuous, brand_emb], dim=1)
    return X.cpu().numpy()


class ResidualMLPWrapper(torch.nn.Module):
    """Isole le MLP résiduel du modèle pour l'explicabilité SHAP (sans le skip GLM)."""
    def __init__(self, model):
        super().__init__()
        self.mlp = model.mlp

    def forward(self, x):
        return self.mlp(x)