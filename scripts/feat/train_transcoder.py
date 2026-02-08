"""
Train a Sparse Transcoder on image embeddings.

The Transcoder learns a sparse latent representation z_hat
from image embeddings z by minimizing reconstruction error
with sparsity regularization.

This module corresponds to the first representation learning
stage in OGRE.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# Model
# --------------------------------------------------
class Transcoder(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        x_hat = self.decoder(h)
        return x_hat, h


# --------------------------------------------------
# Training
# --------------------------------------------------
def train_epoch(model, loader, optimizer, device, l1_weight):
    model.train()
    total_loss = 0.0

    for x in loader:
        x = x[0].to(device)

        optimizer.zero_grad()
        x_hat, h = model(x)

        recon_loss = torch.mean((x - x_hat) ** 2)
        sparsity_loss = torch.mean(torch.abs(h))
        loss = recon_loss + l1_weight * sparsity_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Train sparse Transcoder on image embeddings."
    )
    ap.add_argument("--z_path", required=True, type=str)
    ap.add_argument("--N", required=True, type=int)
    ap.add_argument("--z_dim", required=True, type=int)

    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1_weight", type=float, default=1e-3)

    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Load embeddings
    # --------------------------------------------------
    z = np.memmap(
        args.z_path,
        dtype=np.float32,
        mode="r",
        shape=(args.N, args.z_dim),
    )
    z_tensor = torch.from_numpy(np.asarray(z))
    dataset = TensorDataset(z_tensor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = Transcoder(args.z_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, loader, optimizer, device, args.l1_weight
        )
        history.append({"epoch": epoch, "loss": loss})
        print(f"[epoch {epoch:03d}] loss={loss:.6f}")

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        out_dir / "transcoder.pt",
    )

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[DONE] saved model to {out_dir}")


if __name__ == "__main__":
    main()
