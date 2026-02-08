"""
Train Ontology-Guided Representation Geometry (OGRE).

OGRE aligns image representations with ontology prototype vectors
by combining reconstruction loss and ontology alignment loss.
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
class OGRE(nn.Module):
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
def train_epoch(model, loader, optimizer, device, alpha):
    model.train()
    total_loss = 0.0

    for x, proto in loader:
        x = x.to(device)
        proto = proto.to(device)

        optimizer.zero_grad()
        x_hat, h = model(x)

        recon_loss = torch.mean((x - x_hat) ** 2)
        align_loss = torch.mean((h - proto) ** 2)
        loss = recon_loss + alpha * align_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Train Ontology-Guided Representation Geometry model (OGRE)."
    )
    ap.add_argument("--z_path", required=True, type=str)
    ap.add_argument("--proto_path", required=True, type=str)
    ap.add_argument("--N", required=True, type=int)
    ap.add_argument("--z_dim", required=True, type=int)

    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0)

    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Load embeddings and prototypes
    # --------------------------------------------------
    z = np.memmap(
        args.z_path,
        dtype=np.float32,
        mode="r",
        shape=(args.N, args.z_dim),
    )
    proto = np.memmap(
        args.proto_path,
        dtype=np.float32,
        mode="r",
        shape=(args.N, args.z_dim),
    )

    z_tensor = torch.from_numpy(np.asarray(z))
    proto_tensor = torch.from_numpy(np.asarray(proto))

    dataset = TensorDataset(z_tensor, proto_tensor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = OGRE(args.z_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, loader, optimizer, device, args.alpha
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
        out_dir / "OGRE.pt",
    )

    with open(out_dir / "train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[DONE] saved OGRE model to {out_dir}")


if __name__ == "__main__":
    main()
