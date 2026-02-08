import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def load_kg_index(kg_ids_csv: Path):
    df = pd.read_csv(kg_ids_csv)
    for col in ["concept_id", "cui", "node_id", "id"]:
        if col in df.columns:
            ids = df[col].astype(str).tolist()
            return {cid: i for i, cid in enumerate(ids)}, ids
    raise KeyError("KG id column not found")


def load_proto_head(model_pt: Path):
    state = torch.load(model_pt, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    W = next(v for k, v in state.items() if k.endswith("proto_head.weight"))
    b = next(v for k, v in state.items() if k.endswith("proto_head.bias"))
    return W.numpy(), b.numpy()


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def conflict_score(text_rows, img_rows, kg_vecs_norm):
    T = kg_vecs_norm[text_rows]
    I = kg_vecs_norm[img_rows]
    sim = T @ I.T
    return float((1.0 - sim.max(axis=1)).mean())


# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step5_parquet", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--feat_ids_csv", required=True)
    ap.add_argument("--row_idx_path", required=True)
    ap.add_argument("--z_path", required=True)
    ap.add_argument("--N", type=int, default=48579)
    ap.add_argument("--z_dim", type=int, default=2048)

    ap.add_argument("--kg_ids_csv", required=True)
    ap.add_argument("--kg_vecs_npy", required=True)

    ap.add_argument("--model_pt", required=True)
    ap.add_argument("--img_topM", type=int, default=50)

    ap.add_argument("--quickumls_index", required=True)
    ap.add_argument("--quickumls_threshold", type=float, default=0.7)
    ap.add_argument("--quickumls_similarity", default="jaccard")

    ap.add_argument("--min_text_kg", type=int, default=5)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = pd.read_csv(args.feat_ids_csv)
    ids["study_id"] = ids["study_id"].astype(str)

    row_idx = np.memmap(args.row_idx_path, dtype=np.int64, mode="r", shape=(args.N,))
    z = np.memmap(args.z_path, dtype=np.float32, mode="r", shape=(args.N, args.z_dim))

    study_to_idx = {}
    for i in range(args.N):
        sid = str(ids.iloc[int(row_idx[i])]["study_id"])
        study_to_idx.setdefault(sid, i)

    concept_to_row, _ = load_kg_index(Path(args.kg_ids_csv))
    kg_vecs = np.load(args.kg_vecs_npy).astype(np.float32)
    kg_vecs_norm = l2_normalize(kg_vecs)

    W, b = load_proto_head(Path(args.model_pt))

    from quickumls import QuickUMLS
    matcher = QuickUMLS(
        args.quickumls_index,
        threshold=args.quickumls_threshold,
        similarity_name=args.quickumls_similarity,
    )

    df = pd.read_parquet(args.step5_parquet)
    df = df.sort_values(["pair_id", "y"]).reset_index(drop=True)

    ys, scores = [], []

    for _, row in df.iterrows():
        sid = row["study_id"]
        if sid not in study_to_idx:
            continue

        text = row["text"]
        y = int(row["y"])

        cuis = [m["cui"] for g in matcher.match(text, best_match=True) for m in g]
        text_rows = np.array([concept_to_row[c] for c in cuis if c in concept_to_row])
        if len(text_rows) < args.min_text_kg:
            continue

        z_i = np.asarray(z[study_to_idx[sid]])[None, :]
        proto = l2_normalize(z_i @ W.T + b[None, :])[0]
        sim = proto @ kg_vecs_norm.T
        img_rows = topk_indices(sim, args.img_topM)

        score = conflict_score(text_rows, img_rows, kg_vecs_norm)
        ys.append(y)
        scores.append(score)

    ys = np.asarray(ys)
    scores = np.asarray(scores)

    auc = roc_auc_score(ys, scores) if len(np.unique(ys)) > 1 else None
    out = {"auc_conflict": auc, "num_samples": int(len(ys))}

    with open(out_dir / "conflict_auc.json", "w") as f:
        json.dump(out, f, indent=2)

    if auc is not None:
        fpr, tpr, _ = roc_curve(ys, scores)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Ontology Conflict ROC")
        plt.savefig(out_dir / "roc_conflict.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
