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
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def load_kg_index(kg_ids_csv: Path):
    df = pd.read_csv(kg_ids_csv)
    for col in ["concept_id", "cui", "node_id", "id"]:
        if col in df.columns:
            ids = df[col].astype(str).tolist()
            return {cid: i for i, cid in enumerate(ids)}, ids
    raise KeyError(f"KG id column not found in {kg_ids_csv}")


def load_proto_head(model_pt: Path):
    state = torch.load(model_pt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    w_key = next(k for k in state if k.endswith("proto_head.weight"))
    b_key = next(k for k in state if k.endswith("proto_head.bias"))

    W = state[w_key].cpu().numpy().astype(np.float32)
    b = state[b_key].cpu().numpy().astype(np.float32)
    return W, b


def apply_topk_mask(z: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(z, -k, axis=1)[:, -k:]
    mask = np.zeros_like(z, dtype=np.float32)
    rows = np.arange(z.shape[0])[:, None]
    mask[rows, idx] = 1.0
    return z * mask


# --------------------------------------------------
# QuickUMLS
# --------------------------------------------------
def build_quickumls(index_dir: str, threshold: float, similarity: str):
    from quickumls import QuickUMLS
    return QuickUMLS(index_dir, threshold=threshold, similarity_name=similarity)


def extract_cuis(matcher, text: str):
    if not isinstance(text, str) or text.strip() == "":
        return []
    matches = matcher.match(text, best_match=True, ignore_syntax=False)
    seen, cuis = set(), []
    for group in matches:
        for m in group:
            cui = m.get("cui")
            if cui and cui not in seen:
                seen.add(cui)
                cuis.append(cui)
    return cuis


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
    ap.add_argument("--use_topk_mask", action="store_true")
    ap.add_argument("--topk_mask", type=int, default=32)

    ap.add_argument("--quickumls_index", required=True)
    ap.add_argument("--quickumls_threshold", type=float, default=0.7)
    ap.add_argument("--quickumls_similarity", type=str, default="jaccard")

    ap.add_argument("--score_mode", choices=["linear", "relu"], default="relu")
    ap.add_argument("--tau", type=float, default=0.2)

    ap.add_argument("--max_n_pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)

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
    kg_vecs = np.load(args.kg_vecs_npy, mmap_mode="r").astype(np.float32)
    kg_vecs_norm = l2_normalize(np.asarray(kg_vecs))

    W, b = load_proto_head(Path(args.model_pt))
    matcher = build_quickumls(
        args.quickumls_index,
        args.quickumls_threshold,
        args.quickumls_similarity,
    )

    df = pd.read_parquet(args.step5_parquet)
    df = df.sort_values(["pair_id", "y"]).reset_index(drop=True)

    ys, scores = [], []

    for _, row in df.iterrows():
        sid = row["study_id"]
        y = int(row["y"])
        term = str(row.get("insert_term", "")).strip()

        if sid not in study_to_idx:
            continue

        z_i = np.asarray(z[study_to_idx[sid]])[None, :]

        if term == "":
            ys.append(y)
            scores.append(0.0)
            continue

        cuis = extract_cuis(matcher, term)
        rows = [concept_to_row[c] for c in cuis if c in concept_to_row]
        if not rows:
            continue

        if args.use_topk_mask:
            z_i = apply_topk_mask(z_i, args.topk_mask)

        proto = l2_normalize(z_i @ W.T + b[None, :])
        sim = proto @ kg_vecs_norm.T
        s_img = float(sim[0, rows].max())

        score = 1.0 - s_img if args.score_mode == "linear" else max(0.0, args.tau - s_img)
        ys.append(y)
        scores.append(score)

    ys = np.asarray(ys)
    scores = np.asarray(scores)

    auc = roc_auc_score(ys, scores) if len(np.unique(ys)) > 1 else None
    out = {"auc_insertion": auc, "num_samples": int(len(ys))}

    with open(out_dir / "insertion_auc.json", "w") as f:
        json.dump(out, f, indent=2)

    if auc is not None:
        fpr, tpr, _ = roc_curve(ys, scores)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Insertion Hallucination ROC")
        plt.savefig(out_dir / "roc_insertion.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
