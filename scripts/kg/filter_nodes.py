"""
Filter ontology nodes using semantic and lexical constraints.

This script applies rule-based filtering to remove non-clinical,
ambiguous, or unstable ontology concepts. The filtering criteria
include:
  - Allowed semantic types (TUIs)
  - ASCII ratio constraint
  - Blacklist-based exclusion

Output:
  filtered_nodes.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import re


# --------------------------------------------------
def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_chars = sum(ord(c) < 128 for c in text)
    return ascii_chars / len(text)


# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Filter ontology nodes by semantic and lexical rules."
    )
    ap.add_argument(
        "--nodes_csv",
        type=str,
        required=True,
        help="Input nodes.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    ap.add_argument(
        "--allowed_tuis",
        nargs="+",
        required=True,
        help="List of allowed semantic type TUIs",
    )
    ap.add_argument(
        "--min_ascii_ratio",
        type=float,
        default=0.8,
        help="Minimum ASCII character ratio for concept names",
    )
    ap.add_argument(
        "--blacklist_regex",
        nargs="*",
        default=[],
        help="Regex patterns to exclude concept names",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load nodes
    # --------------------------------------------------
    nodes = pd.read_csv(args.nodes_csv)
    if not {"cui", "name", "tui"}.issubset(nodes.columns):
        raise KeyError("nodes_csv must contain [cui, name, tui] columns")

    # --------------------------------------------------
    # Semantic type filtering
    # --------------------------------------------------
    nodes = nodes[nodes["tui"].isin(args.allowed_tuis)]

    # --------------------------------------------------
    # ASCII ratio filtering
    # --------------------------------------------------
    nodes = nodes[
        nodes["name"].apply(lambda x: ascii_ratio(str(x)) >= args.min_ascii_ratio)
    ]

    # --------------------------------------------------
    # Blacklist filtering
    # --------------------------------------------------
    if args.blacklist_regex:
        pattern = re.compile("|".join(args.blacklist_regex), flags=re.IGNORECASE)
        nodes = nodes[~nodes["name"].str.contains(pattern, na=False)]

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    out_path = out_dir / "filtered_nodes.csv"
    nodes.to_csv(out_path, index=False)
    print(f"[DONE] saved filtered ontology nodes: {out_path} | n={len(nodes)}")


if __name__ == "__main__":
    main()
