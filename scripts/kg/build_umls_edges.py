"""
Build ontology edge table from UMLS relations.

This script constructs directed edges between ontology concepts
using selected hierarchical relations (e.g., isa, part_of)
from UMLS MRREL.

Output:
  edges.csv with columns:
    - src (source CUI)
    - rel (relation type)
    - dst (target CUI)
"""

import argparse
from pathlib import Path
import csv
import pandas as pd


MRREL_COLS = [
    "CUI1", "AUI1", "STYPE1", "REL",
    "CUI2", "AUI2", "STYPE2", "RELA",
    "RUI", "SRUI", "SAB", "SL", "RG",
    "DIR", "SUPPRESS", "CVF", "_DUMMY"
]


# --------------------------------------------------
def load_top_cuis(path: Path):
    with open(path, "r") as f:
        return set(x.strip() for x in f if x.strip())


# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build ontology edges from UMLS MRREL."
    )
    ap.add_argument(
        "--umls_meta",
        type=str,
        required=True,
        help="Path to UMLS META directory (contains MRREL.RRF)",
    )
    ap.add_argument(
        "--nodes_csv",
        type=str,
        required=True,
        help="Path to nodes.csv (used to restrict graph)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    ap.add_argument(
        "--relations",
        nargs="+",
        default=["isa", "part_of"],
        help="Relation types to keep (RELA field)",
    )
    args = ap.parse_args()

    umls_meta = Path(args.umls_meta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load allowed nodes
    # --------------------------------------------------
    nodes = pd.read_csv(args.nodes_csv)
    if "cui" not in nodes.columns:
        raise KeyError("nodes_csv must contain column 'cui'")
    allowed = set(nodes["cui"].astype(str))

    # --------------------------------------------------
    # Load MRREL
    # --------------------------------------------------
    rel = pd.read_csv(
        umls_meta / "MRREL.RRF",
        sep="|",
        header=None,
        names=MRREL_COLS,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        engine="python",
        on_bad_lines="skip",
    )
    rel = rel.drop(columns=["_DUMMY"])

    # --------------------------------------------------
    # Filter relations
    # --------------------------------------------------
    rel = rel[
        rel["RELA"].isin(args.relations) &
        rel["CUI1"].isin(allowed) &
        rel["CUI2"].isin(allowed) &
        (rel["SUPPRESS"] == "N")
    ][["CUI1", "RELA", "CUI2"]].drop_duplicates()

    edges = rel.rename(
        columns={"CUI1": "src", "RELA": "rel", "CUI2": "dst"}
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    out_path = out_dir / "edges.csv"
    edges.to_csv(out_path, index=False)
    print(f"[DONE] saved ontology edges: {out_path} | n={len(edges)}")


if __name__ == "__main__":
    main()
