"""
Build ontology node table from UMLS.

This script constructs ontology nodes by selecting preferred concept
names and semantic types from UMLS release files.

Output:
  nodes.csv with columns:
    - cui
    - name
    - tui
"""

import argparse
from pathlib import Path
import csv
import pandas as pd


# --------------------------------------------------
def load_mrconso(path: Path) -> pd.DataFrame:
    cols = [
        "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI",
        "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR",
        "SRL", "SUPPRESS", "CVF"
    ]
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=cols,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        engine="python",
        on_bad_lines="skip",
    )
    return df


def load_mrsty(path: Path) -> pd.DataFrame:
    cols = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=cols,
        dtype=str,
        quoting=csv.QUOTE_NONE,
        engine="python",
        on_bad_lines="skip",
    )
    return df


# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build ontology nodes from UMLS."
    )
    ap.add_argument(
        "--umls_meta",
        type=str,
        required=True,
        help="Path to UMLS META directory (contains MRCONSO.RRF, MRSTY.RRF)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    args = ap.parse_args()

    umls_meta = Path(args.umls_meta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load UMLS tables
    # --------------------------------------------------
    conso = load_mrconso(umls_meta / "MRCONSO.RRF")
    sty = load_mrsty(umls_meta / "MRSTY.RRF")

    # --------------------------------------------------
    # Select preferred English names
    # --------------------------------------------------
    conso = conso[
        (conso["LAT"] == "ENG") &
        (conso["ISPREF"] == "Y") &
        (conso["SUPPRESS"] == "N")
    ][["CUI", "STR"]].drop_duplicates()

    conso = conso.rename(columns={"STR": "name"})

    # --------------------------------------------------
    # Join semantic types
    # --------------------------------------------------
    nodes = conso.merge(
        sty[["CUI", "TUI"]],
        on="CUI",
        how="left",
    ).drop_duplicates()

    nodes = nodes.rename(columns={"CUI": "cui", "TUI": "tui"})

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    out_path = out_dir / "nodes.csv"
    nodes.to_csv(out_path, index=False)
    print(f"[DONE] saved ontology nodes: {out_path} | n={len(nodes)}")


if __name__ == "__main__":
    main()
