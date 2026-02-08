"""
Extract UMLS concept identifiers (CUIs) from clinical reports using QuickUMLS.

Input:
  --reports_parquet : Parquet file with columns [study_id, report_text]

Output:
  --out_parquet     : Parquet file with columns [study_id, concept_ids]

Note:
  - The QuickUMLS index is NOT distributed with this repository.
  - Users must build their own QuickUMLS index from UMLS.
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Extract UMLS CUIs from report text using QuickUMLS."
    )
    ap.add_argument("--reports_parquet", required=True, type=str)
    ap.add_argument(
        "--quickumls_dir",
        required=True,
        type=str,
        help="Path to QuickUMLS index directory",
    )
    ap.add_argument("--out_parquet", required=True, type=str)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--similarity", type=str, default="jaccard")
    ap.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional limit on number of rows (for debugging)",
    )
    args = ap.parse_args()

    from quickumls import QuickUMLS

    df = pd.read_parquet(args.reports_parquet)
    required_cols = ["study_id", "report_text"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(
                f"Input parquet must contain columns {required_cols}. "
                f"Got: {list(df.columns)}"
            )

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    matcher = QuickUMLS(
        args.quickumls_dir,
        threshold=args.threshold,
        similarity_name=args.similarity,
    )

    out_rows = []
    total = len(df)

    for i, row in enumerate(df.itertuples(index=False), 1):
        study_id = str(row.study_id)
        text = row.report_text

        if text is None or (isinstance(text, float) and pd.isna(text)) or str(text).strip() == "":
            out_rows.append(
                {"study_id": study_id, "concept_ids": []}
            )
            continue

        matches = matcher.match(str(text), best_match=True)
        cuis = set()
        for group in matches:
            for m in group:
                cui = m.get("cui")
                if cui is not None:
                    cuis.add(str(cui))

        out_rows.append(
            {
                "study_id": study_id,
                "concept_ids": sorted(cuis),
            }
        )

        if i % 2000 == 0 or i == total:
            print(f"[extract] {i}/{total}")

    out_df = pd.DataFrame(out_rows)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    print(f"[DONE] saved: {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
