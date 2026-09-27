#!/usr/bin/env python3
"""Compute the average % diff of bw between reported and run benchmark CSVs.

The bw value is read from the "bw" column, or from "SplitKernel" if the file
has no "bw" column (e.g. fig12_paper.csv).
Rows are matched on (gpu, model, prompt_len, bsz) exactly. If both files have
actual_offloading_ratio, rows must also be within a tolerance on it (default 5,
e.g. 35 and 40 match); each row is matched at most once, closest ratio first.
%diff = (bw_run - bw_reported) / bw_reported * 100:
  + means the run bw is higher than reported, - means it is lower.

Usage: python cal_diff.py reported.csv run.csv [--tol 5]
"""
import argparse
import csv
import sys
from collections import defaultdict

VALUE_COLS = ["bw", "SplitKernel"]
RATIO_COL = "actual_offloading_ratio"


def load(path):
    """Return ({(gpu, model, prompt_len, bsz): [(ratio, bw), ...]}, has_ratio).

    ratio is None when the file has no actual_offloading_ratio column.
    """
    groups = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        val_col = next((c for c in VALUE_COLS if c in reader.fieldnames), None)
        if val_col is None:
            sys.exit(f"{path}: no value column, expected one of {VALUE_COLS}")
        has_ratio = RATIO_COL in reader.fieldnames
        for r in reader:
            key = (r["gpu"], r["model"], int(r["prompt_len"]), int(r["bsz"]))
            ratio = float(r[RATIO_COL]) if has_ratio else None
            groups[key].append((ratio, float(r[val_col])))
    return groups, has_ratio


def match(rows_reported, rows_run, tol, use_ratio):
    """Greedy one-to-one matching of rows whose ratios differ by at most tol."""
    candidates = sorted(
        (abs(r_rep - r_run) if use_ratio else 0.0, i, j)
        for i, (r_rep, _) in enumerate(rows_reported)
        for j, (r_run, _) in enumerate(rows_run)
        if not use_ratio or abs(r_rep - r_run) <= tol
    )
    used_rep, used_run, pairs = set(), set(), []
    for _, i, j in candidates:
        if i not in used_rep and j not in used_run:
            used_rep.add(i)
            used_run.add(j)
            pairs.append((rows_reported[i], rows_run[j]))
    return sorted(pairs, key=lambda p: (p[0][0] or 0.0, p[1][0] or 0.0))


def fmt_ratio(ratio):
    return f"{ratio:>11.2f}" if ratio is not None else f"{'-':>11}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reported", help="reported CSV (e.g. fig10_paper.csv)")
    parser.add_argument("run", help="CSV from your own run (e.g. fig10_ref.csv)")
    parser.add_argument("--tol", type=float, default=5.0,
                        help="max actual_offloading_ratio difference to treat rows as the same (default: 5)")
    args = parser.parse_args()

    reported, rep_has_ratio = load(args.reported)
    run, run_has_ratio = load(args.run)
    use_ratio = rep_has_ratio and run_has_ratio

    print(f"{'gpu':<10}{'model':<12}{'prompt_len':>11}{'bsz':>6}{'ratio_rep':>11}{'ratio_run':>11}"
          f"{'bw_rep':>11}{'bw_run':>11}{'%diff':>9}")
    diffs = []
    for key in sorted(reported.keys() & run.keys()):
        gpu, model, prompt_len, bsz = key
        for (r_rep, bw_rep), (r_run, bw_run) in match(reported[key], run[key], args.tol, use_ratio):
            d = (bw_run - bw_rep) / bw_rep * 100
            diffs.append(d)
            print(f"{gpu:<10}{model:<12}{prompt_len:>11}{bsz:>6}{fmt_ratio(r_rep)}{fmt_ratio(r_run)}"
                  f"{bw_rep:>11.2f}{bw_run:>11.2f}{d:>+9.2f}")

    if not diffs:
        sys.exit("no matching rows between the two CSVs")

    print()
    if use_ratio:
        print(f"matched on gpu, model, prompt_len, bsz and {RATIO_COL} (tol {args.tol:g})")
    else:
        print(f"matched on gpu, model, prompt_len, bsz ({RATIO_COL} not in both files)")
    print(f"matched rows: {len(diffs)}")
    print("%diff = (run - reported) / reported; + = run higher than reported, - = run lower")
    print(f"average %diff:   {sum(diffs) / len(diffs):+.2f}%")
    print(f"average |%diff|: {sum(abs(d) for d in diffs) / len(diffs):.2f}%")


if __name__ == "__main__":
    main()
