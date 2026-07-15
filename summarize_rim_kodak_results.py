#!/usr/bin/env python3
"""Summarize Kodak B-RIM grid logs.

For each (K, block_size, log_hash_size), this reports averages over Kodak
images for:
  - PSNR, preferring exact full-image PSNR when present
  - SSIM from the metrics tiles
  - Estimated active hash parameters + MLP trainable parameters
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RunRecord:
    image: str
    k: int
    block_size: int
    log_hash_size: int
    log_file: Path
    psnr: float | None
    psnr_tile: float | None
    psnr_full: float | None
    ssim: float | None
    ms_ssim: float | None
    active_hash_params: int | None
    mlp_params: int | None

    @property
    def active_plus_mlp(self) -> int | None:
        if self.active_hash_params is None or self.mlp_params is None:
            return None
        return self.active_hash_params + self.mlp_params

    @property
    def complete(self) -> bool:
        return self.psnr is not None and self.ssim is not None and self.active_plus_mlp is not None


def parse_int_text(text: str) -> int:
    return int(text.replace(",", "").strip())


def last_int(pattern: re.Pattern[str], text: str) -> int | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return parse_int_text(matches[-1])


def parse_metric_dicts(text: str) -> list[dict]:
    out: list[dict] = []
    for line in text.replace("\r", "\n").splitlines():
        s = line.strip()
        if not (s.startswith("{") and s.endswith("}")):
            continue
        try:
            obj = ast.literal_eval(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def first_metric(d: dict, keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in d:
            value = as_float(d.get(key))
            if value is not None:
                return value
    return None


def parse_log(
    log_file: Path,
    *,
    image: str,
    k: int,
    block_size: int,
    log_hash_size: int,
    psnr_source: str,
) -> RunRecord:
    text = log_file.read_text(errors="ignore")
    active_hash = last_int(re.compile(r"Estimated active hash parameters:\s*([0-9,]+)"), text)
    mlp = last_int(re.compile(r"MLP trainable parameters:\s*([0-9,]+)"), text)

    psnr_tile = None
    psnr_full = None
    ssim = None
    ms_ssim = None
    for metrics in parse_metric_dicts(text):
        psnr_tile = first_metric(metrics, ("psnr", "PSNR(dB)_mean")) or psnr_tile
        psnr_full = first_metric(metrics, ("psnr_full",)) or psnr_full
        ssim = first_metric(metrics, ("ssim", "SSIM_mean")) or ssim
        ms_ssim = first_metric(metrics, ("ms_ssim", "MS_SSIM_mean", "MS-SSIM_mean")) or ms_ssim

    if psnr_source == "tile":
        psnr = psnr_tile if psnr_tile is not None else psnr_full
    elif psnr_source == "full":
        psnr = psnr_full if psnr_full is not None else psnr_tile
    else:
        psnr = psnr_full if psnr_full is not None else psnr_tile

    return RunRecord(
        image=image,
        k=k,
        block_size=block_size,
        log_hash_size=log_hash_size,
        log_file=log_file,
        psnr=psnr,
        psnr_tile=psnr_tile,
        psnr_full=psnr_full,
        ssim=ssim,
        ms_ssim=ms_ssim,
        active_hash_params=active_hash,
        mlp_params=mlp,
    )


def collect_records(log_dir: Path, run_prefix: str, psnr_source: str) -> list[RunRecord]:
    name_re = re.compile(
        rf"^{re.escape(run_prefix)}_(?P<image>.+?)_k(?P<k>\d+)_b(?P<block>\d+)_log(?P<log>\d+)\.txt$"
    )
    records: list[RunRecord] = []
    for log_file in sorted(log_dir.glob(f"{run_prefix}_*_k*_b*_log*.txt")):
        match = name_re.match(log_file.name)
        if not match:
            continue
        records.append(
            parse_log(
                log_file,
                image=match.group("image"),
                k=int(match.group("k")),
                block_size=int(match.group("block")),
                log_hash_size=int(match.group("log")),
                psnr_source=psnr_source,
            )
        )
    return records


def mean(values: Iterable[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt_float(value: float | None, digits: int) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def fmt_int(value: float | None) -> str:
    return "NA" if value is None else f"{value:,.0f}"


def write_outputs(
    records: list[RunRecord],
    *,
    log_dir: Path,
    run_prefix: str,
    expected_runs: int | None,
    digits: int,
) -> None:
    groups: dict[tuple[int, int, int], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.k, record.block_size, record.log_hash_size), []).append(record)

    if expected_runs is None:
        unique_images = {r.image for r in records}
        expected_runs = len(unique_images) if unique_images else 0

    csv_path = log_dir / f"{run_prefix}_summary.csv"
    md_path = log_dir / f"{run_prefix}_summary.md"
    txt_path = log_dir / f"{run_prefix}_summary.txt"

    fieldnames = [
        "K",
        "block_size",
        "log_hash_size",
        "completed_runs",
        "expected_runs",
        "avg_psnr",
        "avg_tile_psnr",
        "avg_full_psnr",
        "avg_ssim",
        "avg_ms_ssim",
        "avg_active_hash_plus_mlp_params",
        "avg_estimated_active_hash_params",
        "avg_mlp_trainable_params",
    ]

    rows: list[dict[str, object]] = []
    for key in sorted(groups):
        runs = groups[key]
        complete_runs = [r for r in runs if r.complete]
        metric_runs = complete_runs if complete_runs else runs
        rows.append(
            {
                "K": key[0],
                "block_size": key[1],
                "log_hash_size": key[2],
                "completed_runs": len(complete_runs),
                "expected_runs": expected_runs,
                "avg_psnr": mean(r.psnr for r in metric_runs),
                "avg_tile_psnr": mean(r.psnr_tile for r in metric_runs),
                "avg_full_psnr": mean(r.psnr_full for r in metric_runs),
                "avg_ssim": mean(r.ssim for r in metric_runs),
                "avg_ms_ssim": mean(r.ms_ssim for r in metric_runs),
                "avg_active_hash_plus_mlp_params": mean(r.active_plus_mlp for r in metric_runs),
                "avg_estimated_active_hash_params": mean(r.active_hash_params for r in metric_runs),
                "avg_mlp_trainable_params": mean(r.mlp_params for r in metric_runs),
            }
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with md_path.open("w") as f:
        f.write("# Kodak B-RIM Summary\n\n")
        f.write("| K | block | log2 hash | runs | PSNR | SSIM | active hash + MLP |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {K} | {block_size} | {log_hash_size} | {completed_runs}/{expected_runs} | "
                "{psnr} | {ssim} | {params} |\n".format(
                    K=row["K"],
                    block_size=row["block_size"],
                    log_hash_size=row["log_hash_size"],
                    completed_runs=row["completed_runs"],
                    expected_runs=row["expected_runs"],
                    psnr=fmt_float(row["avg_psnr"], digits),
                    ssim=fmt_float(row["avg_ssim"], digits),
                    params=fmt_int(row["avg_active_hash_plus_mlp_params"]),
                )
            )

    table_lines = [
        "Kodak B-RIM Summary",
        "PSNR uses exact full-image PSNR when present; SSIM is averaged from metrics tiles.",
        "",
        "K  block  log2_hash  runs       PSNR       SSIM     active_hash+MLP",
        "-- ------ ---------- -------- ---------- ---------- ----------------",
    ]
    for row in rows:
        table_lines.append(
            f"{row['K']:>1}  "
            f"{row['block_size']:>6}  "
            f"{row['log_hash_size']:>10}  "
            f"{row['completed_runs']:>3}/{row['expected_runs']:<3}  "
            f"{fmt_float(row['avg_psnr'], digits):>10}  "
            f"{fmt_float(row['avg_ssim'], digits):>10}  "
            f"{fmt_int(row['avg_active_hash_plus_mlp_params']):>16}"
        )

    with txt_path.open("w") as f:
        f.write("\n".join(table_lines))
        f.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {txt_path}")
    print()
    print("\n".join(table_lines[3:]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Average Kodak B-RIM logs by setting.")
    parser.add_argument("--log-dir", default="logs/rim_kodak", help="Directory containing raw training logs.")
    parser.add_argument("--run-prefix", default="rim_kodak", help="Run prefix used by run_rim_kodak_grid.sh.")
    parser.add_argument(
        "--psnr-source",
        choices=("full", "tile", "auto"),
        default="full",
        help="Which PSNR to average. full uses psnr_full when available and falls back to tile PSNR.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=None,
        help="Expected image count per setting. Defaults to unique images found in logs.",
    )
    parser.add_argument("--digits", type=int, default=4)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory not found: {log_dir}")

    records = collect_records(log_dir, args.run_prefix, args.psnr_source)
    if not records:
        raise SystemExit(f"No matching logs found in {log_dir} for prefix {args.run_prefix!r}")

    write_outputs(
        records,
        log_dir=log_dir,
        run_prefix=args.run_prefix,
        expected_runs=args.expected_runs,
        digits=int(args.digits),
    )


if __name__ == "__main__":
    main()
