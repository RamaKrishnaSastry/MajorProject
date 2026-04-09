"""Compare uncalibrated vs calibrated QWK for Stage 1/2 evaluation outputs.

Reads pipeline_outputs/eval_stage*/comprehensive_metrics.json and prints a compact
table with raw and final QWK values for DME and DR.
"""

import argparse
import json
import os
from typing import Dict, List, Optional


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_stage_summary(stage_name: str, metrics: Dict) -> Dict:
    dme_final = _safe_float(metrics.get("qwk"))

    dme_cal = metrics.get("calibration", {}).get("dme", {})
    dme_raw = _safe_float(dme_cal.get("baseline_qwk"), dme_final)
    dme_gain = dme_final - dme_raw
    dme_applied = bool(dme_cal.get("applied", False))

    dr_block = metrics.get("dr", {}) if isinstance(metrics.get("dr", {}), dict) else {}
    dr_final = _safe_float(dr_block.get("dr_qwk"))

    dr_cal = dr_block.get("calibration", {}) if isinstance(dr_block.get("calibration", {}), dict) else {}
    dr_raw = _safe_float(dr_cal.get("baseline_qwk"), dr_final)
    dr_gain = dr_final - dr_raw
    dr_applied = bool(dr_cal.get("applied", False))

    return {
        "stage": stage_name,
        "dme_raw": dme_raw,
        "dme_final": dme_final,
        "dme_gain": dme_gain,
        "dme_applied": dme_applied,
        "dr_raw": dr_raw,
        "dr_final": dr_final,
        "dr_gain": dr_gain,
        "dr_applied": dr_applied,
    }


def _read_stage_metrics(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_num(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.4f}"


def _print_table(rows: List[Dict]) -> None:
    headers = [
        "stage",
        "dme_raw",
        "dme_final",
        "dme_gain",
        "dme_vs_s1",
        "dme_cal",
        "dr_raw",
        "dr_final",
        "dr_gain",
        "dr_vs_s1",
        "dr_cal",
    ]

    table = []
    for row in rows:
        table.append(
            [
                row["stage"],
                _fmt_num(row["dme_raw"]),
                _fmt_num(row["dme_final"]),
                _fmt_num(row["dme_gain"]),
                _fmt_num(row.get("dme_vs_s1", float("nan"))),
                "yes" if row["dme_applied"] else "no",
                _fmt_num(row["dr_raw"]),
                _fmt_num(row["dr_final"]),
                _fmt_num(row["dr_gain"]),
                _fmt_num(row.get("dr_vs_s1", float("nan"))),
                "yes" if row["dr_applied"] else "no",
            ]
        )

    widths = [len(h) for h in headers]
    for r in table:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def _join(parts: List[str]) -> str:
        return " | ".join(parts)

    print(_join([headers[i].ljust(widths[i]) for i in range(len(headers))]))
    print(_join(["-" * widths[i] for i in range(len(headers))]))
    for r in table:
        print(_join([r[i].ljust(widths[i]) for i in range(len(r))]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare calibrated vs uncalibrated QWK by stage.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pipeline_outputs",
        help="Pipeline output directory containing eval_stage*/comprehensive_metrics.json",
    )
    args = parser.parse_args()

    stage_paths = {
        "stage1": os.path.join(args.output_dir, "eval_stage1", "comprehensive_metrics.json"),
        "stage2": os.path.join(args.output_dir, "eval_stage2", "comprehensive_metrics.json"),
    }

    rows = []
    for stage_name, metrics_path in stage_paths.items():
        metrics = _read_stage_metrics(metrics_path)
        if metrics is None:
            continue
        rows.append(_extract_stage_summary(stage_name, metrics))

    rows.sort(key=lambda r: r.get("stage", ""))

    stage1_row = next((r for r in rows if r.get("stage") == "stage1"), None)
    stage1_dme = _safe_float(stage1_row.get("dme_final")) if stage1_row else float("nan")
    stage1_dr = _safe_float(stage1_row.get("dr_final")) if stage1_row else float("nan")

    for row in rows:
        row["dme_vs_s1"] = row["dme_final"] - stage1_dme
        row["dr_vs_s1"] = row["dr_final"] - stage1_dr

    if not rows:
        print("No stage metrics found. Expected files:")
        for stage_name, metrics_path in stage_paths.items():
            print(f"- {stage_name}: {metrics_path}")
        return

    print("Calibration comparison (raw vs final QWK):")
    _print_table(rows)


if __name__ == "__main__":
    main()
