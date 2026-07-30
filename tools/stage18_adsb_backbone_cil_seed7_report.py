"""汇总 Stage 18 ADS-B 重训 discovery backbone 后的 seed7 完整增量结果。

本报告用于回答一个很直接的问题：Stage 18 的表征重训虽然提升了 GPCC 聚类，
但这种聚类收益能不能真正传到 Long-RADCIL 后端。报告只读取单次实验生成的
CSV 小结果，不读取 checkpoint、replay memory 或 held-out eval 以外的信息。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


BASELINE_R3 = {
    # 当前 seed7 上可对照的较强结构结果：target split / cross-day GPCC 系列的
    # seed7 最佳附近表现。Stage 18 若只是 discovery-only 好看但 CIL 不超过它，
    # 就不能继续扩三种子。
    "name": "current_seed7_best",
    "Overall Acc": 0.5057,
    "Old Acc": 0.4850,
    "New Acc": 0.5583,
}

STATIC_GPCC_R3 = {
    # Stage 6 GPCC + Long-RADCIL seed7 完整增量结果，用于区分“换前端”与
    # “换 discovery backbone”带来的收益。
    "Overall Acc": 0.4839,
    "Old Acc": float("nan"),
    "New Acc": float("nan"),
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 并确保结果非空。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _last_row(path: Path) -> dict[str, str]:
    """读取 incremental_results.csv 的最后一行，也就是 After R3 指标。"""

    return _read_rows(path)[-1]


def _number(row: dict[str, str], key: str) -> float:
    """安全读取浮点指标，缺失时返回 NaN。"""

    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    """统一四位小数输出，便于和前序阶段报告对齐。"""

    return "nan" if math.isnan(value) else f"{value:.4f}"


def _collect_rounds(save_dir: Path) -> list[dict[str, float | str]]:
    """合并三轮 clustering 与 incremental 指标，保留前端和后端的对应关系。"""

    clustering_rows = _read_rows(save_dir / "clustering_results.csv")
    incremental_rows = _read_rows(save_dir / "incremental_results.csv")
    inc_by_stage = {str(row.get("Stage", "")): row for row in incremental_rows}

    rounds: list[dict[str, float | str]] = []
    for row in clustering_rows:
        round_name = str(row.get("Round", ""))
        inc = inc_by_stage.get(f"After {round_name}", {})
        rounds.append(
            {
                "round": round_name,
                "clusters": _number(row, "Final Cluster Count"),
                "ari": _number(row, "ARI"),
                "hungarian": _number(row, "Hungarian Acc"),
                "purity": _number(row, "Purity"),
                "overall": _number(inc, "Overall Acc"),
                "old": _number(inc, "Old Acc"),
                "new": _number(inc, "New Acc"),
                "forgetting": _number(inc, "Forgetting Rate"),
                "macro_f1": _number(inc, "Macro F1"),
            }
        )
    return rounds


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B Stage 18 backbone CIL seed7 结果。")
    parser.add_argument("--save-dir", required=True, help="完整 CIL 输出目录。")
    parser.add_argument("--output", required=True, help="Markdown 报告输出路径。")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    rounds = _collect_rounds(save_dir)
    r3 = next((row for row in rounds if row["round"] == "R3"), rounds[-1])

    overall = float(r3["overall"])
    old = float(r3["old"])
    new = float(r3["new"])
    passed = (
        overall > BASELINE_R3["Overall Acc"]
        and old >= BASELINE_R3["Old Acc"] - 0.010
        and new >= BASELINE_R3["New Acc"] - 0.030
    )

    lines = [
        "# Stage 18 ADS-B Backbone 重训 + GPCC + Long-RADCIL seed7 报告",
        "",
        "本报告验证 discovery backbone 重训带来的聚类收益是否能传到完整增量后端。",
        "",
        (
            "seed7 当前强对照："
            f"Overall={BASELINE_R3['Overall Acc']:.4f}, "
            f"Old={BASELINE_R3['Old Acc']:.4f}, New={BASELINE_R3['New Acc']:.4f}。"
        ),
        f"Stage 6 static GPCC 完整 CIL R3 Overall={STATIC_GPCC_R3['Overall Acc']:.4f}。",
        "",
        "| Round | K | ARI | Hungarian | Purity | Overall | Old | New | Forgetting | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rounds:
        lines.append(
            f"| {row['round']} | {_fmt(float(row['clusters']))} | {_fmt(float(row['ari']))} | "
            f"{_fmt(float(row['hungarian']))} | {_fmt(float(row['purity']))} | "
            f"{_fmt(float(row['overall']))} | {_fmt(float(row['old']))} | "
            f"{_fmt(float(row['new']))} | {_fmt(float(row['forgetting']))} | "
            f"{_fmt(float(row['macro_f1']))} |"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            (
                "通过：R3 Overall 超过当前 seed7 强对照，且 Old/New 没有明显塌缩；可进入 ADS-B 三种子确认。"
                if passed
                else "未通过：聚类提升没有稳定转化为完整 CIL 收益，先不扩三种子。"
            ),
            "",
            f"R3 delta Overall={overall - BASELINE_R3['Overall Acc']:+.4f}, "
            f"Old={old - BASELINE_R3['Old Acc']:+.4f}, "
            f"New={new - BASELINE_R3['New Acc']:+.4f}。",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
