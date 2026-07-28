"""生成阶段 5 ManyTx / ManyRx 补充稳定性验证报告。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage2_wisig_main_report import extract_round_metrics, fmt


def _read_csv(path: Path) -> pd.DataFrame:
    """读取补充结果 CSV，并在缺文件时给出明确错误，避免报告静默缺项。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少补充验证结果文件：{path}")
    return pd.read_csv(path)


def _row_to_public_dict(row: pd.Series) -> dict[str, Any]:
    """把 pandas 行转换成 JSON 友好的字典，同时保留空值为 None。"""
    result: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            result[key] = None
        elif hasattr(value, "item"):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def summarize_manytx(save_dir: Path) -> dict[str, Any]:
    """汇总 ManyTx seed7 CIL 三轮结果，用于确认补充划分可完整闭环。"""
    metrics = extract_round_metrics(save_dir)
    clustering = _read_csv(save_dir / "clustering_results.csv")
    return {
        "dataset": "ManyTx RX2",
        "role": "WiSig 补充发射机划分",
        "source_dir": save_dir.as_posix(),
        "protocol": "10 known + 10x3 unknown rounds, fixed RX2, seed7.",
        "method": "MV-ACC-CIL",
        "rounds": metrics["rounds"],
        "discovery": [_row_to_public_dict(row) for _, row in clustering.iterrows()],
    }


def summarize_manyrx(save_dir: Path) -> dict[str, Any]:
    """汇总 ManyRx 固定跨接收机协议结果，保留 MV-ACC 与视图消融对照。"""
    per_round = _read_csv(save_dir / "per_round_summary_results.csv")
    clustering = _read_csv(save_dir / "clustering_results.csv")
    mvacc_rows = per_round[per_round["Method"] == "MV-ACC"].copy()
    if mvacc_rows.empty:
        raise ValueError(f"ManyRx 结果中未找到 MV-ACC 方法行：{save_dir}")
    return {
        "dataset": "ManyRx fixed cross-RX",
        "role": "WiSig 补充接收机划分",
        "source_dir": save_dir.as_posix(),
        "protocol": "manyrx_protocol_manifest.json, 4 known + 2x3 unknown rounds, seed7.",
        "method": "MV-ACC",
        "rounds": [_row_to_public_dict(row) for _, row in mvacc_rows.iterrows()],
        "all_methods": [_row_to_public_dict(row) for _, row in per_round.iterrows()],
        "discovery": [_row_to_public_dict(row) for _, row in clustering[clustering["Method"] == "MV-ACC"].iterrows()],
    }


def _round_value(row: dict[str, Any], key: str) -> Any:
    """兼容统一小写键和原始 CSV 标题键，便于渲染两个来源的表格。"""
    snake_to_csv = {
        "round": "Round",
        "cluster_count": "Cluster Count",
        "nmi": "NMI",
        "ari": "ARI",
        "purity": "Purity",
        "hungarian_acc": "Hungarian Acc",
        "overall_acc": "Overall Acc",
        "new_acc": "New Acc",
        "forgetting_rate": "Forgetting Rate",
        "macro_f1": "Macro F1",
    }
    return row.get(key, row.get(snake_to_csv.get(key, key)))


def _round_table(dataset: dict[str, Any]) -> list[str]:
    """生成统一指标表；ManyRx 历史结果没有 Old/Macro F1，因此仅展示共有字段。"""
    lines = [
        f"### {dataset['dataset']}",
        "",
        f"- 结果目录：`{dataset['source_dir']}`",
        f"- 协议：{dataset['protocol']}",
        f"- 主方法：{dataset['method']}",
        "",
        "| 轮次 | Cluster Count | NMI | ARI | Purity | Hungarian Acc | Overall | New | Forgetting |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset["rounds"]:
        lines.append(
            "| {round} | {cluster_count} | {nmi} | {ari} | {purity} | {hungarian_acc} | {overall_acc} | {new_acc} | {forgetting_rate} |".format(
                round=_round_value(row, "round"),
                cluster_count=fmt(_round_value(row, "cluster_count")),
                nmi=fmt(_round_value(row, "nmi")),
                ari=fmt(_round_value(row, "ari")),
                purity=fmt(_round_value(row, "purity")),
                hungarian_acc=fmt(_round_value(row, "hungarian_acc")),
                overall_acc=fmt(_round_value(row, "overall_acc")),
                new_acc=fmt(_round_value(row, "new_acc")),
                forgetting_rate=fmt(_round_value(row, "forgetting_rate")),
            )
        )
    return lines


def build_markdown(summary: dict[str, Any]) -> str:
    """构造面向阶段验收的 Markdown 报告，突出是否完成补充验证。"""
    manytx_r3 = next(row for row in summary["manytx"]["rounds"] if _round_value(row, "round") == "R3")
    manyrx_r3 = next(row for row in summary["manyrx"]["rounds"] if _round_value(row, "round") == "R3")
    lines = [
        "# 阶段 5 ManyTx / ManyRx 补充稳定性验证报告",
        "",
        f"- 生成时间 UTC：{summary['generated_at_utc']}",
        "- 性质：只读汇总既有 seed7 结果，不重新训练、不读取 held-out 真值调参。",
        "- 结论：ManyTx 与 ManyRx 均已有三轮补充链路结果，可关闭阶段 5 的补充稳定性验证待办；二者仅作为辅助证据，不替代 WiSig/ADS-B 主实验。",
        "",
    ]
    lines.extend(_round_table(summary["manytx"]))
    lines.extend([""])
    lines.extend(_round_table(summary["manyrx"]))
    lines.extend(
        [
            "",
            "## 阶段判定",
            "",
            f"- ManyTx R3：Overall `{fmt(_round_value(manytx_r3, 'overall_acc'))}`，New `{fmt(_round_value(manytx_r3, 'new_acc'))}`，Forgetting `{fmt(_round_value(manytx_r3, 'forgetting_rate'))}`，发现簇数误差 `{fmt(_round_value(manytx_r3, 'cluster_count') - 10)}`。",
            f"- ManyRx R3：Overall `{fmt(_round_value(manyrx_r3, 'overall_acc'))}`，New `{fmt(_round_value(manyrx_r3, 'new_acc'))}`，Forgetting `{fmt(_round_value(manyrx_r3, 'forgetting_rate'))}`，发现簇数误差 `{fmt(_round_value(manyrx_r3, 'cluster_count') - 2)}`。",
            "- ManyTx 发射机补充划分的发现质量稳定，但增量识别 Overall 偏低；ManyRx 接收机补充划分 R3 Overall 较好但遗忘仍高，说明跨接收机旧类保持仍是风险。",
            "- 当前阶段 5 的目标是补充稳定性验证而非继续调参；因此不再因 ManyTx/ManyRx 扩展实验阻塞阶段 6 交付整理。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ManyTx / ManyRx 阶段 5 补充验证报告。")
    parser.add_argument("--manytx-dir", default="results/manytx_rx2_mvacc_cil_v2_seed7")
    parser.add_argument("--manyrx-dir", default="results/manyrx_fixedrx_trainingv2_seed7")
    parser.add_argument("--output", default="results/stage5/STAGE5_MANYTX_MANYRX_SUPPLEMENT_REPORT.md")
    parser.add_argument("--summary-json", default="results/stage5/stage5_manytx_manyrx_supplement_summary.json")
    args = parser.parse_args()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manytx": summarize_manytx(Path(args.manytx_dir)),
        "manyrx": summarize_manyrx(Path(args.manyrx_dir)),
    }

    output = Path(args.output)
    summary_json = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(summary), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ManyTx/ManyRx 补充验证报告：{output}")
    print(f"ManyTx/ManyRx 补充验证摘要：{summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
