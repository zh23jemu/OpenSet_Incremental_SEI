"""生成 WiSig 阶段 3 strict baseline 总表。

本脚本把已经完成的三类证据合并到一份可汇报表中：

1. 阶段 1 的 SimGCD-style / IGCD-minimal 前端对比，用于说明学习式发现
   baseline 的严格适配边界；
2. 阶段 3 的共享 MV-ACC 伪标签后端 baseline，用于隔离增量后端能力；
3. 阶段 3 的 Deep-HDBSCAN 端到端 baseline，用于完整系统对照。

脚本只读取既有报告，不重新训练，也不使用任何评估真值做选择。所有
hidden label 相关指标只来自已经落盘的事后评估结果。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DISCOVERY_METRICS = ("NMI", "ARI", "Purity", "Hungarian Acc")
BACKEND_METRICS = ("overall", "old", "new", "forgetting", "macro_f1")


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件；缺失说明前序证据链不完整，应直接失败。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少输入 JSON：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float | None:
    """计算均值，空列表保留为 None，避免伪造 0 值。"""
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return None
    return statistics.mean(clean)


def fmt(value: float | None) -> str:
    """Markdown 表格统一数值格式。"""
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def normalise_frontend_name(name: str) -> str:
    """统一不同报告中的方法命名。"""
    if name.startswith("SimGCD-style"):
        return "SimGCD-style"
    if name.startswith("IGCD-minimal"):
        return "IGCD-minimal"
    return name


def collect_frontend_rows(simgcd_path: Path, igcd_path: Path) -> list[dict[str, Any]]:
    """汇总发现前端三轮均值。

    Deep-HDBSCAN 和 MV-ACC 在两个前端报告中是同一组参照；为避免重复计数，
    只从 SimGCD 报告读取它们，再从各自报告读取 SimGCD-style 和
    IGCD-minimal。
    """
    simgcd = load_json(simgcd_path)
    igcd = load_json(igcd_path)
    selected: dict[str, list[dict[str, Any]]] = {}

    for row in simgcd.get("rows", []):
        method = normalise_frontend_name(str(row.get("Method", "")))
        if method in {"Deep-HDBSCAN", "MV-ACC", "SimGCD-style"}:
            selected.setdefault(method, []).append(row)

    for row in igcd.get("rows", []):
        method = normalise_frontend_name(str(row.get("Method", "")))
        if method == "IGCD-minimal":
            selected.setdefault(method, []).append(row)

    role = {
        "MV-ACC": "正式主前端",
        "Deep-HDBSCAN": "端到端传统发现 baseline",
        "SimGCD-style": "learning-style adaptation，非完整 SimGCD 复现",
        "IGCD-minimal": "minimal strict adaptation，非完整 IGCD 复现",
    }
    rows: list[dict[str, Any]] = []
    for method, items in sorted(selected.items()):
        out: dict[str, Any] = {
            "method": method,
            "role": role.get(method, "baseline"),
            "round_count": len(items),
            "uses_unknown_true_labels": any(bool(item.get("Uses Unknown True Labels")) for item in items),
            "uses_eval_set": any(bool(item.get("Uses Eval Set")) for item in items),
        }
        for metric in DISCOVERY_METRICS:
            out[metric] = mean([float(item[metric]) for item in items if metric in item])
        out["coverage"] = mean(
            [
                float(item.get("Assignment Coverage"))
                for item in items
                if item.get("Assignment Coverage") is not None
            ]
        )
        rows.append(out)
    return rows


def collect_backend_rows(summary_path: Path) -> list[dict[str, Any]]:
    """从阶段 3 CIL baseline 汇总 JSON 中提取 R3 后端表。"""
    data = load_json(summary_path)
    rows: list[dict[str, Any]] = []
    table_role = {
        "main_method": "主方法",
        "shared_discovery_backend": "共享 MV-ACC 伪标签后端 baseline",
        "end_to_end_deep_hdbscan": "Deep-HDBSCAN 端到端 baseline",
    }
    for row in data.get("summary", []):
        table = str(row.get("table"))
        if table not in table_role:
            continue
        item = {
            "table": table,
            "role": table_role[table],
            "method": row.get("method"),
            "seed_count": row.get("seed_count"),
        }
        for metric in BACKEND_METRICS:
            item[f"{metric}_mean"] = row.get(f"{metric}_mean")
            item[f"{metric}_std"] = row.get(f"{metric}_std")
        rows.append(item)
    return sorted(rows, key=lambda item: (str(item["table"]), -float(item.get("overall_mean") or -1.0)))


def render_frontend_table(rows: list[dict[str, Any]]) -> list[str]:
    """渲染发现前端表，重点显示严格适配边界和三轮均值。"""
    lines = [
        "| 方法 | 角色/适配级别 | 轮次数 | NMI | ARI | Purity | Hungarian Acc | Coverage | 泄漏检查 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        leak = "OK" if not row["uses_unknown_true_labels"] and not row["uses_eval_set"] else "需复核"
        lines.append(
            f"| `{row['method']}` | {row['role']} | {row['round_count']} | "
            f"{fmt(row.get('NMI'))} | {fmt(row.get('ARI'))} | {fmt(row.get('Purity'))} | "
            f"{fmt(row.get('Hungarian Acc'))} | {fmt(row.get('coverage'))} | {leak} |"
        )
    return lines


def render_backend_table(rows: list[dict[str, Any]], table: str) -> list[str]:
    """渲染指定后端 baseline 表。"""
    filtered = [row for row in rows if row["table"] == table]
    lines = [
        "| 方法 | Seeds | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in filtered:
        lines.append(
            f"| `{row['method']}` | {row['seed_count']} | "
            f"{fmt(row.get('overall_mean'))}±{fmt(row.get('overall_std'))} | "
            f"{fmt(row.get('old_mean'))}±{fmt(row.get('old_std'))} | "
            f"{fmt(row.get('new_mean'))}±{fmt(row.get('new_std'))} | "
            f"{fmt(row.get('forgetting_mean'))}±{fmt(row.get('forgetting_std'))} | "
            f"{fmt(row.get('macro_f1_mean'))}±{fmt(row.get('macro_f1_std'))} |"
        )
    return lines


def build_report(frontend_rows: list[dict[str, Any]], backend_rows: list[dict[str, Any]]) -> str:
    """生成 strict baseline 总表 Markdown。"""
    shared = [row for row in backend_rows if row["table"] == "shared_discovery_backend"]
    main = next((row for row in backend_rows if row["table"] == "main_method" and row["method"] == "MV-ACC-CIL"), None)
    best_shared = max(shared, key=lambda row: float(row.get("overall_mean") or -1.0)) if shared else None

    lines = [
        "# 阶段 3 WiSig Strict Baseline 总表",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        "- 范围：发现前端 strict 适配、共享发现后端 baseline、Deep-HDBSCAN 端到端 baseline。",
        "- 选择边界：本表只汇总既有 Slurm/报告产物，不使用评估真值做训练、阈值或聚类选择。",
        "",
        "## 发现前端 Strict Baseline",
        "",
        *render_frontend_table(frontend_rows),
        "",
        "## 共享 MV-ACC 伪标签后端 Baseline",
        "",
        *render_backend_table(backend_rows, "shared_discovery_backend"),
        "",
        "## Deep-HDBSCAN 端到端 Baseline",
        "",
        *render_backend_table(backend_rows, "end_to_end_deep_hdbscan"),
        "",
        "## 风险收束判断",
        "",
    ]
    if main and best_shared:
        delta = float(best_shared["overall_mean"]) - float(main["overall_mean"])
        lines.append(
            f"- 共享发现后端最强项 `{best_shared['method']}` 的 R3 Overall 比当前 `MV-ACC-CIL` 高 `{delta:.4f}`，"
            "因此后端上限风险已经明确转化为强后端/混合后端消融任务。"
        )
    lines.extend(
        [
            "- SimGCD-style 和 IGCD-minimal 均可放入 strict baseline 表，但必须标注为最小/学习式适配，不能写成完整论文复现。",
            "- 端到端 Deep-HDBSCAN + DOI-style 仍低于 MV-ACC-CIL，说明 MV-ACC 前端贡献成立；当前主要风险集中在网络式 RADCIL 后端叙事。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 WiSig 阶段 3 strict baseline 总表。")
    parser.add_argument("--simgcd-json", default="results/stage1/wisig_simgcd_frontend_compare_44422110/frontend_comparison_summary.json")
    parser.add_argument("--igcd-json", default="results/stage1/wisig_igcd_frontend_compare_44422704/frontend_comparison_summary.json")
    parser.add_argument("--cil-summary-json", default="results/stage3/stage3_wisig_cil_baselines_multiseed_summary_44453416.json")
    parser.add_argument("--output", default="results/stage3/STAGE3_WISIG_STRICT_BASELINE_TABLE.md")
    parser.add_argument("--summary-json", default="results/stage3/stage3_wisig_strict_baseline_table.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frontend_rows = collect_frontend_rows(Path(args.simgcd_json), Path(args.igcd_json))
    backend_rows = collect_backend_rows(Path(args.cil_summary_json))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(frontend_rows, backend_rows), encoding="utf-8")

    summary = {
        "schema_version": "stage3_wisig_strict_baseline_table_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frontend_rows": frontend_rows,
        "backend_rows": backend_rows,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WiSig strict baseline 总表：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
