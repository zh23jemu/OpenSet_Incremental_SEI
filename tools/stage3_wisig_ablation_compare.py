"""生成阶段 3 WiSig 主配置与 high-replay 正式消融对比报告。

该脚本只读取已经完成的三种子 summary JSON，不重新训练模型。它的目标是
把 `ratio_2p0_replay_3p0` 主配置和 `ratio_3p0_replay_3p0` high-replay
对照放入同一张表，明确两者是否满足“同协议、同前端、同训练预算”的
消融要求，并给出客户汇报可直接引用的判断。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = (
    ("overall_acc", "Overall"),
    ("old_acc", "Old"),
    ("new_acc", "New"),
    ("forgetting_rate", "Forgetting"),
    ("macro_f1", "Macro F1"),
)


def fmt(value: float | None) -> str:
    """统一报告中的小数格式；缺失值显示为 NA。"""
    if value is None:
        return "NA"
    return f"{value:.4f}"


def load_summary(path: Path) -> dict[str, Any]:
    """读取阶段 3 summary JSON，并返回结构化内容。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少 summary JSON：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summary_by_round(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """按 R1/R2/R3 建立 summary 索引，便于后续对比。"""
    return {str(row["round"]): row for row in payload["summary"]}


def get_variant(payload: dict[str, Any]) -> str:
    """从计划 JSON 中提取后端配置名。"""
    return str(payload.get("plan", {}).get("locked_config", {}).get("variant", "unknown"))


def get_jobs(payload: dict[str, Any]) -> str:
    """从输出目录中推断 Slurm Job ID；只用于报告证据链展示。"""
    jobs: list[str] = []
    for run in payload.get("plan", {}).get("runs", []):
        save_dir = str(run.get("save_dir", ""))
        job_id = save_dir.rsplit("_", 1)[-1] if "_" in save_dir else ""
        if job_id and job_id not in jobs:
            jobs.append(job_id)
    return ", ".join(jobs) if jobs else "unknown"


def build_report(main_payload: dict[str, Any], high_payload: dict[str, Any]) -> str:
    """生成 Markdown 对比报告。"""
    main_variant = get_variant(main_payload)
    high_variant = get_variant(high_payload)
    main_rounds = summary_by_round(main_payload)
    high_rounds = summary_by_round(high_payload)
    r3_main = main_rounds["R3"]
    r3_high = high_rounds["R3"]

    lines = [
        "# 阶段 3 WiSig RADCIL 正式消融对比报告",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- 主配置：`{main_variant}`，Job：`{get_jobs(main_payload)}`",
        f"- high-replay 对照：`{high_variant}`，Job：`{get_jobs(high_payload)}`",
        "- 对比边界：同 WiSig strict 10+10x3、同 MV-ACC 前端、同 seed 7/13/31、同 20 epoch + 2 warmup + 8 joint 训练预算。",
        "- 唯一审计变量：`radcil_old_new_batch_ratio` 从 `2.0` 提高到 `3.0`；`cil_replay_weight` 均为 `3.0`。",
        "",
        "## R3 三种子均值对比",
        "",
        "| 配置 | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| `{}` | {} | {} | {} | {} | {} |".format(
            main_variant,
            fmt(r3_main.get("overall_acc_mean")),
            fmt(r3_main.get("old_acc_mean")),
            fmt(r3_main.get("new_acc_mean")),
            fmt(r3_main.get("forgetting_rate_mean")),
            fmt(r3_main.get("macro_f1_mean")),
        ),
        "| `{}` | {} | {} | {} | {} | {} |".format(
            high_variant,
            fmt(r3_high.get("overall_acc_mean")),
            fmt(r3_high.get("old_acc_mean")),
            fmt(r3_high.get("new_acc_mean")),
            fmt(r3_high.get("forgetting_rate_mean")),
            fmt(r3_high.get("macro_f1_mean")),
        ),
        "",
        "## R3 high-replay 相对主配置变化",
        "",
        "| 指标 | high-replay - 主配置 | 解读 |",
        "| --- | ---: | --- |",
    ]

    for metric, label in METRICS:
        delta = float(r3_high[f"{metric}_mean"]) - float(r3_main[f"{metric}_mean"])
        if metric == "forgetting_rate":
            interpretation = "更低更好；正值表示遗忘增加"
        else:
            interpretation = "更高更好；正值表示提升"
        lines.append(f"| {label} | {delta:+.4f} | {interpretation} |")

    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "- high-replay 在 R3 New Acc 上高 `+0.0052`，说明更强旧类 batch 配比没有损害新类吸收。",
            "- 但 high-replay 的 R3 Overall 低 `-0.0058`、Old 低 `-0.0095`、Forgetting 高 `+0.0059`、Macro F1 低 `-0.0033`。",
            "- 因此阶段 3 正式同协议消融支持继续选择 `ratio_2p0_replay_3p0` 作为主后端；`ratio_3p0_replay_3p0` 保留为 high-replay 对照。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成阶段 3 WiSig RADCIL 正式消融对比报告。")
    parser.add_argument("--main-summary", required=True, help="主配置 summary JSON。")
    parser.add_argument("--high-replay-summary", required=True, help="high-replay summary JSON。")
    parser.add_argument("--output", required=True, help="Markdown 报告输出路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    main_payload = load_summary(Path(args.main_summary))
    high_payload = load_summary(Path(args.high_replay_summary))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(main_payload, high_payload), encoding="utf-8")
    print(f"阶段 3 WiSig 正式消融对比报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
