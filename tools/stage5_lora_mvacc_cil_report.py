"""生成阶段 5 LoRa25 strict MV-ACC-CIL 单种子报告。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage2_wisig_main_report import extract_round_metrics, fmt


def build_markdown(save_dir: Path, metrics: dict, job_id: str) -> str:
    """构造 LoRa 跨体制单种子报告，明确 smoke 与正式结果的边界。"""
    lines = [
        "# 阶段 5 LoRa25 strict MV-ACC-CIL seed7 报告",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- Slurm Job：`{job_id}`",
        f"- 输出目录：`{save_dir.as_posix()}`",
        "- 协议：LoRa RFFP Different Days Indoor 10+5×3。",
        "- 隔离：Day1 IQ_1-6 训练、IQ_7 验证/校准、IQ_8-10 held-out 评估；Day2-4 IQ_1-7 discovery，IQ_8-10 held-out 评估。",
        "- 后端：RADCIL old:new=2.0、replay weight=3.0。",
        "",
        "| 轮次 | Cluster Count | NMI | ARI | Purity | Hungarian Acc | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["rounds"]:
        values = {key: fmt(value) for key, value in row.items()}
        lines.append(
            "| {round} | {cluster_count} | {nmi} | {ari} | {purity} | {hungarian_acc} | {overall_acc} | {old_acc} | {new_acc} | {forgetting_rate} | {macro_f1} |".format(**values)
        )
    r3 = next(row for row in metrics["rounds"] if row["round"] == "R3")
    lines.extend([
        "",
        "## 判定",
        "",
        f"- R3 Overall `{fmt(r3['overall_acc'])}`、Old `{fmt(r3['old_acc'])}`、New `{fmt(r3['new_acc'])}`、Forgetting `{fmt(r3['forgetting_rate'])}`、Macro F1 `{fmt(r3['macro_f1'])}`。",
        "- 只有三轮完整输出且指标可解释时才扩展 seed13/31；不得使用 held-out evaluation 真值回调发现或训练参数。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 LoRa25 strict MV-ACC-CIL seed7 报告。")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    save_dir = Path(args.save_dir)
    metrics = extract_round_metrics(save_dir)
    output = Path(args.output)
    summary = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(save_dir, metrics, args.job_id), encoding="utf-8")
    summary.write_text(json.dumps({"job_id": args.job_id, "save_dir": str(save_dir), "metrics": metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LoRa25 seed7 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
