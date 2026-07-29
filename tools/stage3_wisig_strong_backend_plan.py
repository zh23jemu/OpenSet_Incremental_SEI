"""生成 WiSig 强后端/混合后端消融计划。

Job 44453416 已证明：在共享 MV-ACC 伪标签条件下，DOI-style、iCaRL 和
TPCIL-style 的 R3 Overall 高于当前网络式 RADCIL。这个脚本把该风险
转成明确的后续消融矩阵，区分“已由现有 baseline 支持的参考上限”和
“需要新增代码后再运行的混合后端候选”。

脚本只生成计划和报告，不启动训练。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_METHODS = ("MV-ACC-CIL", "DOI-style", "iCaRL", "TPCIL-style")


def load_json(path: Path) -> dict[str, Any]:
    """读取阶段 3 baseline 汇总。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少 baseline 汇总 JSON：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_summary(summary: list[dict[str, Any]], table: str, method: str) -> dict[str, Any] | None:
    """按表名和方法名查找聚合行。"""
    for row in summary:
        if row.get("table") == table and row.get("method") == method:
            return row
    return None


def build_candidate_matrix(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造强后端/混合后端候选矩阵。

    第一组是已完成的 reference，不需要重跑；第二组是后续需要在代码中
    显式实现的混合候选，用于检验能否保留 RADCIL 网络学习能力，同时
    吸收原型/样本记忆后端的旧类保持优势。
    """
    main = find_summary(summary, "main_method", "MV-ACC-CIL")
    candidates: list[dict[str, Any]] = []
    for method in ("DOI-style", "iCaRL", "TPCIL-style"):
        row = find_summary(summary, "shared_discovery_backend", method)
        if row is None:
            continue
        delta = None
        if main is not None and row.get("overall_mean") is not None and main.get("overall_mean") is not None:
            delta = float(row["overall_mean"]) - float(main["overall_mean"])
        candidates.append(
            {
                "variant": f"reference_shared_mvacc_{method.lower().replace('-', '').replace(' ', '_')}",
                "status": "completed_reference",
                "backend_source": method,
                "run_requirement": "reuse_job_44453416_summary",
                "r3_overall_mean": row.get("overall_mean"),
                "r3_old_mean": row.get("old_mean"),
                "r3_new_mean": row.get("new_mean"),
                "r3_forgetting_mean": row.get("forgetting_mean"),
                "delta_vs_mvacc_cil_overall": delta,
                "purpose": "作为共享发现条件下的强后端参考上限，约束 RADCIL 后续叙事。",
            }
        )

    candidates.extend(
        [
            {
                "variant": "hybrid_radcil_doi_memory_alignment",
                "status": "completed_negative_ablation",
                "backend_source": "RADCIL + DOI-style",
                "run_requirement": "seed7 与三种子均已完成；三种子未稳定优于主方法，归档为负消融，不继续调 late-fusion 权重。",
                "r3_overall_mean": 0.6088,
                "r3_old_mean": 0.5451,
                "r3_new_mean": 0.8000,
                "r3_forgetting_mean": 0.2311,
                "purpose": "检验 DOI-style 的旧类原型保持是否能与网络式新类学习互补；结果显示该 late-fusion 机制不能解决后端上限风险。",
            },
            {
                "variant": "hybrid_radcil_icarl_exemplar_classifier_fallback",
                "status": "completed_negative_ablation",
                "backend_source": "RADCIL + iCaRL",
                "run_requirement": "seed7 Job 44771723 过门槛后扩展三种子 Job 44771757；三种子未稳定优于主方法。",
                "r3_overall_mean": 0.5923,
                "r3_old_mean": 0.5479,
                "r3_new_mean": 0.7256,
                "r3_forgetting_mean": 0.2285,
                "purpose": "检验 iCaRL 样本记忆分类器能否补足 RADCIL 旧类决策边界漂移；结果显示 seed7 局部收益不能跨种子稳定复现。",
            },
            {
                "variant": "hybrid_radcil_tpcil_graph_smoothed_prototypes",
                "status": "deferred_new_mechanism",
                "backend_source": "RADCIL + TPCIL-style",
                "run_requirement": "仅当 iCaRL fallback 不通过且仍继续后端研究时再实现；需先证明图平滑原型会带来结构性旧类保持收益。",
                "purpose": "检验 TPCIL-style 原型拓扑平滑是否能降低旧类遗忘而不牺牲新类。",
            },
        ]
    )
    return candidates


def build_plan(summary_path: Path) -> dict[str, Any]:
    """生成完整计划 JSON。"""
    data = load_json(summary_path)
    summary = data.get("summary", [])
    matrix = build_candidate_matrix(summary)
    return {
        "schema_version": "stage3_wisig_strong_backend_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary": str(summary_path),
        "source_job": data.get("job_id", "44453416"),
        "risk_resolved_as": "backend_ceiling_exposed_hybrid_absorption_negative",
        "selection_rule": (
            "先将 DOI-style/iCaRL/TPCIL-style 作为共享 MV-ACC 伪标签强后端参考；"
            "DOI-memory 与 iCaRL fallback 均已完成 seed7/三种子验证，但都未稳定优于当前 MV-ACC-CIL。"
        ),
        "candidate_matrix": matrix,
        "acceptance_gate": {
            "short_seed": 7,
            "primary_metric": "R3 Overall Acc",
            "secondary_metrics": ["R3 Old Acc", "R3 Forgetting Rate", "R3 New Acc"],
            "minimum_next_step": "不要继续调 DOI-memory 或 iCaRL fallback；若继续后端研究，必须换成结构不同的新机制并先预注册 seed7 门槛。",
        },
        "blocked_items": [
            "DOI-memory late fusion 已完成 seed7 与三种子验证，但未稳定优于主方法，不能作为主后端。",
            "iCaRL fallback seed7 通过但三种子未通过，不能作为主后端；TPCIL hybrid 仍延后。",
            "不能直接把共享发现 frozen-feature baseline 写成新主方法，只能作为后端上限、局限说明和后续机制设计依据。",
        ],
    }


def render_report(plan: dict[str, Any]) -> str:
    """渲染 Markdown 计划报告。"""
    lines = [
        "# 阶段 3 WiSig 强后端/混合后端消融计划",
        "",
        f"- 生成时间 UTC：{plan['created_at_utc']}",
        f"- 来源 Job：`{plan['source_job']}`",
        "- 目标：把共享发现后端 baseline 强于当前 RADCIL 的风险，转化为可执行消融，并记录 DOI-memory 与 iCaRL fallback 的负消融边界。",
        "",
        "## 候选矩阵",
        "",
        "| 变体 | 状态 | 后端来源 | R3 Overall | R3 Old | R3 New | R3 Forgetting | 下一步 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in plan["candidate_matrix"]:
        lines.append(
            "| `{variant}` | {status} | {source} | {overall} | {old} | {new} | {forgetting} | {requirement} |".format(
                variant=item["variant"],
                status=item["status"],
                source=item["backend_source"],
                overall=_fmt(item.get("r3_overall_mean")),
                old=_fmt(item.get("r3_old_mean")),
                new=_fmt(item.get("r3_new_mean")),
                forgetting=_fmt(item.get("r3_forgetting_mean")),
                requirement=item["run_requirement"],
            )
        )
    lines.extend(
        [
            "",
            "## 执行门槛",
            "",
            f"- 短验证 seed：`{plan['acceptance_gate']['short_seed']}`。",
            f"- 主指标：`{plan['acceptance_gate']['primary_metric']}`。",
            "- 次指标：`R3 Old Acc`、`R3 Forgetting Rate`、`R3 New Acc`。",
            f"- 扩展条件：{plan['acceptance_gate']['minimum_next_step']}",
            "",
            "## 当前限制",
            "",
        ]
    )
    for item in plan["blocked_items"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """表格数值格式化。"""
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 WiSig 强后端/混合后端消融计划。")
    parser.add_argument("--source-summary", default="results/stage3/stage3_wisig_cil_baselines_multiseed_summary_44453416.json")
    parser.add_argument("--output", default="results/stage3/STAGE3_WISIG_STRONG_BACKEND_PLAN.md")
    parser.add_argument("--plan-json", default="results/stage3/stage3_wisig_strong_backend_plan.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(Path(args.source_summary))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(plan), encoding="utf-8")

    plan_path = Path(args.plan_json)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WiSig 强后端/混合后端消融计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
