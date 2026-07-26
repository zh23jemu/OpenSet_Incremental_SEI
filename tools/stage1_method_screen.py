"""阶段 1 方法筛选与短实验计划生成工具。

该脚本不训练模型、不读取测试真值做调参，只汇总仓库中已经存在的
MV-ACC-CIL 结果，并输出阶段 1 后续短实验矩阵。它的作用是把“可靠
伪标签 + 回放 + 蒸馏效果一般”的口头风险，转成可审计、可复现的
筛选依据和下一步 Slurm 任务入口。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 为字典列表；文件缺失时返回空列表，便于跨机器复用。"""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str | None) -> float | None:
    """将 CSV 字符串安全转为 float；空值或非法值统一返回 None。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _summarize_per_round(path: Path) -> dict[str, Any]:
    """从 per_round_summary_results.csv 提取阶段 1 需要的核心信号。"""
    rows = _read_csv_rows(path)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                "round": row.get("Round"),
                "cluster_count": _as_float(row.get("Cluster Count")),
                "pseudo_label_classes": _as_float(row.get("Incremental Pseudo-label Classes")),
                "purity": _as_float(row.get("Purity")),
                "nmi": _as_float(row.get("NMI")),
                "ari": _as_float(row.get("ARI")),
                "hungarian_acc": _as_float(row.get("Hungarian Acc")),
                "overall_acc": _as_float(row.get("Overall Acc")),
                "new_acc": _as_float(row.get("New Acc")),
                "forgetting_rate": _as_float(row.get("Forgetting Rate")),
            }
        )
    return {
        "path": str(path),
        "exists": path.exists(),
        "rounds": parsed,
    }


def _build_existing_evidence(project_root: Path) -> dict[str, Any]:
    """汇总现有 WiSig、ManyTx、ADS-B 的 MV-ACC-CIL 证据。"""
    result_files = {
        "wisig_rx3_cil_v2": project_root / "results/wisig_rx3_mvacc_cil_v2_seed7/per_round_summary_results.csv",
        "manytx_rx2_cil_v2": project_root / "results/manytx_rx2_mvacc_cil_v2_seed7/per_round_summary_results.csv",
        "adsb_90known_strict": project_root / "results/adsb_90known_mvacc_cil_strict_seed31/per_round_summary_results.csv",
    }
    evidence = {name: _summarize_per_round(path) for name, path in result_files.items()}
    evidence["diagnosis"] = [
        {
            "dataset": "WiSig RX3",
            "judgment": "发现质量高，新类能学进分类头，但 R2/R3 旧类遗忘明显；优先检查表征漂移、回放容量、KD 权重和冻结/解冻策略。",
        },
        {
            "dataset": "ManyTx RX2",
            "judgment": "发现指标尚可但端到端准确率很低；既有报告显示 Day1 神经分类器 held-out 初始准确率偏低，优先视为表征瓶颈。",
        },
        {
            "dataset": "ADS-B",
            "judgment": "strict 流程可跑通但初始 90 类模型弱且后续欠聚类；优先处理长序列表征和类别发现前端。",
        },
    ]
    return evidence


def _build_sota_candidates() -> list[dict[str, Any]]:
    """列出阶段 1 候选 SOTA 及公平适配边界。"""
    return [
        {
            "name": "IGCD",
            "role": "strict_candidate",
            "source": "Incremental Generalized Category Discovery, ICCV 2023 / arXiv 2304.14310",
            "url": "https://arxiv.org/abs/2304.14310",
            "why": "任务形态接近多轮新类发现与旧类保持，可作为开放增量发现/保持的严格候选。",
            "rf_adaptation_boundary": [
                "输入统一使用本项目 WiSig strict 60/10/30 协议。",
                "图像增强替换为项目 IQ 增强，禁止用未知轮次真值选择类别数。",
                "先做接口草图和单轮小样本验证，再决定是否纳入正式 baseline。",
            ],
            "current_status": "adapter_design_required",
        },
        {
            "name": "SimGCD",
            "role": "strict_candidate_discovery_frontend",
            "source": "Parametric Classification for Generalized Category Discovery, arXiv 2211.11727",
            "url": "https://arxiv.org/abs/2211.11727",
            "code": "https://github.com/CVMI-Lab/SimGCD",
            "why": "代码可用，适合作为学习式类别发现头，对比 Deep-HDBSCAN/MV-ACC 的手工密度聚类前端。",
            "rf_adaptation_boundary": [
                "每轮只使用已知类训练/验证数据和当前 discovery 未标注样本。",
                "类别数和阈值只能来自 Day1 验证或无标签准则，不能使用未知真值。",
                "先作为发现前端候选，不单独声称解决完整 CIL 遗忘。",
            ],
            "current_status": "short_experiment_required",
        },
    ]


def _build_short_experiment_matrix() -> list[dict[str, Any]]:
    """生成 WiSig 单种子短实验矩阵，供 Slurm 脚本和人工审计共用。"""
    base_common = [
        "--dataset_path 数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl",
        "--selected_rx_list 2",
        "--initial_known_classes 10",
        "--round_size 10",
        "--num_rounds 3",
        "--seed 7",
        "--train_closedset",
        "--epochs 8",
        "--batch_size 128",
        "--test_batch_size 256",
        "--disable_visualization",
        "--disable_cil_baselines",
        "--cil_head_warmup_epochs 1",
        "--cil_joint_epochs 3",
    ]
    return [
        {
            "variant": "ce_baseline",
            "purpose": "复现轻量 CE 初始骨干，作为表征短实验下界。",
            "args": base_common,
            "expected_outputs": [
                "per_round_summary_results.csv",
                "incremental_results.csv",
                "clustering_results.csv",
                "fixed_day1_retention.csv",
            ],
        },
        {
            "variant": "supcon_representation",
            "purpose": "比较 SupCon 初始表征是否降低旧类遗忘或改善发现质量。",
            "args": base_common + ["--use_supcon", "--supcon_weight 0.1"],
            "expected_outputs": [
                "per_round_summary_results.csv",
                "incremental_results.csv",
                "clustering_results.csv",
                "fixed_day1_retention.csv",
            ],
        },
    ]


def build_report(project_root: Path) -> dict[str, Any]:
    """组装完整阶段 1 筛选报告。"""
    return {
        "schema_version": "stage1_method_screen_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "existing_evidence": _build_existing_evidence(project_root),
        "sota_candidates": _build_sota_candidates(),
        "short_experiment_matrix": _build_short_experiment_matrix(),
        "decision": {
            "pseudo_replay_distill_status": "baseline_or_ablation_only",
            "main_contribution_priority": [
                "RF/IQ 表征稳定性",
                "学习式或约束式类别发现前端",
                "经过消融证明有效的后端训练策略",
            ],
            "next_gate": "运行 WiSig 单种子短实验并比较 CE baseline 与 SupCon representation 变体。",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成阶段 1 方法复盘和短实验筛选报告。")
    parser.add_argument("--project-root", default=".", help="项目根目录。")
    parser.add_argument("--output", default="results/stage1/stage1_method_screen.json", help="输出 JSON 路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    report = build_report(project_root)
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 1 方法筛选报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
