"""生成 SimGCD/IGCD 最小适配契约。

阶段 1 暂不直接把视觉 GCD 代码塞进主实验，而是先固化适配契约：
哪些输入允许使用、哪些输出必须提供、哪些调参来源被禁止。这样后续
实现 SimGCD 式学习发现头或 IGCD baseline 时，可以对照该 JSON 做
接口审计，避免无意间使用未知真值或评估集信息。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_contract(project_root: Path) -> dict[str, Any]:
    """返回阶段 1 GCD/SOTA 适配契约。"""
    return {
        "schema_version": "stage1_discovery_adapter_contract_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "shared_protocol": {
            "primary_dataset": "WiSig RX3 strict",
            "split": "Day1 60% train / 10% validation / 30% evaluation; each novel round uses discovery/enrollment data only for adaptation and held-out 30% only for evaluation.",
            "known_inputs_allowed": [
                "Day1 train IQ and known labels",
                "Day1 validation IQ and known labels for checkpoint/model selection/calibration",
                "Current round discovery IQ as unlabeled data",
                "Protocol-level expected round size when already fixed before seeing unknown labels",
            ],
            "forbidden_inputs": [
                "Current or future novel true labels for threshold selection, class-number selection, pseudo-label filtering, or training",
                "Held-out evaluation IQ or labels for model selection",
                "Posthoc Hungarian mapping except for final reporting",
            ],
        },
        "adapter_interface": {
            "fit_inputs": [
                "known_features: float32 [N_known, D]",
                "known_labels: int64 [N_known]",
                "unlabeled_features: float32 [N_unlabeled, D]",
                "validation_features: optional float32 [N_val, D]",
                "validation_labels: optional int64 [N_val]",
            ],
            "fit_outputs": [
                "adapter_state: serializable metadata and learned head parameters path",
                "calibration: thresholds or confidence rules with source=Day1 validation or label-free criterion",
            ],
            "predict_outputs": [
                "cluster_labels: int64 [N_unlabeled], dense 0..K-1 or -1 only before no-drop consolidation",
                "confidence: float32 [N_unlabeled], label-free confidence",
                "estimated_new_classes: int",
                "diagnostics: JSON-safe statistics",
            ],
        },
        "simgcd_minimal_adapter": {
            "role": "learning_discovery_frontend",
            "first_step": "Train a parametric discovery head on frozen RF/IQ embeddings using known CE plus unlabeled consistency or pseudo-cluster loss.",
            "comparison_target": "Deep-HDBSCAN/MV-ACC discovery labels under the same frozen embeddings and same round data.",
            "required_report_metrics": [
                "estimated_new_classes",
                "cluster_count_error",
                "assignment_coverage",
                "purity",
                "NMI",
                "ARI",
                "Hungarian Acc",
                "confidence_mean",
                "confidence_min",
            ],
            "implementation_status": "contract_ready_adapter_pending",
        },
        "igcd_minimal_adapter": {
            "role": "strict_candidate_baseline",
            "first_step": "Map IGCD time-step API to project rounds R1/R2/R3 and replace image augmentations with IQ augmentations.",
            "comparison_target": "Same 60/10/30 protocol and same initial known classes as MV-ACC-CIL.",
            "required_report_metrics": [
                "overall_acc",
                "old_acc",
                "new_acc",
                "forgetting_rate",
                "macro_f1",
                "discovery_purity_if_applicable",
            ],
            "implementation_status": "contract_ready_adapter_pending",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成阶段 1 SimGCD/IGCD 适配契约。")
    parser.add_argument("--project-root", default=".", help="项目根目录。")
    parser.add_argument("--output", default="results/stage1/stage1_discovery_adapter_contract.json", help="输出 JSON 路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_contract(project_root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 1 GCD 适配契约：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
