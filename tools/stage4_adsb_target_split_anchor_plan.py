"""生成 ADS-B target split + 训练期旧类原型锚定 seed7 消融计划。

该计划用于继续收敛阶段 4 的后端遗忘风险。前端固定为已验证的
target split，不再搜索发现小参数；只比较 RADCIL 训练期是否通过
Teacher replay 类中心锚定降低旧类漂移。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEED7_CHECKPOINT = "results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth"


def weight_tag(weight: float) -> str:
    """把浮点权重转换成稳定目录后缀，避免路径里出现多个点。"""

    return str(weight).replace(".", "p")


def experiment_args(data_root: str, save_dir: str, weight: float) -> list[str]:
    """构造单个 seed7 训练期锚定实验参数。

    这里显式关闭 CIL baselines，让本次作业只验证主 RADCIL 后端自身。
    target split、RADCIL old:new 配比和 replay 权重保持上一轮已收敛配置，
    从而保证差异只来自 `radcil_prototype_anchor_weight`。
    """

    return [
        "--data_root", data_root,
        "--save_dir", save_dir,
        "--checkpoint", SEED7_CHECKPOINT,
        "--initial_known_classes", "90",
        "--round_size", "10",
        "--num_rounds", "3",
        "--seed", "7",
        "--backbone", "adsb_long",
        "--batch_size", "128",
        "--test_batch_size", "256",
        "--use_supcon",
        "--supcon_weight", "0.1",
        "--cil_head_warmup_epochs", "2",
        "--cil_joint_epochs", "8",
        "--cil_replay_weight", "3.0",
        "--radcil_old_new_batch_ratio", "2.0",
        "--radcil_prototype_anchor_weight", str(weight),
        "--disable_cil_baselines",
        "--disable_mvacc_calibration",
        "--mvacc_min_cluster_ratio", "0.03",
        "--mvacc_merge_threshold", "0.74",
        "--mvacc_split_size_factor", "1.75",
        "--mvacc_split_silhouette", "0.30",
        "--enable_mvacc_target_split",
        "--mvacc_target_split_clusters", "10",
        "--mvacc_target_split_max_added", "4",
        "--mvacc_target_split_silhouette", "0.26",
        "--disable_visualization",
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """生成包含 baseline 和两个锚定权重的 seed7 矩阵。"""

    weights = [float(item.strip()) for item in args.weights.split(",") if item.strip()]
    runs = []
    for weight in weights:
        save_dir = f"{args.output_prefix}_w{weight_tag(weight)}_{args.job_id}"
        runs.append({
            "seed": 7,
            "anchor_weight": weight,
            "save_dir": save_dir,
            "checkpoint": SEED7_CHECKPOINT,
            "argv": experiment_args(args.data_root, save_dir, weight),
        })

    return {
        "schema_version": "stage4_adsb_target_split_anchor_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "seed": 7,
        "weights": weights,
        "output_prefix": args.output_prefix,
        "risk_question": (
            "默认 target split 已缓解 ADS-B 欠聚类后，训练期旧类 Teacher 原型锚定"
            "是否能降低 RADCIL Forgetting，同时不牺牲 Overall/New。"
        ),
        "locked_frontend": {
            "target_split": True,
            "target_clusters": 10,
            "max_added": 4,
            "silhouette": 0.26,
            "min_cluster_ratio": 0.03,
            "merge_threshold": 0.74,
        },
        "decision_rule": (
            "先看 seed7。若某个非零锚定权重相对 weight=0 能保持 Overall 不低于"
            " 0.002，同时 Old Acc 提升至少 0.010 或 Forgetting 降低至少 0.010，"
            "且 New Acc 下降不超过 0.020，则扩展 seed13/31；否则归档为负消融。"
        ),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ADS-B target split 原型锚定 seed7 消融计划。")
    parser.add_argument("--data-root", default="数据集/ADS-B/Dataset")
    parser.add_argument("--output-prefix", default="results/stage4/adsb_target_split_anchor_seed7")
    parser.add_argument("--weights", default="0,0.10,0.25")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", default="results/stage4/stage4_adsb_target_split_anchor_seed7_plan.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_plan(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ADS-B target split 原型锚定计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
