"""生成 ADS-B 阶段 4 单种子主流程计划。

ADS-B 是阶段 4 主实验入口。当前 `exp_adsb_mvacc_cil_strict.py` 与 WiSig
strict 脚本同构，但尚未接入 WiSig 阶段 3 使用的 old:new batch ratio
后端参数，也尚未把阶段 1 开发出的 ADS-B long-sequence backbone 合入
正式 strict 主流程。因此本计划先固化“可运行的 legacy strict 入口”，
并显式记录需要迁移的后端/表征缺口。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_plan(project_root: Path, data_root: str, save_dir: str, seed: int, job_id: str) -> dict[str, Any]:
    """构造 ADS-B 阶段 4 单种子审计计划。"""
    argv = [
        "--data_root",
        data_root,
        "--save_dir",
        save_dir,
        "--initial_known_classes",
        "90",
        "--round_size",
        "10",
        "--num_rounds",
        "3",
        "--seed",
        str(seed),
        "--train_closedset",
        "--epochs",
        "30",
        "--batch_size",
        "128",
        "--test_batch_size",
        "256",
        "--disable_cil_baselines",
        "--use_supcon",
        "--supcon_weight",
        "0.1",
        "--cil_head_warmup_epochs",
        "2",
        "--cil_joint_epochs",
        "8",
        "--cil_replay_weight",
        "3.0",
    ]
    return {
        "schema_version": "stage4_adsb_main_single_seed_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "stage": "stage4",
        "dataset": "ADS-B",
        "job_id": job_id,
        "seed": seed,
        "entry": "experiments/exp_adsb_mvacc_cil_strict.py",
        "save_dir": save_dir,
        "argv": argv,
        "protocol": "ADS-B strict 90 known + 10 x 3 incremental rounds; held-out evaluation is never used for model selection.",
        "selection_basis": {
            "stage0_loader_audit": "results/stage0/stage0_strict_loader_audit_44401081.json",
            "legacy_adsb_run": "results/adsb_90known_mvacc_cil_strict_seed31/ADSB_STRICT_RUN_REPORT.md",
            "closedset_dev": "results/adsb_closedset_long_dev_seed41/ADSB_PHASE1_CLOSEDSET_REPORT.md",
        },
        "known_gaps_before_formal_multiseed": [
            "正式 ADS-B strict 入口仍使用原 ClosedSetSEI 骨干，尚未接入 ADS-B long-sequence backbone。",
            "ADS-B strict 入口尚未实现 WiSig RADCIL 的 old:new batch ratio 参数；本计划只能锁定 replay weight，不能完全复刻 ratio_2p0_replay_3p0。",
            "历史 ADS-B strict seed31 初始闭集弱且 R2/R3 欠聚类，单种子结果只能作为迁移 smoke test。",
        ],
        "go_no_go": [
            "先确认脚本在 Slurm 当前 .venv 和 ADS-B 解压路径下可完整跑完三轮。",
            "若 R3 仍明显受闭集表征限制，下一步优先把 ADS-BLongClosedSet 合入 strict 主入口，而不是继续调 CIL 后端。",
            "只有单种子协议和产物完整后，才扩展 seed 7/13/31 正式三种子。",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 ADS-B 阶段 4单种子主流程计划。")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="数据集/ADS-B/Dataset")
    parser.add_argument("--save-dir", default="results/stage4/adsb_main_legacy_seed31_manual")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", default="results/stage4/stage4_adsb_main_single_seed_plan.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        project_root=project_root,
        data_root=args.data_root,
        save_dir=args.save_dir,
        seed=args.seed,
        job_id=args.job_id,
    )
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ADS-B 阶段 4 单种子计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
