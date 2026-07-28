"""LoRa25 Day1 跨 transmission 闭集表征筛选。

四个预注册训练配方只使用 IQ_1-6 训练、IQ_7 选择 checkpoint 和候选；
IQ_8-10 准确率仅在训练结束后写入事后审计，不参与排序或选型。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lora25_strict_loader import load_lora25_diffdays_3round  # noqa: E402
from experiments.exp_wisig_mvacc_cil_strict import to_dataset, train_closedset_model  # noqa: E402
from utils.improved_closedset_training import TRAINING_RECIPE_VERSION, evaluate_classifier  # noqa: E402


VARIANTS = (
    {"name": "ce_no_aug", "use_supcon": False, "use_rf_augmentation": False},
    {"name": "ce_rf_aug", "use_supcon": False, "use_rf_augmentation": True},
    {"name": "supcon_no_aug", "use_supcon": True, "use_rf_augmentation": False},
    {"name": "supcon_rf_aug", "use_supcon": True, "use_rf_augmentation": True},
)


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRa25 strict Day1 表征筛选。")
    parser.add_argument("--npz-path", default="datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_lora25_diffdays_3round(args.npz_path)
    train_set = to_dataset(splits["day1_backbone_train"]["X"], splits["day1_backbone_train"]["y"])
    validation_set = to_dataset(splits["day1_known_validation"]["X"], splits["day1_known_validation"]["y"])
    evaluation_set = to_dataset(splits["day1_initial_eval"]["X"], splits["day1_initial_eval"]["y"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for variant in VARIANTS:
        checkpoint = output_dir / f"{variant['name']}.pth"
        metadata = {
            "dataset_path": str(Path(args.npz_path).resolve()),
            "selected_rx_list": [2],
            "known_tx": list(range(10)),
            "training_day_index": 0,
            "development_ratio": 0.70,
            "dataset_profile": "lora25",
            "sample_split_protocol": "lora25_transmission_disjoint_60_10_30_v1",
            "seed": int(args.seed),
            "use_supcon": bool(variant["use_supcon"]),
            "supcon_weight": 0.1,
            "supcon_temperature": 0.2,
            "training_recipe_version": TRAINING_RECIPE_VERSION,
            "closedset_validation_fraction": 1.0 / 7.0,
            "rf_augmentation": bool(variant["use_rf_augmentation"]),
            "supcon_projection_dim": 64 if variant["use_supcon"] else 0,
            "supcon_projection_hidden_dim": 128 if variant["use_supcon"] else 0,
        }
        model = train_closedset_model(
            train_set, 10, 128, args.epochs, args.batch_size, args.lr, device, str(checkpoint),
            use_supcon=variant["use_supcon"], supcon_weight=0.1, supcon_temperature=0.2,
            checkpoint_metadata=metadata, seed=args.seed, use_rf_augmentation=variant["use_rf_augmentation"],
            validation_set=validation_set,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        selection = payload.get("selection", {})
        eval_acc, eval_loss = evaluate_classifier(
            model, DataLoader(evaluation_set, batch_size=256, shuffle=False), device
        )
        rows.append({
            **variant,
            "validation_accuracy": float(selection.get("validation_accuracy", 0.0)),
            "validation_loss": float(selection.get("validation_loss", 0.0)),
            "selection_epoch": int(selection.get("epoch", 0)),
            "posthoc_initial_eval_accuracy": float(eval_acc),
            "posthoc_initial_eval_loss": float(eval_loss),
            "checkpoint": str(checkpoint),
        })

    rows.sort(key=lambda row: (-row["validation_accuracy"], row["validation_loss"], row["name"]))
    report = {
        "selection_metric": "IQ_7 validation_accuracy only",
        "heldout_eval_role": "posthoc report only; not used for ranking",
        "selected_variant": rows[0]["name"],
        "rows": rows,
    }
    path = output_dir / "representation_screen.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
