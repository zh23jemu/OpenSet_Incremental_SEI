"""阶段 5 LoRa25 单种子训练 smoke test。

该脚本用于验证 LoRa strict loader 能进入真实 PyTorch 训练/表征/发现链路。
它刻意保持为轻量 smoke：

* 只在 Day1 IQ_1-6 的 60% 训练池训练闭集骨干；
* 只用 Day1 IQ_7 验证集报告模型状态，不据此反复调参；
* 对三轮未知 discovery pool 做固定参数 HDBSCAN 审计；
* NMI/ARI/Hungarian 等标签指标只写入事后审计，不参与训练或聚类选择。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from torch.utils.data import DataLoader, TensorDataset

try:
    import hdbscan
except ImportError as exc:  # pragma: no cover - 环境缺依赖时给出明确错误
    raise ImportError("LoRa smoke test requires hdbscan in the project .venv.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lora25_strict_loader import load_lora25_diffdays_3round  # noqa: E402
from models.vup_model import ClosedSetSEI  # noqa: E402


def set_seed(seed: int) -> None:
    """固定随机种子，保证 smoke 结果可复查。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """把 numpy IQ split 转成 PyTorch DataLoader。"""
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_closedset(
    model: ClosedSetSEI,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[dict[str, float]]:
    """执行最小闭集训练，只用于确认 LoRa 数据能驱动现有 1D 骨干。"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        correct = 0
        total = 0
        for xb, yb in to_loader(x, y, batch_size=batch_size, shuffle=True):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(yb.numel())
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(yb.numel())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss_sum / max(total, 1),
                "train_acc": correct / max(total, 1),
            }
        )
    return history


@torch.no_grad()
def evaluate_closedset(model: ClosedSetSEI, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> dict[str, float]:
    """评估已知类闭集准确率；验证集和评估集只用于报告。"""
    model.eval()
    correct = 0
    total = 0
    for xb, yb in to_loader(x, y, batch_size=batch_size, shuffle=False):
        xb = xb.to(device)
        yb = yb.to(device)
        _, logits = model(xb)
        correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.numel())
    return {"samples": int(total), "accuracy": correct / max(total, 1)}


@torch.no_grad()
def extract_features(model: ClosedSetSEI, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """提取 L2-normalized deep embedding，供固定发现参数 smoke 审计使用。"""
    model.eval()
    chunks = []
    for xb, _ in to_loader(x, np.zeros(len(x), dtype=np.int64), batch_size=batch_size, shuffle=False):
        feat, _ = model(xb.to(device))
        feat = F.normalize(feat, dim=1)
        chunks.append(feat.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """事后审计聚类-真值最佳匹配准确率；不用于发现流程。"""
    valid = y_pred >= 0
    if not np.any(valid):
        return 0.0
    true = y_true[valid]
    pred = y_pred[valid]
    true_values = np.unique(true)
    pred_values = np.unique(pred)
    cost = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    for i, tv in enumerate(true_values):
        for j, pv in enumerate(pred_values):
            cost[i, j] = -int(np.sum((true == tv) & (pred == pv)))
    rows, cols = linear_sum_assignment(cost)
    matched = -int(cost[rows, cols].sum()) if len(rows) else 0
    return matched / max(len(y_true), 1)


def audit_discovery(features: np.ndarray, y_true: np.ndarray, min_cluster_ratio: float, min_samples: int) -> dict[str, Any]:
    """使用固定 HDBSCAN 参数审计未知发现结构。"""
    min_cluster_size = max(5, int(round(len(features) * min_cluster_ratio)))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    labels = clusterer.fit_predict(features)
    valid = labels >= 0
    unique_clusters = sorted(int(v) for v in np.unique(labels[valid]))
    if len(unique_clusters) >= 2 and int(valid.sum()) > len(unique_clusters):
        sil = float(silhouette_score(features[valid], labels[valid]))
    else:
        sil = None
    return {
        "samples": int(len(y_true)),
        "true_classes": int(np.unique(y_true).size),
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "clusters": int(len(unique_clusters)),
        "noise": int(np.sum(~valid)),
        "label_free_silhouette": sil,
        "posthoc_nmi": float(normalized_mutual_info_score(y_true, labels)),
        "posthoc_ari": float(adjusted_rand_score(y_true, labels)),
        "posthoc_hungarian": float(hungarian_accuracy(y_true, labels)),
    }


def build_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """执行 LoRa 单种子 smoke，并返回完整 JSON 报告。"""
    set_seed(args.seed)
    # auto 在 GPU 节点优先使用 CUDA；CPU/defq smoke 环境则必须显式回落到 cpu，
    # 避免把字符串 "auto" 直接传给 torch.device 导致 Slurm smoke 失败。
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    splits = load_lora25_diffdays_3round(args.npz_path)

    model = ClosedSetSEI(num_known_classes=10, feat_dim=args.feat_dim).to(device)
    history = train_closedset(
        model,
        splits["day1_backbone_train"]["X"],
        splits["day1_backbone_train"]["y"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    validation = evaluate_closedset(
        model,
        splits["day1_known_validation"]["X"],
        splits["day1_known_validation"]["y"],
        batch_size=args.batch_size,
        device=device,
    )
    initial_eval = evaluate_closedset(
        model,
        splits["day1_initial_eval"]["X"],
        splits["day1_initial_eval"]["y"],
        batch_size=args.batch_size,
        device=device,
    )

    discovery_rows = []
    for round_index, split_key in enumerate(
        ("day2_unknown_round1", "day3_unknown_round2", "day4_unknown_round3"),
        start=1,
    ):
        item = splits[split_key]
        features = extract_features(model, item["X"], batch_size=args.batch_size, device=device)
        row = audit_discovery(features, item["y"], args.min_cluster_ratio, args.min_samples)
        row.update({"round": int(round_index), "split": split_key})
        discovery_rows.append(row)

    output = {
        "schema_version": "stage5_lora_closedset_smoke_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "dataset": "LoRa RFFP Different Days Indoor compact aligned subset",
        "npz_path": args.npz_path,
        "seed": int(args.seed),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "protocol_boundary": "Day1 IQ_1-6 train, IQ_7 validation, IQ_8-10 held-out eval; Day2-4 IQ_1-7 discovery and IQ_8-10 held-out eval.",
        "selection_boundary": "训练 smoke 不根据 validation/eval 或 posthoc 聚类真值调参；发现参数来自命令行固定值。",
        "train_history": history,
        "day1_validation": validation,
        "day1_initial_eval": initial_eval,
        "discovery_audit": discovery_rows,
        "next_step_if_ok": "在 Slurm 上运行同一 smoke；若资源与依赖通过，再扩展为 MV-ACC-CIL 单种子入口。",
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 5 LoRa25 单种子闭集与发现 smoke test。")
    parser.add_argument("--npz-path", default="datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz")
    parser.add_argument("--output", default="results/stage5/stage5_lora_closedset_smoke_seed7_local.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--min-cluster-ratio", type=float, default=0.03)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_smoke(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
