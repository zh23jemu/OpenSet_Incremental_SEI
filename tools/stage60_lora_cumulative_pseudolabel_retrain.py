#!/usr/bin/env python3
"""Stage60 LoRa 累积伪标签全量重训验证。

Stage53-59 已经排除一批相邻小机制。本阶段改用更大的结构性验证：
每一轮都用上一轮模型的 LoRaChirp 表征给当前 discovery 固定 K=5 聚类，
把新簇注册成伪类，然后用所有已见真实/伪标签样本重新训练一个分类器。

这个流程不读取 held-out eval 真值，不用 HDBSCAN，也不使用 RADCIL 的增量
head/KD/replay 训练逻辑。它用于回答一个更直接的问题：如果完全避开
RADCIL 旧/新类训练权衡，只靠累积数据重训，LoRa strict 低分能不能被拉起。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lora25_strict_loader import load_lora25_diffdays_3round
from models.lora_chirp_model import LoRaChirpClosedSet
from utils.improved_closedset_training import train_closedset_with_validation


STAGE48_MULTISEED = {
    "overall": 0.2803,
    "old": 0.2312,
    "new": 0.4770,
    "forgetting": 0.1802,
}


def set_seed(seed: int) -> None:
    """固定随机源，尽量保证 Slurm seed7 短实验可复现。"""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    """把 numpy I/Q 和标签转成训练工具需要的 TensorDataset。"""
    return TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))


@torch.no_grad()
def extract_features(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """提取 LoRaChirp backbone 特征，供当前轮 KMeans discovery 使用。"""
    loader = DataLoader(TensorDataset(torch.as_tensor(x, dtype=torch.float32)), batch_size=int(batch_size), shuffle=False)
    feats = []
    model.eval()
    for (xb,) in loader:
        feat, _ = model(xb.to(device))
        feats.append(feat.detach().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """批量输出分类 logits。"""
    loader = DataLoader(TensorDataset(torch.as_tensor(x, dtype=torch.float32)), batch_size=int(batch_size), shuffle=False)
    logits = []
    model.eval()
    for (xb,) in loader:
        _, logit = model(xb.to(device))
        logits.append(logit.detach().cpu().numpy())
    return np.concatenate(logits, axis=0).astype(np.float32)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    save_path: Path,
    args: argparse.Namespace,
    device: str,
) -> LoRaChirpClosedSet:
    """训练一个指定输出类别数的 LoRaChirp 分类器。

    验证集仍使用协议允许的 Day1 IQ_7 已知类，用于约束旧类不被完全遗忘；
    新类 discovery 真值不会进入训练或 checkpoint 选择。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model_factory = lambda: LoRaChirpClosedSet(num_known_classes=int(num_classes), feat_dim=int(args.feat_dim))
    train_closedset_with_validation(
        train_set=to_dataset(x_train, y_train),
        model_factory=model_factory,
        num_classes=int(num_classes),
        feat_dim=int(args.feat_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        save_path=str(save_path),
        seed=int(args.seed),
        use_supcon=bool(args.use_supcon),
        supcon_weight=float(args.supcon_weight),
        supcon_temperature=float(args.supcon_temperature),
        checkpoint_metadata={
            "stage": "stage60_lora_cumulative_pseudolabel_retrain",
            "num_classes": int(num_classes),
            "seed": int(args.seed),
        },
        validation_set=to_dataset(x_val, y_val),
        use_rf_augmentation=not bool(args.disable_rf_augmentation),
    )
    checkpoint = torch.load(save_path, map_location=device)
    model = model_factory().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def recording_consensus(labels: np.ndarray, recording_id: np.ndarray, threshold: float) -> tuple[np.ndarray, dict[str, float]]:
    """用可观测 recording_id 对 discovery 聚类做多数共识修正。"""
    labels = np.asarray(labels, dtype=np.int64).copy()
    recording_id = np.asarray(recording_id)
    if float(threshold) <= 0:
        return labels, {
            "Recording Consensus Threshold": 0.0,
            "Recording Consensus Accepted Groups": 0,
            "Recording Consensus Changed Samples": 0,
            "Recording Consensus Mean Consistency": 0.0,
        }
    accepted = 0
    changed = 0
    consistencies = []
    for group in np.unique(recording_id):
        idx = np.where(recording_id == group)[0]
        values, counts = np.unique(labels[idx], return_counts=True)
        best_pos = int(np.argmax(counts))
        consistency = float(counts[best_pos] / len(idx))
        consistencies.append(consistency)
        if consistency >= float(threshold):
            accepted += 1
            before = labels[idx].copy()
            labels[idx] = int(values[best_pos])
            changed += int(np.sum(before != labels[idx]))
    return labels, {
        "Recording Consensus Threshold": float(threshold),
        "Recording Consensus Accepted Groups": int(accepted),
        "Recording Consensus Changed Samples": int(changed),
        "Recording Consensus Mean Consistency": float(np.mean(consistencies)) if consistencies else 0.0,
    }


def hungarian_acc(y_true: np.ndarray, cluster_labels: np.ndarray) -> float:
    """计算 discovery 聚类 Hungarian accuracy，仅用于离线报告。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    true_ids = sorted(np.unique(y_true).tolist())
    cluster_ids = sorted(np.unique(cluster_labels).tolist())
    contingency = np.zeros((len(true_ids), len(cluster_ids)), dtype=np.int64)
    true_pos = {int(v): i for i, v in enumerate(true_ids)}
    cluster_pos = {int(v): i for i, v in enumerate(cluster_ids)}
    for yt, yc in zip(y_true, cluster_labels):
        contingency[true_pos[int(yt)], cluster_pos[int(yc)]] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    return float(contingency[row_ind, col_ind].sum() / max(1, len(y_true)))


def build_mapping(y_true: np.ndarray, cluster_labels: np.ndarray, pseudo_y: np.ndarray) -> dict[int, int]:
    """把伪类映射到真实新类，仅用于报告指标。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    pseudo_y = np.asarray(pseudo_y, dtype=np.int64)
    cluster_ids = sorted(np.unique(cluster_labels).tolist())
    true_ids = sorted(np.unique(y_true).tolist())
    contingency = np.zeros((len(true_ids), len(cluster_ids)), dtype=np.int64)
    true_pos = {int(v): i for i, v in enumerate(true_ids)}
    for col, cluster_id in enumerate(cluster_ids):
        idx = cluster_labels == int(cluster_id)
        for value, count in zip(*np.unique(y_true[idx], return_counts=True)):
            contingency[true_pos[int(value)], col] = int(count)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {}
    for cluster_id in cluster_ids:
        pseudo = int(np.unique(pseudo_y[cluster_labels == int(cluster_id)])[0])
        mapping[pseudo] = -(1_000_000 + pseudo)
    for row, col in zip(row_ind, col_ind):
        cluster_id = cluster_ids[int(col)]
        pseudo = int(np.unique(pseudo_y[cluster_labels == int(cluster_id)])[0])
        mapping[pseudo] = int(true_ids[int(row)])
    return mapping


def acc_range(y_true: np.ndarray, y_pred: np.ndarray, start: int, end: int) -> float:
    """计算类别区间准确率。"""
    mask = (y_true >= int(start)) & (y_true < int(end))
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_true[mask] == y_pred[mask]))


def evaluate(stage: str, y_true: np.ndarray, raw_pred: np.ndarray, mapping: dict[int, int], initial_ref: float | None) -> dict[str, float | str]:
    """按项目 old/new/forgetting 定义计算指标。"""
    if stage == "Initial":
        seen_classes = 10
        old_start, old_end = 0, 10
        new_start, new_end = None, None
    else:
        round_index = int(stage.replace("After R", ""))
        seen_classes = 10 + round_index * 5
        old_start, old_end = 0, 10 + (round_index - 1) * 5
        new_start, new_end = 10 + (round_index - 1) * 5, 10 + round_index * 5
    seen_mask = np.asarray(y_true, dtype=np.int64) < int(seen_classes)
    y_seen = np.asarray(y_true, dtype=np.int64)[seen_mask]
    pred_seen = np.asarray([mapping.get(int(p), int(p)) for p in raw_pred[seen_mask]], dtype=np.int64)
    initial_acc = acc_range(y_seen, pred_seen, 0, 10)
    forgetting = 0.0 if initial_ref is None else float(initial_ref - initial_acc)
    return {
        "Method": "LoRa cumulative pseudo-label retrain",
        "Stage": stage,
        "Seen Classes": int(seen_classes),
        "Eval Samples": int(len(y_seen)),
        "Overall Acc": float(np.mean(y_seen == pred_seen)),
        "Old Acc": acc_range(y_seen, pred_seen, old_start, old_end),
        "New Acc": float("nan") if new_start is None else acc_range(y_seen, pred_seen, new_start, new_end),
        "Initial Known Acc": initial_acc,
        "Forgetting Rate": forgetting,
        "Macro F1": float(f1_score(y_seen, pred_seen, average="macro", zero_division=0)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """执行 Stage60 累积伪标签重训。"""
    set_seed(int(args.seed))
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    splits = load_lora25_diffdays_3round(args.dataset_path)

    x_train = np.asarray(splits["day1_backbone_train"]["X"], dtype=np.float32)
    y_train = np.asarray(splits["day1_backbone_train"]["y"], dtype=np.int64)
    x_val = np.asarray(splits["day1_known_validation"]["X"], dtype=np.float32)
    y_val = np.asarray(splits["day1_known_validation"]["y"], dtype=np.int64)
    cumulative_x = [x_train]
    cumulative_y = [y_train]
    pseudo_to_true = {int(label): int(label) for label in range(10)}
    next_label = 10
    clustering_rows = []
    incremental_rows = []

    model = train_model(
        np.concatenate(cumulative_x),
        np.concatenate(cumulative_y),
        x_val,
        y_val,
        10,
        save_dir / "cumulative_model_initial.pth",
        args,
        device,
    )
    initial_logits = predict_logits(model, splits["day1_initial_eval"]["X"], args.test_batch_size, device)
    initial_pred = np.argmax(initial_logits, axis=1).astype(np.int64)
    initial_row = evaluate("Initial", splits["day1_initial_eval"]["y"], initial_pred, pseudo_to_true, None)
    incremental_rows.append(initial_row)
    initial_ref = float(initial_row["Initial Known Acc"])

    for round_index in range(1, 4):
        discovery_key = f"day{round_index + 1}_unknown_round{round_index}"
        eval_key = f"day{round_index + 1}_eval_after_r{round_index}"
        discovery_x = np.asarray(splits[discovery_key]["X"], dtype=np.float32)
        discovery_y_true = np.asarray(splits[discovery_key]["y"], dtype=np.int64)
        discovery_features = extract_features(model, discovery_x, args.test_batch_size, device)
        scaled_features = StandardScaler().fit_transform(discovery_features)
        cluster_labels = KMeans(n_clusters=5, n_init=20, random_state=int(args.seed) + round_index).fit_predict(scaled_features)
        cluster_labels, consensus_diag = recording_consensus(
            cluster_labels,
            splits[discovery_key]["recording_id"],
            args.recording_consensus_threshold,
        )
        cluster_ids = sorted(np.unique(cluster_labels).astype(int).tolist())
        cid_to_pseudo = {cid: int(next_label + idx) for idx, cid in enumerate(cluster_ids)}
        pseudo_y = np.asarray([cid_to_pseudo[int(cid)] for cid in cluster_labels], dtype=np.int64)
        pseudo_to_true.update(build_mapping(discovery_y_true, cluster_labels, pseudo_y))
        next_label += len(cluster_ids)
        cumulative_x.append(discovery_x)
        cumulative_y.append(pseudo_y)

        model = train_model(
            np.concatenate(cumulative_x),
            np.concatenate(cumulative_y),
            x_val,
            y_val,
            10 + round_index * 5,
            save_dir / f"cumulative_model_after_r{round_index}.pth",
            args,
            device,
        )
        eval_logits = predict_logits(model, splits[eval_key]["X"], args.test_batch_size, device)
        eval_pred = np.argmax(eval_logits, axis=1).astype(np.int64)
        incremental_rows.append(
            evaluate(f"After R{round_index}", splits[eval_key]["y"], eval_pred, pseudo_to_true, initial_ref)
        )

        values, counts = np.unique(cluster_labels, return_counts=True)
        clustering_rows.append({
            "Method": "cumulative pseudo-label retrain discovery",
            "Round": f"R{round_index}",
            "Samples": int(len(discovery_y_true)),
            "Final Cluster Count": int(len(values)),
            "Cluster Count Error": int(abs(len(values) - 5)),
            "Assignment Coverage": 1.0,
            "NMI": float(normalized_mutual_info_score(discovery_y_true, cluster_labels)),
            "ARI": float(adjusted_rand_score(discovery_y_true, cluster_labels)),
            "Hungarian Acc": hungarian_acc(discovery_y_true, cluster_labels),
            "Cluster Size CV": float(np.std(counts) / (np.mean(counts) + 1e-8)),
            **consensus_diag,
        })

    inc_df = pd.DataFrame(incremental_rows)
    clu_df = pd.DataFrame(clustering_rows)
    inc_df.to_csv(save_dir / "incremental_results.csv", index=False, encoding="utf-8-sig")
    clu_df.to_csv(save_dir / "clustering_results.csv", index=False, encoding="utf-8-sig")
    r3 = inc_df[inc_df["Stage"] == "After R3"].iloc[-1].to_dict()
    gate = (
        float(r3["Overall Acc"]) >= STAGE48_MULTISEED["overall"] + 0.01
        and float(r3["Old Acc"]) >= STAGE48_MULTISEED["old"] - 0.02
        and float(r3["New Acc"]) >= STAGE48_MULTISEED["new"] - 0.05
        and float(r3["Forgetting Rate"]) <= STAGE48_MULTISEED["forgetting"] + 0.05
    )
    summary = {
        "schema_version": "stage60_lora_cumulative_pseudolabel_retrain_v1",
        "dataset_path": str(args.dataset_path),
        "save_dir": str(save_dir),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "recording_consensus_threshold": float(args.recording_consensus_threshold),
        "device": device,
        "r3": {
            "overall": float(r3["Overall Acc"]),
            "old": float(r3["Old Acc"]),
            "new": float(r3["New Acc"]),
            "forgetting": float(r3["Forgetting Rate"]),
            "macro_f1": float(r3["Macro F1"]),
        },
        "stage48_multiseed": STAGE48_MULTISEED,
        "gate": "PASS" if gate else "FAIL",
    }
    (save_dir / "stage60_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Stage60 LoRa 累积伪标签重训 Seed7 报告",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 seed7 门槛，可进入三种子。' if gate else '未通过 seed7 门槛，不扩三种子。'}",
        f"- R3 Overall/Old/New/Forgetting = {summary['r3']['overall']:.4f}/{summary['r3']['old']:.4f}/{summary['r3']['new']:.4f}/{summary['r3']['forgetting']:.4f}。",
        "- 本阶段每轮累积全部已见真/伪标签样本重新训练 LoRaChirp 分类器，用来测试绕开 RADCIL 后端是否能改善 LoRa。",
        "",
        "## R3 对比",
        "",
        "| Method | Overall | Old | New | Forgetting | Gate |",
        "|---|---:|---:|---:|---:|---|",
        f"| Stage60 cumulative retrain | {summary['r3']['overall']:.4f} | {summary['r3']['old']:.4f} | {summary['r3']['new']:.4f} | {summary['r3']['forgetting']:.4f} | {summary['gate']} |",
        f"| Stage48 multiseed | {STAGE48_MULTISEED['overall']:.4f} | {STAGE48_MULTISEED['old']:.4f} | {STAGE48_MULTISEED['new']:.4f} | {STAGE48_MULTISEED['forgetting']:.4f} | baseline |",
        "",
    ]
    (save_dir / "STAGE60_LORA_CUMULATIVE_PSEUDOLABEL_RETRAIN_SEED7_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage60 LoRa 累积伪标签全量重训")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--test-batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--recording-consensus-threshold", type=float, default=0.65)
    parser.add_argument("--use-supcon", action="store_true")
    parser.add_argument("--supcon-weight", type=float, default=0.1)
    parser.add_argument("--supcon-temperature", type=float, default=0.2)
    parser.add_argument("--disable-rf-augmentation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
