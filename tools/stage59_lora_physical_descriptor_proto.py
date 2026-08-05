#!/usr/bin/env python3
"""Stage59 LoRa 物理描述符发现 + 原型分类快速验证。

本阶段刻意绕开 RADCIL 增量分类头，验证 LoRa 低分是否主要来自神经
增量后端的旧/新类权衡。流程只使用 strict 协议允许的数据：

1. Day1 IQ_1-6 已知训练样本提供旧类原型；
2. 每轮 Day2-4 IQ_1-7 discovery 样本提取物理描述符，并用固定 K=5
   KMeans 发现新伪类；
3. held-out IQ_8-10 只在所有聚类、原型和标准化完成后做最终评估；
4. discovery 真值仅用于离线 Hungarian 报告映射，不参与训练或选参。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lora25_strict_loader import load_lora25_diffdays_3round


STAGE48_MULTISEED = {
    "overall": 0.2803,
    "old": 0.2312,
    "new": 0.4770,
    "forgetting": 0.1802,
}


def _safe_stats(values: np.ndarray) -> list[float]:
    """提取一维序列的稳定统计量，覆盖幅度、相位差和频域包络。"""
    values = np.asarray(values, dtype=np.float32)
    q10, q25, q50, q75, q90 = np.percentile(values, [10, 25, 50, 75, 90])
    mean = float(np.mean(values))
    std = float(np.std(values) + 1e-8)
    centered = values - mean
    skew = float(np.mean(centered**3) / (std**3 + 1e-8))
    kurt = float(np.mean(centered**4) / (std**4 + 1e-8))
    return [mean, std, float(q10), float(q25), float(q50), float(q75), float(q90), skew, kurt]


def _pooled_log_spectrum(iq_complex: np.ndarray, bins: int = 32) -> np.ndarray:
    """把复数 IQ 的 log spectrum 均匀池化成固定维度频域指纹。"""
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft(iq_complex))))
    chunks = np.array_split(spectrum.astype(np.float32), int(bins))
    return np.asarray([float(np.mean(chunk)) for chunk in chunks], dtype=np.float32)


def extract_physical_descriptors(x: np.ndarray, spectrum_bins: int = 32) -> np.ndarray:
    """从 [N, 2, L] LoRa I/Q 中提取轻量物理描述符。

    特征只来自单个样本自身，不依赖标签或评估集统计。描述符包含 I/Q、
    幅度、相位差、瞬时频率稳定性和池化频谱，用来测试传统 RF fingerprint
    信息是否能绕开神经增量头的旧/新类冲突。
    """
    x = np.asarray(x, dtype=np.float32)
    rows: list[np.ndarray] = []
    for sample in x:
        i_sig = sample[0]
        q_sig = sample[1]
        complex_sig = i_sig.astype(np.float32) + 1j * q_sig.astype(np.float32)
        magnitude = np.sqrt(i_sig**2 + q_sig**2 + 1e-8)
        phase = np.unwrap(np.angle(complex_sig))
        phase_delta = np.diff(phase, prepend=phase[:1])
        iq_cross = i_sig * q_sig
        envelope_ratio = magnitude / (np.mean(magnitude) + 1e-8)
        stats = []
        for seq in (i_sig, q_sig, magnitude, phase_delta, iq_cross, envelope_ratio):
            stats.extend(_safe_stats(seq))
        stats.append(float(np.mean(np.abs(np.diff(magnitude)))))
        stats.append(float(np.mean(np.abs(np.diff(phase_delta)))))
        spectrum = _pooled_log_spectrum(complex_sig, bins=spectrum_bins)
        rows.append(np.concatenate([np.asarray(stats, dtype=np.float32), spectrum], axis=0))
    return np.vstack(rows).astype(np.float32)


def _l2norm(z: np.ndarray) -> np.ndarray:
    """行级 L2 归一化，供 cosine nearest-prototype 使用。"""
    z = np.nan_to_num(np.asarray(z, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def _build_prototypes(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按伪标签/真标签计算单原型。"""
    prototypes, proto_labels = [], []
    for label in sorted(np.unique(labels).astype(int).tolist()):
        prototypes.append(np.mean(features[labels == label], axis=0))
        proto_labels.append(label)
    return _l2norm(np.vstack(prototypes)), np.asarray(proto_labels, dtype=np.int64)


def _predict_proto(features: np.ndarray, prototypes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """使用 cosine 最近原型输出 raw pseudo-label。"""
    scores = _l2norm(features) @ prototypes.T
    return labels[np.argmax(scores, axis=1)].astype(np.int64)


def _hungarian_acc(y_true: np.ndarray, cluster_labels: np.ndarray) -> float:
    """计算聚类 Hungarian accuracy，作为 discovery 质量诊断。"""
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


def _pseudo_mapping(y_true: np.ndarray, cluster_labels: np.ndarray, next_label: int) -> tuple[dict[int, int], np.ndarray]:
    """把任意 cluster id 编成连续伪标签，并生成仅报告用 Hungarian 映射。"""
    cluster_ids = sorted(np.unique(cluster_labels).astype(int).tolist())
    pseudo_for_cluster = {cid: int(next_label + idx) for idx, cid in enumerate(cluster_ids)}
    pseudo_y = np.asarray([pseudo_for_cluster[int(cid)] for cid in cluster_labels], dtype=np.int64)

    true_ids = sorted(np.unique(y_true).astype(int).tolist())
    contingency = np.zeros((len(true_ids), len(cluster_ids)), dtype=np.int64)
    true_pos = {int(v): i for i, v in enumerate(true_ids)}
    for col, cid in enumerate(cluster_ids):
        mask = cluster_labels == cid
        for value, count in zip(*np.unique(y_true[mask], return_counts=True)):
            contingency[true_pos[int(value)], col] = int(count)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {pseudo_for_cluster[cid]: -(1_000_000 + pseudo_for_cluster[cid]) for cid in cluster_ids}
    for row, col in zip(row_ind, col_ind):
        mapping[pseudo_for_cluster[cluster_ids[int(col)]]] = int(true_ids[int(row)])
    return mapping, pseudo_y


def _recording_consensus(labels: np.ndarray, recording_id: np.ndarray, threshold: float) -> tuple[np.ndarray, dict[str, float]]:
    """对 discovery 聚类标签做 recording 内多数共识修正。"""
    labels = np.asarray(labels, dtype=np.int64).copy()
    recording_id = np.asarray(recording_id)
    changed = 0
    accepted = 0
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
        "recording_groups": int(np.unique(recording_id).size),
        "recording_consensus_accepted": int(accepted),
        "recording_consensus_changed": int(changed),
        "recording_consensus_mean": float(np.mean(consistencies)) if consistencies else 0.0,
    }


def _acc_range(y_true: np.ndarray, y_pred: np.ndarray, start: int, end: int) -> float:
    """计算类别区间准确率。"""
    mask = (y_true >= int(start)) & (y_true < int(end))
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_true[mask] == y_pred[mask]))


def _evaluate(stage: str, y_true: np.ndarray, pred_raw: np.ndarray, mapping: dict[int, int], initial_ref: float | None) -> dict[str, float | str]:
    """按主实验 old/new/forgetting 定义计算指标。"""
    if stage == "Initial":
        seen_classes = 10
        old_start, old_end = 0, 10
        new_start, new_end = None, None
    else:
        round_index = int(stage.replace("After R", ""))
        seen_classes = 10 + round_index * 5
        old_start, old_end = 0, 10 + (round_index - 1) * 5
        new_start, new_end = 10 + (round_index - 1) * 5, 10 + round_index * 5
    seen_mask = y_true < seen_classes
    y_seen = y_true[seen_mask]
    pred_seen = np.asarray([mapping.get(int(p), int(p)) for p in pred_raw[seen_mask]], dtype=np.int64)
    initial_acc = _acc_range(y_seen, pred_seen, 0, 10)
    forgetting = 0.0 if initial_ref is None else float(initial_ref - initial_acc)
    return {
        "Method": "LoRa physical-descriptor proto",
        "Stage": stage,
        "Seen Classes": int(seen_classes),
        "Eval Samples": int(len(y_seen)),
        "Overall Acc": float(np.mean(y_seen == pred_seen)),
        "Old Acc": _acc_range(y_seen, pred_seen, old_start, old_end),
        "New Acc": float("nan") if new_start is None else _acc_range(y_seen, pred_seen, new_start, new_end),
        "Initial Known Acc": initial_acc,
        "Forgetting Rate": forgetting,
        "Macro F1": float(f1_score(y_seen, pred_seen, average="macro", zero_division=0)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """执行 Stage59 seed7 物理描述符链路。"""
    splits = load_lora25_diffdays_3round(args.dataset_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 先一次性抽取所有 split 的样本级物理描述符；PCA/Scaler 仍在每轮只用
    # 已知 train + 当前 discovery 拟合，避免评估集统计泄漏。
    feature_cache = {
        key: extract_physical_descriptors(value["X"], spectrum_bins=args.spectrum_bins)
        for key, value in splits.items()
        if isinstance(value, dict) and "X" in value
    }

    enrolled_raw_features = [feature_cache["day1_backbone_train"]]
    enrolled_labels = [np.asarray(splits["day1_backbone_train"]["y"], dtype=np.int64)]
    pseudo_to_true = {int(label): int(label) for label in range(10)}
    next_label = 10
    clustering_rows = []
    incremental_rows = []

    def transform_round(train_keys: list[str], eval_key: str) -> tuple[np.ndarray, np.ndarray]:
        fit_features = np.vstack([feature_cache[key] for key in train_keys])
        scaler = StandardScaler().fit(fit_features)
        transformed_fit = scaler.transform(fit_features)
        n_components = min(int(args.pca_dim), transformed_fit.shape[1], max(2, transformed_fit.shape[0] - 1))
        pca = PCA(n_components=n_components, random_state=args.seed).fit(transformed_fit)
        return pca.transform(scaler.transform(fit_features)), pca.transform(scaler.transform(feature_cache[eval_key]))

    # Initial：只用 Day1 backbone train 拟合标准化/PCA，再评估 Day1 held-out。
    _, eval_initial_z = transform_round(["day1_backbone_train"], "day1_initial_eval")
    train_initial_z, _ = transform_round(["day1_backbone_train"], "day1_initial_eval")
    prototypes, proto_labels = _build_prototypes(train_initial_z, enrolled_labels[0])
    initial_pred = _predict_proto(eval_initial_z, prototypes, proto_labels)
    initial_row = _evaluate("Initial", splits["day1_initial_eval"]["y"], initial_pred, pseudo_to_true, None)
    initial_ref = float(initial_row["Initial Known Acc"])
    incremental_rows.append(initial_row)

    for round_index in range(1, 4):
        discovery_key = f"day{round_index + 1}_unknown_round{round_index}"
        eval_key = f"day{round_index + 1}_eval_after_r{round_index}"
        train_keys = ["day1_backbone_train"] + [f"day{i + 1}_unknown_round{i}" for i in range(1, round_index + 1)]
        fit_z, eval_z = transform_round(train_keys, eval_key)

        # fit_z 按 train_keys 顺序拼接，最后一段就是当前轮 discovery。
        discovery_count = len(splits[discovery_key]["y"])
        discovery_z = fit_z[-discovery_count:]
        cluster_labels = KMeans(n_clusters=5, n_init=20, random_state=args.seed + round_index).fit_predict(discovery_z)
        consensus_diag = {}
        if args.recording_consensus_threshold > 0:
            cluster_labels, consensus_diag = _recording_consensus(
                cluster_labels,
                splits[discovery_key]["recording_id"],
                args.recording_consensus_threshold,
            )
        mapping, pseudo_y = _pseudo_mapping(splits[discovery_key]["y"], cluster_labels, next_label)
        pseudo_to_true.update(mapping)
        next_label += len(np.unique(cluster_labels))

        enrolled_raw_features.append(feature_cache[discovery_key])
        enrolled_labels.append(pseudo_y)
        # 使用当轮 scaler/PCA 重新投影所有已注册样本，避免不同轮特征空间混用。
        enrolled_z = fit_z
        enrolled_y = np.concatenate(enrolled_labels)
        prototypes, proto_labels = _build_prototypes(enrolled_z, enrolled_y)
        pred = _predict_proto(eval_z, prototypes, proto_labels)
        incremental_rows.append(_evaluate(f"After R{round_index}", splits[eval_key]["y"], pred, pseudo_to_true, initial_ref))

        y_disc = np.asarray(splits[discovery_key]["y"], dtype=np.int64)
        values, counts = np.unique(cluster_labels, return_counts=True)
        clustering_rows.append({
            "Method": "physical-descriptor KMeans",
            "Round": f"R{round_index}",
            "True New Classes": 5,
            "Samples": int(len(y_disc)),
            "Final Cluster Count": int(len(values)),
            "Cluster Count Error": int(abs(len(values) - 5)),
            "Assignment Coverage": 1.0,
            "NMI": float(normalized_mutual_info_score(y_disc, cluster_labels)),
            "ARI": float(adjusted_rand_score(y_disc, cluster_labels)),
            "Hungarian Acc": _hungarian_acc(y_disc, cluster_labels),
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
    )
    summary = {
        "schema_version": "stage59_lora_physical_descriptor_proto_v1",
        "dataset_path": str(args.dataset_path),
        "save_dir": str(save_dir),
        "seed": int(args.seed),
        "pca_dim": int(args.pca_dim),
        "spectrum_bins": int(args.spectrum_bins),
        "recording_consensus_threshold": float(args.recording_consensus_threshold),
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
    (save_dir / "stage59_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Stage59 LoRa 物理描述符原型 Seed7 报告",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 seed7 门槛，可进入三种子。' if gate else '未通过 seed7 门槛，不扩三种子。'}",
        f"- R3 Overall/Old/New/Forgetting = {summary['r3']['overall']:.4f}/{summary['r3']['old']:.4f}/{summary['r3']['new']:.4f}/{summary['r3']['forgetting']:.4f}。",
        "- 本阶段不使用 HDBSCAN 或 RADCIL，只验证 LoRa 物理描述符 + 固定 K 原型链路是否能突破旧/新类跷跷板。",
        "",
        "## R3 对比",
        "",
        "| Method | Overall | Old | New | Forgetting | Gate |",
        "|---|---:|---:|---:|---:|---|",
        f"| Stage59 physical proto | {summary['r3']['overall']:.4f} | {summary['r3']['old']:.4f} | {summary['r3']['new']:.4f} | {summary['r3']['forgetting']:.4f} | {summary['gate']} |",
        f"| Stage48 multiseed | {STAGE48_MULTISEED['overall']:.4f} | {STAGE48_MULTISEED['old']:.4f} | {STAGE48_MULTISEED['new']:.4f} | {STAGE48_MULTISEED['forgetting']:.4f} | baseline |",
        "",
    ]
    (save_dir / "STAGE59_LORA_PHYSICAL_DESCRIPTOR_PROTO_SEED7_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage59 LoRa 物理描述符原型链路")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--spectrum-bins", type=int, default=32)
    parser.add_argument("--recording-consensus-threshold", type=float, default=0.65)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
