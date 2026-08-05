#!/usr/bin/env python3
"""汇总 Stage58 LoRa 双模型后验路由 seed7 结果。

Stage58 不再重做相邻训练小机制，而是把 Stage57 暴露出的两个互补信号
拆开验证：base 模型相对稳旧类，top80 refined-filter 模型更偏新类吸收。
本报告器只读取显式转储的 held-out logits / raw pred / pseudo_to_true 映射，
按预注册的置信度路由规则离线组合预测；组合规则不读取评估真值做选参。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


STAGE48_SEED7 = {
    "overall": 0.2809523810,
    "old": 0.2220238095,
    "new": 0.5166666667,
    "forgetting": 0.1726190476,
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """稳定计算 softmax，避免离线读取 logits 时出现溢出。"""
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-8)


def _load_dump(save_dir: Path, stage: str) -> dict[str, np.ndarray]:
    """读取一个模型变体在某个阶段的预测转储。"""
    path = save_dir / "end_to_end_eval_dumps" / f"{stage}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval dump: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _mapping_from_dump(dump: dict[str, np.ndarray]) -> dict[int, int]:
    """把 NPZ 中的伪标签映射数组还原为字典。"""
    keys = dump["pseudo_to_true_keys"].astype(np.int64)
    values = dump["pseudo_to_true_values"].astype(np.int64)
    return {int(k): int(v) for k, v in zip(keys, values)}


def _map_prediction(raw_pred: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """将模型 raw pseudo-label 预测映射到报告用全局真实标签空间。"""
    return np.asarray([mapping.get(int(value), int(value)) for value in raw_pred], dtype=np.int64)


def _acc_on_range(y_true: np.ndarray, y_pred: np.ndarray, start: int, end: int) -> float:
    """计算指定类别区间准确率；无样本时返回 NaN。"""
    mask = (y_true >= int(start)) & (y_true < int(end))
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_true[mask] == y_pred[mask]))


def _evaluate(stage_label: str, y_true: np.ndarray, y_pred: np.ndarray, initial_reference_acc: float | None) -> dict[str, float | str]:
    """按项目主入口一致的 old/new 定义计算 symbol-level 指标。"""
    if stage_label == "initial":
        seen_classes = 10
        old_start, old_end = 0, 10
        new_start, new_end = None, None
    else:
        round_index = int(stage_label.replace("after_r", ""))
        seen_classes = 10 + round_index * 5
        old_start, old_end = 0, 10 + (round_index - 1) * 5
        new_start, new_end = 10 + (round_index - 1) * 5, 10 + round_index * 5

    seen_mask = y_true < seen_classes
    y_seen = y_true[seen_mask]
    pred_seen = y_pred[seen_mask]
    initial_known_acc = _acc_on_range(y_seen, pred_seen, 0, 10)
    if initial_reference_acc is None or np.isnan(initial_known_acc):
        forgetting = 0.0 if stage_label == "initial" else float("nan")
    else:
        forgetting = float(initial_reference_acc - initial_known_acc)
    return {
        "stage": "Initial" if stage_label == "initial" else f"After R{stage_label[-1]}",
        "overall": float(np.mean(y_seen == pred_seen)),
        "old": _acc_on_range(y_seen, pred_seen, old_start, old_end),
        "new": float("nan") if new_start is None else _acc_on_range(y_seen, pred_seen, new_start, new_end),
        "initial_known": initial_known_acc,
        "forgetting": forgetting,
        "macro_f1": float(f1_score(y_seen, pred_seen, average="macro", zero_division=0)),
    }


def _route_predictions(
    base: dict[str, np.ndarray],
    top80: dict[str, np.ndarray],
    stage_label: str,
    rule: str,
) -> np.ndarray:
    """执行预注册双模型路由，输出映射后的全局标签预测。

    路由只看两个模型自身的 raw pseudo-label 与 softmax confidence：
    - ``base`` / ``top80`` 直接使用单模型预测；
    - ``new_conf_delta_0p00/0p05/0p10`` 只在 top80 预测当前轮新伪类且
      置信度不明显低于 base 时切到 top80，否则保留 base。
    """
    base_map = _mapping_from_dump(base)
    top80_map = _mapping_from_dump(top80)
    base_raw = base["pred_raw"].astype(np.int64)
    top80_raw = top80["pred_raw"].astype(np.int64)
    if rule == "base":
        return _map_prediction(base_raw, base_map)
    if rule == "top80":
        return _map_prediction(top80_raw, top80_map)

    if stage_label == "initial":
        return _map_prediction(base_raw, base_map)
    round_index = int(stage_label.replace("after_r", ""))
    old_head_count = 10 + (round_index - 1) * 5
    margin_text = rule.replace("new_conf_delta_", "").replace("p", ".")
    margin = float(margin_text)
    base_prob = _softmax(base["logits"])
    top80_prob = _softmax(top80["logits"])
    base_conf = np.max(base_prob, axis=1)
    top80_conf = np.max(top80_prob, axis=1)
    use_top80 = (top80_raw >= old_head_count) & (top80_conf + margin >= base_conf)
    routed = _map_prediction(base_raw, base_map)
    routed[use_top80] = _map_prediction(top80_raw, top80_map)[use_top80]
    return routed


def collect_records(root: Path, job_id: str, variants: dict[str, str]) -> list[dict[str, float | str]]:
    """收集 base、top80 和双模型路由的逐阶段指标。"""
    save_dirs = {name: root / f"lora_s58_{suffix}_{job_id}" for name, suffix in variants.items()}
    stages = ["initial", "after_r1", "after_r2", "after_r3"]
    dumps = {
        name: {stage: _load_dump(save_dir, stage) for stage in stages}
        for name, save_dir in save_dirs.items()
    }
    rules = ["base", "top80", "new_conf_delta_0p00", "new_conf_delta_0p05", "new_conf_delta_0p10"]
    records: list[dict[str, float | str]] = []
    for rule in rules:
        initial_pred = _route_predictions(dumps["base"]["initial"], dumps["top80"]["initial"], "initial", rule)
        initial_metrics = _evaluate("initial", dumps["base"]["initial"]["y_true"].astype(np.int64), initial_pred, None)
        initial_ref = float(initial_metrics["initial_known"])
        records.append({"variant": rule, **initial_metrics})
        for stage in stages[1:]:
            pred = _route_predictions(dumps["base"][stage], dumps["top80"][stage], stage, rule)
            metrics = _evaluate(stage, dumps["base"][stage]["y_true"].astype(np.int64), pred, initial_ref)
            records.append({"variant": rule, **metrics})
    return records


def _r3(records: list[dict[str, float | str]], variant: str) -> dict[str, float | str]:
    """取指定变体 R3 行。"""
    rows = [item for item in records if item["variant"] == variant and item["stage"] == "After R3"]
    if not rows:
        raise ValueError(f"Missing R3 row for {variant}")
    return rows[-1]


def _passes_gate(row: dict[str, float | str]) -> bool:
    """Stage58 seed7 门槛：必须涨 Overall，同时不牺牲 New 和遗忘。"""
    if row["variant"] in {"base", "top80"}:
        return False
    return (
        float(row["overall"]) >= STAGE48_SEED7["overall"] + 0.01
        and float(row["old"]) >= STAGE48_SEED7["old"] - 0.01
        and float(row["new"]) >= STAGE48_SEED7["new"] - 0.02
        and float(row["forgetting"]) <= STAGE48_SEED7["forgetting"] + 0.05
    )


def _table(records: list[dict[str, float | str]]) -> str:
    """生成 R3 指标表。"""
    lines = [
        "| Variant | Overall | Old | New | Forgetting | ΔOverall vs Stage48 | ΔNew vs Stage48 | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in ["base", "top80", "new_conf_delta_0p00", "new_conf_delta_0p05", "new_conf_delta_0p10"]:
        row = _r3(records, variant)
        lines.append(
            "| {variant} | {overall:.4f} | {old:.4f} | {new:.4f} | {forgetting:.4f} | {do:+.4f} | {dn:+.4f} | {gate} |".format(
                variant=variant,
                overall=float(row["overall"]),
                old=float(row["old"]),
                new=float(row["new"]),
                forgetting=float(row["forgetting"]),
                do=float(row["overall"]) - STAGE48_SEED7["overall"],
                dn=float(row["new"]) - STAGE48_SEED7["new"],
                gate="PASS" if _passes_gate(row) else "FAIL",
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage58 LoRa 双模型后验路由报告器")
    parser.add_argument("--root", default="results/stage58")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = collect_records(root, str(args.job_id), {"base": "base", "top80": "filter_top080"})
    r3_rows = [_r3(records, variant) for variant in ["new_conf_delta_0p00", "new_conf_delta_0p05", "new_conf_delta_0p10"]]
    passed = [str(row["variant"]) for row in r3_rows if _passes_gate(row)]
    best = max([_r3(records, variant) for variant in ["base", "top80", "new_conf_delta_0p00", "new_conf_delta_0p05", "new_conf_delta_0p10"]], key=lambda row: float(row["overall"]))

    lines = [
        f"# Stage58 LoRa 双模型路由 Seed7 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 seed7 门槛，可固定路由规则后扩三种子。' if passed else '未通过 seed7 门槛，不扩三种子。'}",
        f"- 最佳 R3 Overall：{best['variant']}，Overall={float(best['overall']):.4f}，Old={float(best['old']):.4f}，New={float(best['new']):.4f}。",
        "- 路由只读取模型 raw pseudo-label 与 confidence；held-out 真值只用于本报告最终统计。",
        "",
        "## R3 指标",
        "",
        _table(records),
        "",
        "## 判定规则",
        "",
        "- 固定 Stage48/57 raw s28 + recording-consensus 0.65 + Chirp + joint discovery-CIL 配置。",
        "- base 与 top80 分别独立训练并保存 eval dumps；双模型路由不反向修改训练结果。",
        "- 通过门槛：R3 Overall 至少比 Stage48 seed7 高 0.01，Old 下降不超过 0.01，New 下降不超过 0.02，Forgetting 恶化不超过 0.05。",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage58_lora_dual_model_route_summary_v1",
                "job_id": str(args.job_id),
                "stage48_seed7": STAGE48_SEED7,
                "records": records,
                "passed_variants": passed,
                "gate": "PASS" if passed else "FAIL",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
