"""阶段 0 strict loader 完整性审计工具。

本脚本只做数据加载、形状统计、类别统计和协议边界检查，不执行训练、
调参、聚类或评估选择。它的目标是把 WiSig 与 ADS-B strict loader 的
输入输出事实固化为 JSON，便于后续进入算法阶段前确认：

1. 数据路径来自命令行或配置文件，避免继续依赖开发者机器绝对路径。
2. 各 split 的形状、dtype、样本数和类别集合符合实施计划 1.1。
3. discovery/enrollment split 与 held-out evaluation split 的标签边界清晰。

注意：WiSig 与 ADS-B loader 会按当前实现把目标 split 转为内存数组；
完整数据审计建议在 Slurm 服务器运行，不建议在磁盘或内存紧张的本地机器运行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _add_project_to_path(project_root: Path) -> None:
    """确保从任意工作目录运行时都能导入项目内 datasets 模块。"""

    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _read_config(path: Path | None) -> dict[str, Any]:
    """读取可选 JSON 路径配置；未提供时返回空配置。"""

    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(project_root: Path, value: str | None) -> Path | None:
    """把配置中的相对路径解析到项目根目录下，绝对路径保持不变。"""

    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _split_summary(name: str, item: dict[str, Any]) -> dict[str, Any]:
    """提取单个 loader split 的审计摘要，避免把数组内容写入报告。"""

    import numpy as np

    x = np.asarray(item["X"])
    y = np.asarray(item["y"])
    labels, counts = np.unique(y, return_counts=True)
    return {
        "name": name,
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "y_shape": list(y.shape),
        "y_dtype": str(y.dtype),
        "sample_count": int(len(y)),
        "class_count": int(len(labels)),
        "label_min": int(labels.min()) if len(labels) else None,
        "label_max": int(labels.max()) if len(labels) else None,
        "labels": [int(label) for label in labels.tolist()],
        "per_class_min": int(counts.min()) if len(counts) else 0,
        "per_class_max": int(counts.max()) if len(counts) else 0,
        "per_class_unique_counts": sorted({int(count) for count in counts.tolist()}),
        "day": item.get("day"),
        "split": item.get("split"),
    }


def _audit_wisig(project_root: Path, wisig_pkl: Path | None) -> dict[str, Any]:
    """运行 WiSig 10+10x3 loader，并检查 70/30 split 与类别边界。"""

    import numpy as np

    if wisig_pkl is None:
        wisig_pkl = project_root / "数据集" / "WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl"
    result: dict[str, Any] = {"path": str(wisig_pkl), "exists": wisig_pkl.exists()}
    if not wisig_pkl.exists():
        result["ok"] = False
        result["error"] = "WiSig PKL 不存在，无法执行 strict loader 审计。"
        return result

    from datasets.wisig_crossday_incremental_loader import load_wisig_crossday_incremental

    splits = load_wisig_crossday_incremental(dataset_path=str(wisig_pkl), train_ratio=0.70)
    split_names = [name for name, item in splits.items() if isinstance(item, dict) and "X" in item]
    summaries = {name: _split_summary(name, splits[name]) for name in split_names}

    expected_classes = {
        "day1_known_train": list(range(0, 10)),
        "day1_initial_eval": list(range(0, 10)),
        "day2_unknown_round1": list(range(10, 20)),
        "day2_eval_after_r1": list(range(0, 20)),
        "day3_unknown_round2": list(range(20, 30)),
        "day3_eval_after_r2": list(range(0, 30)),
        "day4_unknown_round3": list(range(30, 40)),
        "day4_eval_after_r3": list(range(0, 40)),
    }
    class_checks = {
        name: summaries[name]["labels"] == expected
        for name, expected in expected_classes.items()
        if name in summaries
    }
    shape_checks = {
        name: summary["x_shape"][1:] == [2, 256] and summary["x_shape"][0] == summary["sample_count"]
        for name, summary in summaries.items()
    }

    result.update(
        {
            "ok": bool(all(class_checks.values()) and all(shape_checks.values())),
            "protocol": "WiSig Cross-Day 10 + 10x3, stored-order 70/30 per Tx/Rx/day",
            "splits": summaries,
            "class_checks": class_checks,
            "shape_checks": shape_checks,
            "train_ratio": float(splits.get("train_ratio", 0.70)),
        }
    )
    return result


def _audit_adsb(project_root: Path, adsb_root: Path | None) -> dict[str, Any]:
    """运行 ADS-B 90+10x3 strict loader，并检查文件解压与类别边界。"""

    candidates = []
    if adsb_root is not None:
        candidates.append(adsb_root)
    candidates.extend(
        [
            project_root / "数据集" / "ADS-B" / "Dataset",
            project_root / "数据集" / "Dataset",
        ]
    )
    base = next((candidate for candidate in candidates if candidate.exists()), None)
    result: dict[str, Any] = {
        "requested_path": str(adsb_root) if adsb_root else None,
        "resolved_path": str(base) if base else None,
        "exists": base is not None,
    }
    if base is None:
        result["ok"] = False
        result["error"] = "ADS-B 解压目录不存在，需先解压 ADS-B.rar 后再执行 strict loader 审计。"
        return result

    from datasets.adsb_90known_strict_loader import load_adsb_90known_3round

    splits = load_adsb_90known_3round(data_root=str(base))
    split_names = [name for name, item in splits.items() if isinstance(item, dict) and "X" in item]
    summaries = {name: _split_summary(name, splits[name]) for name in split_names}
    expected_class_counts = {
        "day1_known_train": 90,
        "day1_initial_eval": 90,
        "day2_unknown_round1": 10,
        "day2_eval_after_r1": 100,
        "day3_unknown_round2": 10,
        "day3_eval_after_r2": 110,
        "day4_unknown_round3": 10,
        "day4_eval_after_r3": 120,
    }
    class_count_checks = {
        name: summaries[name]["class_count"] == expected
        for name, expected in expected_class_counts.items()
        if name in summaries
    }
    shape_checks = {
        name: summary["x_shape"][1] == 2 and summary["x_shape"][0] == summary["sample_count"]
        for name, summary in summaries.items()
    }

    result.update(
        {
            "ok": bool(all(class_count_checks.values()) and all(shape_checks.values())),
            "protocol": splits.get("protocol_version"),
            "normalization": splits.get("normalization"),
            "known_split_audit": splits.get("known_split_audit"),
            "round_counts": splits.get("round_counts"),
            "splits": summaries,
            "class_count_checks": class_count_checks,
            "shape_checks": shape_checks,
        }
    )
    return result


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    """根据命令行和配置文件构建完整阶段 0 loader 审计报告。"""

    config = _read_config(Path(args.config).resolve() if args.config else None)
    project_root = _resolve_path(Path.cwd(), args.project_root or config.get("project_root")) or Path.cwd()
    project_root = project_root.resolve()
    _add_project_to_path(project_root)

    data_root = _resolve_path(project_root, args.data_root or config.get("data_root"))
    wisig_pkl = _resolve_path(project_root, args.wisig_pkl or config.get("wisig_pkl"))
    adsb_root = _resolve_path(project_root, args.adsb_root or config.get("adsb_root"))
    if data_root is not None:
        wisig_pkl = wisig_pkl or data_root / "WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl"
        adsb_root = adsb_root or data_root / "ADS-B" / "Dataset"

    report = {
        "project_root": str(project_root),
        "data_root": str(data_root) if data_root else None,
        "config": str(Path(args.config).resolve()) if args.config else None,
        "wisig": None,
        "adsb": None,
    }
    if args.dataset in {"wisig", "all"}:
        report["wisig"] = _audit_wisig(project_root, wisig_pkl)
    if args.dataset in {"adsb", "all"}:
        report["adsb"] = _audit_adsb(project_root, adsb_root)

    checked = [value for value in (report["wisig"], report["adsb"]) if isinstance(value, dict)]
    report["ok"] = bool(checked and all(item.get("ok") for item in checked))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 0 WiSig/ADS-B strict loader 审计。")
    parser.add_argument("--config", help="可选 JSON 路径配置，例如 configs/data_paths.local.json。")
    parser.add_argument("--project-root", help="项目根目录；未提供时使用当前工作目录或配置文件。")
    parser.add_argument("--data-root", help="数据根目录；相对路径会基于项目根目录解析。")
    parser.add_argument("--wisig-pkl", help="WiSig 完整 PKL 路径；相对路径会基于项目根目录解析。")
    parser.add_argument("--adsb-root", help="ADS-B 解压后的 Dataset 目录；相对路径会基于项目根目录解析。")
    parser.add_argument("--dataset", choices=["wisig", "adsb", "all"], default="all", help="选择要审计的数据集。")
    parser.add_argument("--output", help="保存 JSON 报告的路径；未提供时只打印到 stdout。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_audit(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
