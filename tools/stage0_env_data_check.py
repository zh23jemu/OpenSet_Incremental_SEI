"""阶段 0 环境与数据自检工具。

这个脚本用于在本地或 Slurm 项目目录中快速确认三件事：
1. 训练依赖是否可以导入，尤其是 torch / hdbscan / umap 等主实验依赖。
2. CUDA 是否可用，并通过一个很小的矩阵乘法确认 torch 可以实际调用 GPU。
3. GitHub Release 同步下来的大数据文件是否存在、哈希是否匹配，紧凑数据是否可读取。

注意：脚本不会解压或删除任何数据文件，也不会读取完整大 PKL 到内存。对 ManyTx/ManyRx
压缩包只检查 ZIP 目录元信息；对 ADS-B 如果已解压，则读取 NPY 头部形状，否则只记录压缩包存在。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


# 已通过 GitHub Release 和 Slurm 端校验的原始数据 SHA-256。
EXPECTED_SHA256 = {
    "WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl": "63794418ba7ffaaa2198fabbbd5e3429ebd0f1f6346ac2c33db8c363cdd3195f",
    "ADS-B.rar": "4748fb7cf0cf0bc0da710e4057961ace831a384f9ca6931a7f839b6714fca2ad",
    "ManyRx.pkl.zip": "d2b23108c3f6f63a10ebbb149d7b08d6e1c1961cf5184926fbab452def3049de",
    "ManyTx.pkl.zip": "a8fc3e35134a240bfb4dab8862a6e482cef44de000b813d42417b853c47ccc7e",
}


REQUIRED_IMPORTS = {
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "sklearn",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "h5py": "h5py",
    "tqdm": "tqdm",
    "hdbscan": "hdbscan",
    "umap": "umap",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """以分块方式计算文件哈希，避免一次性读取大文件。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path, *, with_hash: bool) -> dict[str, Any]:
    """返回单个文件的存在性、大小和可选哈希检查结果。"""

    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return info

    info["size_bytes"] = path.stat().st_size
    expected = EXPECTED_SHA256.get(path.name)
    if with_hash and expected:
        actual = sha256_file(path)
        info["sha256"] = actual
        info["expected_sha256"] = expected
        info["sha256_ok"] = actual.lower() == expected.lower()
    return info


def inspect_zip(path: Path) -> dict[str, Any]:
    """读取 ZIP 中的文件目录，用于确认 ManyTx/ManyRx 压缩包结构。"""

    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            result.update(
                {
                    "zip_ok": True,
                    "member_count": len(members),
                    "members": [
                        {
                            "name": member.filename,
                            "compressed_size": member.compress_size,
                            "file_size": member.file_size,
                        }
                        for member in members[:10]
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001 - 自检脚本需要把错误转成报告。
        result.update({"zip_ok": False, "error": repr(exc)})
    return result


def inspect_npz(path: Path) -> dict[str, Any]:
    """读取 NPZ 元信息和数组形状，避免把所有数组强制复制到内存。"""

    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            result.update(
                {
                    "npz_ok": True,
                    "arrays": {
                        key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
                        for key in data.files
                    },
                }
            )
    except Exception as exc:  # noqa: BLE001 - 自检脚本需要继续输出其它检查项。
        result.update({"npz_ok": False, "error": repr(exc)})
    return result


def inspect_adsb_extracted(data_root: Path) -> dict[str, Any]:
    """如果 ADS-B 已解压，读取关键 NPY 文件头部形状；未解压时只报告状态。"""

    candidates = [data_root / "ADS-B" / "Dataset", data_root / "Dataset"]
    base = next((candidate for candidate in candidates if candidate.exists()), None)
    result: dict[str, Any] = {"extracted": base is not None, "base": str(base) if base else None}
    if base is None:
        return result

    required = [
        "X_train_90Class.npy",
        "Y_train_90Class.npy",
        "X_train_30Class.npy",
        "Y_train_30Class.npy",
        "X_test_30Class.npy",
        "Y_test_30Class.npy",
    ]
    try:
        import numpy as np

        files: dict[str, Any] = {}
        for name in required:
            path = base / name
            if not path.exists():
                files[name] = {"exists": False}
                continue
            arr = np.load(path, mmap_mode="r")
            files[name] = {"exists": True, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        result["files"] = files
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    return result


def check_imports() -> dict[str, Any]:
    """检查关键 Python 依赖是否可以导入，并记录版本。"""

    results: dict[str, Any] = {}
    for public_name, module_name in REQUIRED_IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
            results[public_name] = {
                "ok": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as exc:  # noqa: BLE001
            results[public_name] = {"ok": False, "error": repr(exc)}
    return results


def check_torch_cuda() -> dict[str, Any]:
    """检查 torch 与 CUDA，并运行一个很小的 GPU 张量计算。"""

    result: dict[str, Any] = {}
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_version"] = torch.version.cuda
        result["device_count"] = int(torch.cuda.device_count())
        result["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            x = torch.randn(128, 128, device=device)
            y = x @ x.T
            torch.cuda.synchronize(device)
            result["gpu_smoke_ok"] = bool(y.is_cuda and y.shape == (128, 128))
        else:
            result["gpu_smoke_ok"] = False
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    return result


def build_report(project_root: Path, data_root: Path, *, hash_large_files: bool) -> dict[str, Any]:
    """汇总阶段 0 自检报告。"""

    data_files = {
        "wisig": file_info(data_root / "WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl", with_hash=hash_large_files),
        "adsb": file_info(data_root / "ADS-B.rar", with_hash=hash_large_files),
        "manyrx": file_info(data_root / "ManyRx.pkl.zip", with_hash=hash_large_files),
        "manytx": file_info(data_root / "ManyTx.pkl.zip", with_hash=hash_large_files),
    }
    compact_files = {
        "lora25_compact": inspect_npz(project_root / "datasets" / "lora25_compact" / "lora25_diffdays_indoor_aligned_group_256.npz"),
        "manyrx_compact": inspect_npz(project_root / "datasets" / "manyrx_compact" / "manyrx_4known_3round_fixed_cross_rx.npz"),
    }
    report = {
        "python": sys.version,
        "project_root": str(project_root),
        "data_root": str(data_root),
        "imports": check_imports(),
        "torch_cuda": check_torch_cuda(),
        "data_files": data_files,
        "zip_files": {
            "manyrx": inspect_zip(data_root / "ManyRx.pkl.zip"),
            "manytx": inspect_zip(data_root / "ManyTx.pkl.zip"),
        },
        "compact_npz": compact_files,
        "adsb_extracted": inspect_adsb_extracted(data_root),
    }
    report["ok"] = summarize_ok(report)
    return report


def summarize_ok(report: dict[str, Any]) -> bool:
    """根据关键项给出总体验证是否通过。"""

    imports_ok = all(item.get("ok") for item in report["imports"].values())
    cuda_ok = bool(report["torch_cuda"].get("cuda_available")) and bool(report["torch_cuda"].get("gpu_smoke_ok"))
    files_ok = all(item.get("exists") for item in report["data_files"].values())
    hashes = [item.get("sha256_ok") for item in report["data_files"].values() if "sha256_ok" in item]
    hashes_ok = all(hashes) if hashes else True
    zip_ok = all(item.get("zip_ok") for item in report["zip_files"].values())
    compact_ok = all(item.get("npz_ok") for item in report["compact_npz"].values())
    return bool(imports_ok and cuda_ok and files_ok and hashes_ok and zip_ok and compact_ok)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 0 环境、依赖、数据和 CUDA 自检。")
    parser.add_argument("--project-root", default=".", help="项目根目录，默认当前目录。")
    parser.add_argument("--data-root", default="数据集", help="Release 数据目录，默认项目下的数据集目录。")
    parser.add_argument(
        "--hash-large-files",
        action="store_true",
        help="计算 WiSig、ADS-B、ManyTx、ManyRx 原始大文件 SHA-256，耗时较长但可验证同步完整性。",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出报告。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    report = build_report(project_root, data_root, hash_large_files=bool(args.hash_large_files))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("阶段 0 自检结果：" + ("通过" if report["ok"] else "未通过"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
