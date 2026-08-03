"""下载 LoRa RFFP Setup 1 strict 协议需要的原始 I/Q 文件。

这个脚本只面向当前项目已经固化的 Different Days Indoor strict 10+5x3
协议，不下载 LoRa 数据集的其它 setup 或其它场景。默认行为是生成可审计
manifest，不实际下载大文件；只有显式传入 ``--download`` 时才会把必要的
``IQ_*.dat`` 原始文件保存到本地目录。

设计要点：
* 复用 ``build_lora25_compact.py`` 中的设备顺序和 stage 划分，确保后续
  原始数据重切窗时不会改变现有实验协议。
* 下载范围限定为当前 strict split 会用到的 Day/Device/IQ 组合：
  Day1 已知类 IQ_1-10，Day2-4 discovery IQ_1-7 与 held-out eval IQ_8-10。
* 不读取未知真值做任何选择；这里的 label/stage 只来自公开实验协议和已有
  strict 划分，用于生成文件清单与目录结构。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from build_lora25_compact import BASE_URL, DEVICE_ORDER, make_records, source_url


DEFAULT_RAW_DIR = (
    Path(__file__).resolve().parents[1] / "raw_setup1_iq"
)


def raw_relative_path(record: dict[str, int | str]) -> Path:
    """返回官方目录结构风格的相对路径，方便人工核对和后续重切窗。

    例如：``Day1/Device1/IQ_1.dat``。这里不复用旧 compact cache 文件名，
    因为后续处理原始 I/Q 时保留原始层级更直观，也更容易和 OSU 链接对应。
    """

    return (
        Path(f"Day{int(record['day'])}")
        / f"Device{int(record['device'])}"
        / f"IQ_{int(record['transmission'])}.dat"
    )


def raw_output_path(raw_dir: Path, record: dict[str, int | str]) -> Path:
    """把一条协议记录映射到本地原始 I/Q 目标文件。"""

    return raw_dir / raw_relative_path(record)


def remote_size(url: str, timeout: int = 60) -> int | None:
    """尽量读取远端文件大小；服务器不支持 HEAD 时退回 Range 探测。

    OSU 静态服务器历史上支持 byte range，但 HEAD/Content-Length 在不同网络
    环境下可能不稳定。因此这里把 size 作为可选审计字段，不把它作为生成
    manifest 的硬依赖。
    """

    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length:
                return int(length)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        pass

    try:
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_range = response.headers.get("Content-Range")
            if content_range and "/" in content_range:
                return int(content_range.rsplit("/", maxsplit=1)[1])
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return None
    return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算 SHA-256，避免一次性把大型 I/Q 文件读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: Path, retries: int = 5) -> dict[str, object]:
    """下载单个原始 I/Q 文件，使用 ``.part`` 临时文件降低中断风险。

    如果下载中断，临时文件会保留在原地，便于人工排查网络问题；脚本不会
    删除任何已有文件。下载成功后再用原子 replace 落到最终路径。
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=300) as response:
                with part_path.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            part_path.replace(target)
            return {
                "downloaded": True,
                "bytes": int(target.stat().st_size),
                "sha256": sha256_file(target),
            }
        except Exception as exc:  # noqa: BLE001 - 记录网络/HTTP/文件系统综合错误
            last_error = exc
            time.sleep(min(60, 2 ** attempt))

    raise RuntimeError(f"download failed for {url}: {last_error}")


def build_manifest_records(
    raw_dir: Path,
    *,
    check_remote: bool,
    download: bool,
    limit: int | None,
) -> list[dict[str, object]]:
    """生成或下载当前 strict 协议需要的原始 I/Q 文件记录。"""

    records = make_records()
    if limit is not None:
        records = records[: int(limit)]

    manifest_records: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        url = source_url(record)
        target = raw_output_path(raw_dir, record)
        exists = target.exists()
        local_size = int(target.stat().st_size) if exists else None
        size = remote_size(url) if check_remote else None
        status = "exists" if exists else "planned"
        sha256 = sha256_file(target) if exists else None

        if download and not exists:
            result = download_file(url, target)
            exists = True
            local_size = int(result["bytes"])
            sha256 = str(result["sha256"])
            status = "downloaded"

        manifest_records.append(
            {
                "index": index,
                "stage": record["stage"],
                "day": int(record["day"]),
                "device": int(record["device"]),
                "transmission": int(record["transmission"]),
                "url": url,
                "relative_path": str(raw_relative_path(record)).replace("\\", "/"),
                "local_path": str(target.resolve()),
                "remote_size_bytes": size,
                "local_size_bytes": local_size,
                "sha256": sha256,
                "status": status,
            }
        )
        print(
            f"[{index:04d}/{len(records):04d}] {status}: "
            f"Day{record['day']} Device{record['device']} IQ_{record['transmission']}",
            flush=True,
        )

    return manifest_records


def stage_counts(records: Iterable[dict[str, object]]) -> dict[str, int]:
    """统计每个 strict stage 需要的原始 I/Q 文件数量。"""

    counts: dict[str, int] = {}
    for record in records:
        stage = str(record["stage"])
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备 LoRa RFFP Setup 1 Different Days Indoor 原始 I/Q 必要文件清单。"
    )
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RAW_DIR),
        help="原始 IQ 文件保存目录；默认位于 datasets/lora25_compact/raw_setup1_iq。",
    )
    parser.add_argument("--manifest", required=True, help="输出 JSON manifest 路径。")
    parser.add_argument(
        "--download",
        action="store_true",
        help="实际下载全部必要 IQ_*.dat 文件；不加该参数时只生成计划清单。",
    )
    parser.add_argument(
        "--check-remote",
        action="store_true",
        help="探测远端文件大小，用于确认 OSU 链接仍可访问。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 条记录，适合做链接/下载 smoke test。",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = build_manifest_records(
        raw_dir,
        check_remote=bool(args.check_remote),
        download=bool(args.download),
        limit=args.limit,
    )
    total_remote_known = sum(
        int(item["remote_size_bytes"] or 0) for item in records
    )
    missing_remote_size = sum(1 for item in records if item["remote_size_bytes"] is None)
    downloaded_count = sum(1 for item in records if item["status"] in {"exists", "downloaded"})

    manifest = {
        "dataset": "Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning",
        "subset": "LoRa RFFP Dataset - Different Days Indoor Scenario / Setup 1",
        "base_url": BASE_URL,
        "raw_dir": str(raw_dir),
        "device_order": DEVICE_ORDER,
        "protocol": "lora25_diffdays_indoor_strict_10known_5x3_v1",
        "download_requested": bool(args.download),
        "check_remote_requested": bool(args.check_remote),
        "record_limit": args.limit,
        "records_total": len(records),
        "records_present_or_downloaded": downloaded_count,
        "remote_size_known_bytes": total_remote_known,
        "remote_size_missing_records": missing_remote_size,
        "stage_counts": stage_counts(records),
        "notes": [
            "只覆盖当前 strict 协议需要的 Day/Device/IQ 原始文件，不下载其它 LoRa 场景。",
            "Day1 使用 IQ_1-7 作为开发池、IQ_8-10 作为 held-out eval；Day2-4 使用 IQ_1-7 discovery/enrollment、IQ_8-10 held-out eval。",
            "manifest 中的 stage/label 边界来自公开协议和既有 strict split，不用于未知真值调参。",
        ],
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
