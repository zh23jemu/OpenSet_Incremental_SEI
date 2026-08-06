#!/usr/bin/env python3
"""下载 Stage69 旧类跨天校准所需的最小 LoRa 原始 I/Q 文件。

Stage44 只下载了原 strict 协议必要文件：Day2-4 新类 discovery IQ_1-7，
以及累计已见类 held-out eval IQ_8-10。Stage69 需要额外验证“如果客户允许
每个旧设备在新采集日提供少量标注校准信号，LoRa 是否能接近 50%”，因此只
补下载 Day2-4 旧设备的 IQ_1，而不是下载完整 Setup 1。

该脚本只做文件准备和 manifest 审计，不参与训练、不读取 held-out eval 真值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from build_lora25_compact import BASE_URL, DEVICE_ORDER


def old_day_calibration_records(transmissions: list[int]) -> list[dict[str, int | str]]:
    """生成 Day2-4 旧设备校准文件清单。

    R1 评估日为 Day2，旧类是初始 10 个设备；R2 的旧类为前 15 个设备；
    R3 的旧类为前 20 个设备。这里的标签/设备顺序来自公开实验协议，不是
    从 held-out eval 结果反推。
    """

    specs = [
        ("after_r1_old_calibration", 2, DEVICE_ORDER[:10]),
        ("after_r2_old_calibration", 3, DEVICE_ORDER[:15]),
        ("after_r3_old_calibration", 4, DEVICE_ORDER[:20]),
    ]
    records: list[dict[str, int | str]] = []
    for stage, day, devices in specs:
        for device in devices:
            for transmission in transmissions:
                records.append(
                    {
                        "stage": stage,
                        "day": int(day),
                        "device": int(device),
                        "transmission": int(transmission),
                    }
                )
    return records


def source_url(record: dict[str, int | str]) -> str:
    """生成 OSU LoRa Setup 1 官方文件 URL。"""

    return (
        f"{BASE_URL}/Day{int(record['day'])}/Device{int(record['device'])}/"
        f"IQ_{int(record['transmission'])}.dat"
    )


def raw_relative_path(record: dict[str, int | str]) -> Path:
    """生成本地保存相对路径，保持和 Stage44 原始目录结构一致。"""

    return (
        Path(f"Day{int(record['day'])}")
        / f"Device{int(record['device'])}"
        / f"IQ_{int(record['transmission'])}.dat"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算 SHA-256，避免一次性读取 160MB 文件。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: Path, retries: int) -> None:
    """下载单个文件到 `.part`，成功后原子替换为最终路径。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(int(retries)):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "OpenSet-Incremental-SEI-stage69/1.0"},
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                with part_path.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            part_path.replace(target)
            return
        except Exception as exc:  # noqa: BLE001 - 网络和文件系统异常统一记录后重试
            last_error = exc
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"download failed for {url}: {last_error}")


def prepare_one_record(
    raw_dir: Path,
    record: dict[str, int | str],
    index: int,
    total: int,
    download: bool,
    retries: int,
) -> dict[str, object]:
    """下载或检查单个校准文件，并返回 manifest 记录。

    该函数被串行/并发两种路径复用。并发只影响文件准备速度，不改变文件清单、
    保存路径或后续校准协议；每个文件都先写入独立的 `.part` 临时文件，成功后
    再原子替换为最终 `.dat`，避免半下载文件被误认为可用数据。
    """

    target = raw_dir / raw_relative_path(record)
    url = source_url(record)
    existed_before = target.exists()
    status = "exists" if existed_before else "planned"
    if download and not existed_before:
        download_file(url, target, retries=retries)
        status = "downloaded"
    exists_after = target.exists()
    manifest_record = {
        "index": int(index),
        "stage": str(record["stage"]),
        "day": int(record["day"]),
        "device": int(record["device"]),
        "transmission": int(record["transmission"]),
        "url": url,
        "relative_path": str(raw_relative_path(record)).replace("\\", "/"),
        "local_path": str(target.resolve()),
        "status": status if exists_after else "missing",
        "local_size_bytes": int(target.stat().st_size) if exists_after else None,
        "sha256": sha256_file(target) if exists_after else None,
    }
    print(
        f"[{index:03d}/{total:03d}] {manifest_record['status']}: "
        f"Day{record['day']} Device{record['device']} IQ_{record['transmission']}",
        flush=True,
    )
    return manifest_record


def build_manifest(
    raw_dir: Path,
    records: list[dict[str, int | str]],
    download: bool,
    retries: int,
    jobs: int,
) -> dict[str, object]:
    """下载/检查文件并返回可审计 manifest。

    `jobs` 只控制下载并发度。默认值为 1，保持历史串行行为；Slurm 下载阶段可
    显式设为 4，加快 Stage69 最小校准集补齐，降低 1 小时短任务超时风险。
    """

    manifest_records: list[dict[str, object]] = []
    indexed_records = [(index, record) for index, record in enumerate(records, start=1)]
    total = len(indexed_records)
    if int(jobs) <= 1:
        for index, record in indexed_records:
            manifest_records.append(
                prepare_one_record(
                    raw_dir=raw_dir,
                    record=record,
                    index=index,
                    total=total,
                    download=download,
                    retries=retries,
                )
            )
    else:
        # 并发路径只并行“文件准备”，最后按 index 排序写 manifest，保证审计文件稳定。
        with ThreadPoolExecutor(max_workers=int(jobs)) as executor:
            futures = [
                executor.submit(
                    prepare_one_record,
                    raw_dir,
                    record,
                    index,
                    total,
                    download,
                    retries,
                )
                for index, record in indexed_records
            ]
            for future in as_completed(futures):
                manifest_records.append(future.result())
        manifest_records.sort(key=lambda item: int(item["index"]))

    present = sum(1 for item in manifest_records if item["status"] in {"exists", "downloaded"})
    return {
        "dataset": "Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning",
        "subset": "LoRa RFFP Dataset - Different Days Indoor Scenario / Setup 1",
        "purpose": "Stage69 old-device same-day calibration minimal raw IQ supplement",
        "raw_dir": str(raw_dir.resolve()),
        "base_url": BASE_URL,
        "download_requested": bool(download),
        "download_jobs": int(jobs),
        "records_total": len(manifest_records),
        "records_present_or_downloaded": int(present),
        "estimated_total_bytes_if_complete": int(sum(int(item["local_size_bytes"] or 0) for item in manifest_records)),
        "notes": [
            "只补 Day2-4 旧设备 IQ_1 校准文件；不下载完整 Setup 1。",
            "这些文件用于新增标注校准协议验证，不属于原 strict 无旧类跨天校准成绩。",
            "不读取 IQ_8-10 held-out eval 真值做校准。",
        ],
        "records": manifest_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 Stage69 LoRa 旧类同日校准最小 IQ 文件")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--transmissions", default="1")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=1, help="并发下载线程数；默认 1 保持串行行为。")
    args = parser.parse_args()

    transmissions = [int(item.strip()) for item in str(args.transmissions).split(",") if item.strip()]
    if not transmissions:
        raise ValueError("--transmissions must contain at least one IQ index.")
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        raw_dir=raw_dir,
        records=old_day_calibration_records(transmissions),
        download=bool(args.download),
        retries=int(args.retries),
        jobs=max(1, int(args.jobs)),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
