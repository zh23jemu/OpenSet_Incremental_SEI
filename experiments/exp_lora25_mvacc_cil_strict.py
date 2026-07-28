"""LoRa25 Different Days Indoor strict MV-ACC-CIL 薄入口。

算法实现复用 WiSig strict 主流程，避免复制 MV-ACC 发现和 RADCIL 后端。
本入口只负责锁定 LoRa 数据 profile、10+5×3 类别规模和 transmission
隔离协议；Day1 IQ_7 是唯一校准/验证 split，IQ_8-10 始终仅用于评估。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp_wisig_mvacc_cil_strict import main as shared_main


def main() -> None:
    """调用共享 strict 主流程，并由命令行显式传入 LoRa profile。"""
    if "--dataset_profile" not in sys.argv:
        sys.argv.extend(["--dataset_profile", "lora25"])
    shared_main()


if __name__ == "__main__":
    main()
