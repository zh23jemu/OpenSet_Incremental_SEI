"""验证 ADS-B 自适应密度门控的关键接受与拒绝分支。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许从任意工作目录直接运行 smoke test，无需额外设置 PYTHONPATH。
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mvacc_adaptive_density import (
    AdaptiveDensityThresholds,
    select_adaptive_density_candidate,
)


def make_result(ratio, labels, clusters, silhouette, confidence, cv, small_fraction):
    """构造不包含真实类别标签的最小候选输入。"""

    return {
        "ratio": ratio,
        "labels": np.asarray(labels, dtype=np.int64),
        "cluster_count": clusters,
        "silhouette": silhouette,
        "confidence": confidence,
        "cluster_size_cv": cv,
        "small_cluster_fraction": small_fraction,
    }


def main() -> int:
    anchor_labels = [0] * 6 + [1] * 6 + [2] * 6
    anchor = make_result(0.03, anchor_labels, 3, 0.25, 0.78, 0.40, 0.00)

    # 仅把一个锚点簇拆成两个规模合理的子簇，且所有结构门槛均改善，应接受。
    conservative_candidate = make_result(
        0.02,
        [0] * 6 + [1] * 6 + [2] * 3 + [3] * 3,
        4,
        0.34,
        0.77,
        0.45,
        0.00,
    )
    accepted = select_adaptive_density_candidate(
        anchor,
        conservative_candidate,
        AdaptiveDensityThresholds(),
    )
    assert accepted["candidate_accepted"]
    assert accepted["selected_ratio"] == 0.02

    # 即使 silhouette 更高，只要新增簇过多也必须回退锚点，防止早期过切分。
    fragmented_candidate = make_result(
        0.02,
        list(range(18)),
        18,
        0.60,
        0.80,
        0.30,
        0.00,
    )
    rejected_fragmented = select_adaptive_density_candidate(
        anchor,
        fragmented_candidate,
        AdaptiveDensityThresholds(),
    )
    assert not rejected_fragmented["candidate_accepted"]
    assert "adds_limited_clusters" in rejected_fragmented["rejected_by"]

    # 候选若只增加簇但没有足够分离度增益，也不能为了增加簇数而切换。
    weak_candidate = dict(conservative_candidate, silhouette=0.27)
    rejected_weak = select_adaptive_density_candidate(
        anchor,
        weak_candidate,
        AdaptiveDensityThresholds(),
    )
    assert not rejected_weak["candidate_accepted"]
    assert "silhouette_improves" in rejected_weak["rejected_by"]

    output = Path("results/stage4/stage4_adsb_adaptive_density_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "accepted_case": accepted,
                "fragmentation_rejection": rejected_fragmented,
                "weak_gain_rejection": rejected_weak,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B 自适应密度门控 smoke test 通过：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
