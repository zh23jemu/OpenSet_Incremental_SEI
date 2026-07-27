"""RADCIL 阶段 2/3 WiSig 主组合与对照配置。

本模块把阶段 1 多种子确认后的主后端选择固化为可复用配置，避免
Slurm 脚本、计划生成器和后续报告各自手写一份参数。这里不改变旧
实验入口的默认行为，只为阶段 2/3 新主流程和正式消融提供显式、
可审计的参数源。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RadcilWisigConfig:
    """WiSig RADCIL 主流程/正式对照的最小锁定配置。

    字段含义：
    - `variant`：阶段 1 报告中使用的配置名称，便于和结果表对齐。
    - `frontend`：当前主发现前端；阶段 1 证明 MV-ACC 强于 SimGCD-style
      和 IGCD-minimal，因此首个主流程继续使用稳定 MV-ACC。
    - `seed`：首个单种子主流程使用的随机种子，先沿用阶段 1 锚点 seed 7。
    - `cil_replay_weight`：旧类 replay CE loss 权重。
    - `old_new_batch_ratio`：replay 旧类 batch 相对当前新类 batch 的比例。
    - `closedset_epochs`、`cil_head_warmup_epochs`、`cil_joint_epochs`：主流程
      训练轮数。它们保持为显式配置，确保主方法和 high-replay 消融
      使用同协议、同训练预算，只改变需要审计的后端强度参数。
    """

    variant: str = "ratio_2p0_replay_3p0"
    frontend: str = "MV-ACC"
    seed: int = 7
    cil_replay_weight: float = 3.0
    old_new_batch_ratio: float = 2.0
    initial_known_classes: int = 10
    round_size: int = 10
    num_rounds: int = 3
    selected_rx_list: str = "2"
    closedset_epochs: int = 20
    cil_head_warmup_epochs: int = 2
    cil_joint_epochs: int = 8
    supcon_weight: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 友好的配置字典。"""
        return asdict(self)

    def experiment_args(self, dataset_path: str, save_dir: str) -> list[str]:
        """生成 `exp_wisig_mvacc_cil_strict.py` 可直接使用的参数列表。

        返回值保持为拆分后的 argv 形式，而不是拼接成一整段 shell 字符串；
        这样计划 JSON 可以清楚审计每个参数，也减少路径或数值转义错误。
        """
        return [
            "--dataset_path",
            dataset_path,
            "--save_dir",
            save_dir,
            "--selected_rx_list",
            self.selected_rx_list,
            "--initial_known_classes",
            str(self.initial_known_classes),
            "--round_size",
            str(self.round_size),
            "--num_rounds",
            str(self.num_rounds),
            "--seed",
            str(self.seed),
            "--train_closedset",
            "--epochs",
            str(self.closedset_epochs),
            "--batch_size",
            "128",
            "--test_batch_size",
            "256",
            "--disable_cil_baselines",
            "--use_supcon",
            "--supcon_weight",
            str(self.supcon_weight),
            "--cil_head_warmup_epochs",
            str(self.cil_head_warmup_epochs),
            "--cil_joint_epochs",
            str(self.cil_joint_epochs),
            "--cil_replay_weight",
            str(self.cil_replay_weight),
            "--radcil_old_new_batch_ratio",
            str(self.old_new_batch_ratio),
        ]


def get_radcil_wisig_config(variant: str = "ratio_2p0_replay_3p0", seed: int | None = None) -> RadcilWisigConfig:
    """返回 WiSig RADCIL 配置。

    `ratio_2p0_replay_3p0` 是阶段 2/3 主配置；`ratio_3p0_replay_3p0`
    是正式同协议 high-replay 后端消融。这里集中管理两个配置，避免
    Slurm 脚本和报告脚本各自复制参数后产生偏差。
    """
    configs = {
        "ratio_2p0_replay_3p0": RadcilWisigConfig(),
        "ratio_3p0_replay_3p0": RadcilWisigConfig(
            variant="ratio_3p0_replay_3p0",
            old_new_batch_ratio=3.0,
        ),
    }
    if variant not in configs:
        valid = ", ".join(sorted(configs))
        raise ValueError(f"未知 RADCIL WiSig 配置：{variant}；可选值：{valid}")

    base = configs[variant]
    if seed is None or seed == base.seed:
        return base
    return RadcilWisigConfig(
        variant=base.variant,
        frontend=base.frontend,
        seed=int(seed),
        cil_replay_weight=base.cil_replay_weight,
        old_new_batch_ratio=base.old_new_batch_ratio,
        initial_known_classes=base.initial_known_classes,
        round_size=base.round_size,
        num_rounds=base.num_rounds,
        selected_rx_list=base.selected_rx_list,
        closedset_epochs=base.closedset_epochs,
        cil_head_warmup_epochs=base.cil_head_warmup_epochs,
        cil_joint_epochs=base.cil_joint_epochs,
        supcon_weight=base.supcon_weight,
    )


def get_stage2_wisig_main_config(seed: int | None = None) -> RadcilWisigConfig:
    """返回阶段 2/3 WiSig 主流程配置，保持旧调用方兼容。"""
    return get_radcil_wisig_config("ratio_2p0_replay_3p0", seed=seed)
