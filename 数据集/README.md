# 数据集清单

本目录保存客户补充的超大原始数据压缩包。压缩包默认保留原样，不纳入 Git；实验运行前根据目标数据集按需解压，避免同时展开全部数据占满本地磁盘。

## 当前数据

| 数据集 | 文件 | 压缩包内容 | 状态 |
| --- | --- | --- | --- |
| ADS-B | `ADS-B.rar` | `Dataset/` 下 10、20、30、90 类训练/测试所需的 14 个 `.npy` 文件 | 完整，作为规范副本 |
| ADS-B | `Dataset.rar` | 与 `ADS-B.rar` 逐字节相同 | 重复副本，暂不删除 |
| ManyTx | `ManyTx.pkl.zip` | `ManyTx.pkl`，解压后约 4.18 GB | 完整 |
| ManyRx | `ManyRx.pkl.zip` | `ManyRx.pkl`，解压后约 2.04 GB | 完整 |

项目根目录另有完整 WiSig 数据：

- `../WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl`

仓库内还保留了可直接使用的紧凑数据：

- `../datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz`
- `../datasets/manyrx_compact/manyrx_4known_3round_fixed_cross_rx.npz`

## ADS-B 文件核对

`ADS-B.rar` 已确认包含当前加载器要求的核心文件：

- `X_train_90Class.npy`、`Y_train_90Class.npy`
- `X_train_30Class.npy`、`Y_train_30Class.npy`
- `X_test_30Class.npy`、`Y_test_30Class.npy`

压缩包还包含 10 类和 20 类的训练、测试文件，可用于补充协议实验。

## 使用约定

- 当前 C 盘剩余空间不足以稳妥地同时展开全部数据，优先在训练服务器或释放足够空间后按需解压。
- ADS-B 以 `ADS-B.rar` 作为规范副本；`Dataset.rar` 仅作为保留的重复副本。
- 解压后通过实验参数显式传入数据路径，不依赖源码中的开发者绝对路径。
- 原始数据、压缩包和本地解压目录均不提交到 Git。

## 完整性结论

当前主线所需的 WiSig、ADS-B、LoRa 数据均已具备。ManyTx 和 ManyRx 的完整原始压缩包也已具备。ORACLE/Orbit RF 原始数据仍未提供，但它们不属于当前确认的 WiSig、ADS-B、LoRa 主实验范围。
