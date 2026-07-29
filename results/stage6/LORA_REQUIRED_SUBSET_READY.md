# LoRa 实验必要子集准备说明

## 当前结论

LoRa 不需要下载完整数据集。当前项目已经具备实验所需的 `Different Days Indoor Scenario` 紧凑子集，并已通过 strict 协议审计。

- 子集文件：`datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz`
- 文件大小：约 9.7 MB
- 审计结果：`results/stage6/lora_required_subset_audit.json`，`ok=true`
- 数据来源：`Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning` 中的 `LoRa RFFP Dataset - Different Days Indoor Scenario`

## 覆盖范围

当前子集只保留实验需要的 25 个设备和 10+5x3 开集增量协议：

- Day1：10 个已知设备
  - IQ_1-6：闭集 backbone 训练
  - IQ_7：验证/校准
  - IQ_8-10：held-out 初始评估
- Day2：新增 5 个未知设备，作为 R1
- Day3：新增 5 个未知设备，作为 R2
- Day4：新增 5 个未知设备，作为 R3
- Day2-4：
  - IQ_1-7：discovery/enrollment
  - IQ_8-10：held-out evaluation

## 为什么不下载完整 LoRa

完整 LoRa 数据体量较大，当前目标只需要跨天 Different Days Indoor 子集来验证：

- 未知设备发现
- 聚类和伪标签
- 网络增量训练
- 旧类保持和跨天漂移局限

因此只保留必要子集更合适，避免占用本地磁盘和拖慢同步。

## 可对客户说明

LoRa 不下完整包了，只保留实验需要的 Different Days Indoor 子集。这个子集已经切成项目需要的 10+5x3 协议，并且本地审计通过，可以直接支撑当前 LoRa 实验和交付说明。
