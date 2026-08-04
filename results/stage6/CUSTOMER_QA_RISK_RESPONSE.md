# 阶段 6 客户问答与风险口径

## 1. DOI-style 是什么

DOI-style 是本项目中加入的简化版增量后端 baseline，不是官方完整实现。原版来自 DOI 框架论文，但代码没有完整开源，无法严格复现。因此这里按论文思路复现了核心逻辑：使用稳定原型/记忆来帮助旧类保持，并作为 RADCIL 网络后端的对照。

对外说明时应避免说“复现了 DOI 官方方法”，更准确的说法是：这是一个 DOI-inspired / DOI-style 的简化对照，用来观察原型记忆类后端在旧类保持上的上限和局限。

## 2. LoRa 数据集能否下载

LoRa 数据集可以下载，但完整数据体量较大。当前已经按实验需要下载了 `Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning` 里的 `LoRa RFFP Dataset - Different Days Indoor Scenario`，也就是 Setup 1 必要原始 I/Q 子集。

当前有两份口径：本地紧凑审计子集约 9.7 MB；服务器上实验用 Setup 1 原始 I/Q 为 385 个 `.dat`，约 58GB，放在 `/mnt/usmidet/billy_test`，不再占 `/mnt/users/xj62kv` 家目录。后续如果扩展 LoRa 其它场景，再继续按需下载，不需要一次性下载全部 LoRa。

## 3. ADS-B 和 LoRa 结果是否偏低

是，应该如实承认，但口径要更新。ADS-B 通过联合 discovery-CIL 后，R3 Overall 三种子到 `0.5149±0.0107`，已经过 50%。LoRa 也不是之前 15%-17% 或 25.4% 的版本了；Stage 48 用 Setup 1 原始 I/Q s28 多窗 + recording-consensus 0.65，R3 Overall/Old/New/Forgetting 为 `0.2803±0.0148/0.2312±0.0133/0.4770±0.0463/0.1802±0.0185`。

因此 ADS-B 和 LoRa 不适合包装成“效果很好”。更稳妥的客户口径是：

- ADS-B：现在三种子已经过 50%，当前正式候选是联合 discovery-CIL，但还要如实说明稳定性和旧新类权衡。
- LoRa：当前正式候选从 Stage 27 Chirp 的 `25.4%` 提升到 Stage 48 的 `28.0%`，Old 和遗忘改善明显，New 只小幅低一点；但绝对准确率仍偏低，不能说已经完全解决。
- 后续若继续提升，应继续围绕原始 I/Q 多窗表征和 New 稳定性，而不是回到 density ratio、BN、old-logit bias、late-fusion 或 anchor 权重这类小参数。

## 4. 建议口语回复

DOI-style 是我加的一个简化版 baseline，不是官方完整实现。原版来自 DOI 框架论文，但代码没完整开源，所以没法严格复现。我这里只是按论文思路做了一个简洁版，主要用来对比原型/记忆这类后端对旧类保持的效果。

LoRa 不用全量下载所有场景。我们现在已经下载了实验需要的 Setup 1 / Different Days Indoor 原始 I/Q，大概 58GB，放在服务器大盘里，不占家目录；本地还有 9.7MB 的紧凑审计子集。

ADS-B 现在三种子已经过 50%，LoRa 也从之前 25.4% 提到 28.0%。LoRa 这块提升主要来自 Setup 1 原始 I/Q 重新切 s28 多窗，再加 recording-consensus 0.65；Old 和遗忘改善明显，New 基本还在 47.7%。但 LoRa 绝对值还是不算高，所以可以说“有结构性提升”，不能说“已经完全解决”。
