# 阶段 6 客户问答与风险口径

## 1. DOI-style 是什么

DOI-style 是本项目中加入的简化版增量后端 baseline，不是官方完整实现。原版来自 DOI 框架论文，但代码没有完整开源，无法严格复现。因此这里按论文思路复现了核心逻辑：使用稳定原型/记忆来帮助旧类保持，并作为 RADCIL 网络后端的对照。

对外说明时应避免说“复现了 DOI 官方方法”，更准确的说法是：这是一个 DOI-inspired / DOI-style 的简化对照，用来观察原型记忆类后端在旧类保持上的上限和局限。

## 2. LoRa 数据集能否下载

LoRa 数据集可以下载，但完整数据体量较大。当前实验使用的是 `Comprehensive LoRa RF Datasets for Device Fingerprinting Using Deep Learning` 中的 `LoRa RFFP Dataset - Different Days Indoor Scenario` 子集，并已基于紧凑版数据完成 10+5x3 strict 协议验证。

如果后续需要完整复现或扩展 LoRa，可以在 Slurm 服务器上按需下载完整数据，或优先只下载/切分当前协议需要的 Different Days Indoor Scenario 必要子集，避免占用本地 C 盘空间。

## 3. ADS-B 和 LoRa 结果是否偏低

是，应该如实承认。ADS-B 通过 long backbone 和默认 target split 后，R3 Overall 提升到约 `0.4932±0.0128`，接近 50%，但遗忘率仍高于 DOI-style。LoRa 当前 seed7 最好的一组 DOI-style / grouped hybrid 结果约为 `0.15-0.17`，明显低于主数据集水平。

因此 ADS-B 和 LoRa 不适合包装成“效果很好”。更稳妥的客户口径是：

- ADS-B：发现欠聚类风险已经被 target split 收敛到可用候选，但整体准确率仍接近 50%，剩余问题是后端旧新类权衡。
- LoRa：完整链路已经跑通，但跨天/跨体制旧类漂移很强，目前更适合作为局限分析和补充验证。
- 后续若继续提升，应换更强的表征或结构不同的增量后端，而不是继续调 density ratio、target split threshold、late-fusion 权重或 anchor 权重。

## 4. 建议口语回复

DOI-style 是我加的一个简化版 baseline，不是官方完整实现。原版来自 DOI 框架论文，但代码没完整开源，所以没法严格复现。我这里只是按论文思路做了一个简洁版，主要用来对比原型/记忆这类后端对旧类保持的效果。

LoRa 数据集是能下载的，不过完整数据比较大。现在用的是 Different Days Indoor 这个子集，后面如果要完整复现，可以在服务器上继续下载完整数据，或者只下载我们实验需要的部分。

ADS-B 和 LoRa 目前结果确实不高。ADS-B 加了 target split 后接近 50%，但遗忘还是偏高；LoRa 更低一些，主要说明跨体制很难。所以这两个现在更适合做补充实验和局限分析，不能硬说效果很好。
