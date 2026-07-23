# 主实验：WiSig、ManyTx、ManyRx

三套主实验采用同一协议：每个阶段先按 70/30 产生 discovery/enrollment 与独立评估集；Day1 的 70% 再进行分层 6:1 划分，形成整体 **60% 训练 / 10% 验证 / 30% 评估**。验证集仅用于 checkpoint 选择、MV-ACC 校准与 CF-LCG 特征选择，未知轮次和评估集标签从不进入这些步骤。

## 完整方法（MV-ACC）

1. 1D ResNet backbone，以 CE + SupCon 训练；验证准确率保存 checkpoint，并使用保守 IQ 增强。
2. 从 Day1 验证集的候选经典 RF 特征中，CF-LCG 以 10-NN 局部纯度选择最佳视图；纯度低于 0.80 时关闭最终方法的 RF 权重。
3. 深度视图与选择后的 RF 视图构建自适应融合图；HDBSCAN 产生微簇。
4. MV-ACC 执行多视图无丢样本的簇合并、噪声重分配和过大簇自适应分裂，随后以多原型注册新设备。

`cflcg_feature_selection.json` 记录每次实验实际选中的特征和门控结果。不要把其中的结果当作测试结果：它仅来自 Day1 验证集。

## 运行

在项目根目录 PowerShell 执行：

```powershell
.\experiments\run_wisig_rx3_mvacc.ps1
.\experiments\run_manytx_mvacc.ps1
.\experiments\run_manyrx_mvacc.ps1
```

每套 runner 默认开启 CE+SupCon、60/10/30、CF-LCG、自适应图融合和 UMAP+t-SNE。若要重新训练 Day1 backbone，在相应 runner 中加入 `-TrainClosedset`（或者直接在 PyCharm 的参数中加入 `--train_closedset`）。

## 可视化位置

每次运行在各自 `save_dir` 内生成，不会混入旧的诊断图：

```text
paper_visualizations/
  r1/umap/seen_true_labels.png
  r1/umap/seen_operational_labels.png
  r1/tsne/seen_true_labels.png
  ... r2, r3 ...
```

`true_labels` 仅用于解释聚类几何结构；`operational_labels` 展示实际流程中的已知类标签和发现到的伪标签。默认不再输出每个基线的大量 unknown-only 图；如需旧诊断图，增加 `--save_detailed_visualizations`。
