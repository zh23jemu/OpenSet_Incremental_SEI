# Stage83 LoRa 旧类 logreg 校准 + 新类保护门控 seed7 报告

- R3 Overall/Old/New/Forgetting: `0.3138/0.3119/0.3214/0.1905`
- 主模型原始 R3 Overall/Old/New: `0.2576/0.2387/0.3333`
- Stage73 oracle old-mask 诊断上限 R3 Overall/Old/New: `0.4695/0.4488/0.5524`
- 旧类路由比例：`0.5452`；新类保护比例：`0.2971`；门控阈值：`0.40`。
- 通过门槛：`False`；决策：`negative_or_diagnostic_only_do_not_expand`。
- 说明：本阶段不使用 held-out eval 真值做路由，目标是把 Stage73 的诊断上限转成可部署门控。
