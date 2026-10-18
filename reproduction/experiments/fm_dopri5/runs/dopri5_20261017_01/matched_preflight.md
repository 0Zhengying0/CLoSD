# FM 300k：四种求解器的新同预算闭环对照

四组共用 FM 300k EMA checkpoint、CFG 7.5、seed 0、4096 环境、episode length 500 与官方成功率口径。
Euler/Heun/RK4 固定每次规划 NFE；DOPRI5 使用 `rtol=1e-3, atol=1e-4`，实际 NFE 随规划调用变化。
公共预算来自 20 个真实条件快照的 DOPRI5 校准，并取最接近的 4 的倍数。

| 求解器 | 状态 | 步数 | NFE 均值 [最小,最大] | MDM 前向均值 | Planner 延迟均值/中位数(s) | REACH | KICK | PUNCH | SIT | GET_UP | 等权平均 | Job ID |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| euler | NOT_SUBMITTED | — | — | — | — | — | — | — | — | — | — | — |
| heun | NOT_SUBMITTED | — | — | — | — | — | — | — | — | — | — | — |
| rk4 | NOT_SUBMITTED | — | — | — | — | — | — | — | — | — | — | — |
| dopri5 | NOT_SUBMITTED | — | — | — | — | — | — | — | — | — | — | — |

## 分析

- 校准或正式提交尚未完成，不填写成功率或公共 NFE 预算。
- 未完成/失败任务保留状态；不从部分日志猜测成功率。
- 历史最佳 RK4 12 NFE 成功率 83.82%；它属于不同计算预算，仅作背景参考。
- MDM 使用 `round(t × 999)`；时间索引跳变可能影响 DOPRI5 自适应步长与 NFE。

## 运行元数据

- FM-ME：checkpoint `—`；Git `—`；seed `—`；配置 `—`；源码 SHA256 `—`。
- FM-MH：checkpoint `—`；Git `—`；seed `—`；配置 `—`；源码 SHA256 `—`。
- FM-MR：checkpoint `—`；Git `—`；seed `—`；配置 `—`；源码 SHA256 `—`。
- FM-MD：checkpoint `—`；Git `—`；seed `—`；配置 `—`；源码 SHA256 `—`。
