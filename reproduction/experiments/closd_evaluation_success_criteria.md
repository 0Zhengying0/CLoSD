# CLoSD 闭环评测：回合结束与动作成功判定

本文记录仓库中 `closd_eval=True` 的**多任务闭环成功率评测**。评测对象是在 Isaac Gym 中执行的整段动作，而不是单独生成的动作视频；DiP、Flow Matching 或 Rectified Flow 只要接入同一闭环评测路径，就使用下述环境任务判定与统计代码。本文所说的“成功”均指**该评测脚本的指标**，不代表真机可部署性。

主要源码：[评测计数器](../../closd/learning/im_amp_players.py#L51-L92)、[多任务完成条件与状态机](../../closd/env/tasks/closd_multitask.py#L270-L332)、[回合终止判定](../../closd/env/tasks/humanoid.py#L1555-L1590)。下文的数值对应[现有复现评测命令](../scripts/hpc/dip_eval_recipe_300k_30m.slurm#L42-L60)：控制器为 `CLoSD_multitask_finetune`，`closd_eval=True`、`num_envs=4096`、`episode_length=500`、规划器 CFG 系数为 `7.5`。换配置时需重新核对。

## 1. 一次测试是什么，何时结束，持续多久

一次**单回合测试**从某个并行环境 `reset` 开始，到该环境的 `done=1` 结束。达到动作目标本身通常**不会立即结束回合**：REACH 与 GET_UP 会继续运行；SIT 成功后可以转入 GET_UP；KICK/PUNCH 击倒箱子后可以转入 HALT。[任务状态更新](../../closd/env/tasks/closd_task.py#L178-L200) · [状态转换](../../closd/env/tasks/closd_multitask.py#L280-L328)

`done=1` 有两种来源：[Humanoid 回合终止函数](../../closd/env/tasks/humanoid.py#L1555-L1590)。

1. **提前终止 `terminate=1`**：启用 `enableEarlyTermination` 时，忽略配置中允许接触的身体部位；其余部位中只要存在接触力任一分量绝对值 `>0.1`，且存在非豁免部位高度低于终止高度，并且进度步数 `>1`，就判定跌倒并结束。两个 `存在` 判断在代码中分别求值，不要求是同一部位。当前基础高度阈值为 `0.15 m`，头部为至少 `0.30 m`。这是代码里的**跌倒代理条件**，不是完整的静态或动态平衡检测。[允许接触部位](../../closd/data/cfg/env/closd_base.yaml#L20-L23) · [终止配置](../../closd/data/cfg/env/im_single_prim.yaml#L40-L48) · [高度阈值](../../closd/env/tasks/humanoid.py#L653-L668)
2. **达到时间上限**：`progress_buf >= episode_length - 1` 时结束。现有复现实验用命令行覆盖为 `env.episode_length=500`；仓库的多任务 YAML 默认值是 `400`，不能用默认值解释这批日志。[复现评测命令](../scripts/hpc/dip_eval_recipe_300k_30m.slurm#L42-L60) · [默认配置](../../closd/data/cfg/env/closd_multitask.yaml#L1-L6)

每次重置后 `progress_buf=0`，每个控制步结束加 1，因此上述 `500` 设置下，正常到时的首次 `done` 出现在**第 499 个控制步结束**。[重置计数器](../../closd/env/tasks/humanoid.py#L582-L585) · [逐步递增](../../closd/env/tasks/humanoid.py#L1297-L1309) · [上限比较](../../closd/env/tasks/humanoid.py#L1586-L1590) 仿真时间步长为 `1/60 s`，控制动作每 2 个仿真步更新一次，即 **30 Hz**。所以正常到时约为 `499/30 ≈ 16.63 s` 仿真时间；提前跌倒则更短。口头上的“500 帧、约 16.7 秒”是近似说法。这里的秒数是**仿真时间**，不是包含模型采样与 4096 环境计算开销的实际运行耗时。[仿真步长](../../closd/utils/config.py#L42) · [控制频率](../../closd/data/cfg/env/im_single_prim.yaml#L33-L35) · [控制周期计算](../../closd/env/tasks/humanoid.py#L118-L120)

**整个评测作业**与单回合不同：计数器要求每个待评估任务累计至少 `1000` 个已结束回合，才打印成功数 / 实际结束数并退出；并行环境为 `4096` 个，某类最终分母可能高于 `1000`。这些输出是一次评测运行的结果，不是多随机种子的平均值。无任务过滤时，初始任务池为 KICK、PUNCH、SIT、SIT、REACH、REACH；GET_UP 与 HALT 只由状态转换进入，不是独立抽取的初始任务。[停止条件与输出](../../closd/learning/im_amp_players.py#L55-L92) · [任务池](../../closd/env/tasks/closd_multitask.py#L45-L66) · [评测命令](../scripts/hpc/dip_eval_recipe_300k_30m.slurm#L42-L60)

## 2. 统计脚本怎样把结束的回合记为成功

对每个环境，记 `D=done`、`T=terminate`、`Q=task_done`、`S=task_state`。通常的成功基础条件是 **`B = D ∧ ¬T ∧ Q`**：回合结束、没有因跌倒提前终止、任务完成标志为真。[评测公式](../../closd/learning/im_amp_players.py#L63-L83) `task_done` 会在检测到目标后保留到当前子任务切换或环境重置，因而通常表示“此前已达到过当前子任务目标”，而不是“结束瞬间仍在目标上”。[完成标志更新](../../closd/env/tasks/closd_task.py#L196-L202)

| 日志项 | 回合计入分母的条件 | 回合计入成功数的条件 |
| --- | --- | --- |
| REACH | `D ∧ (S=REACH)` | `B ∧ (S=REACH)` |
| SIT | `D ∧ (S∈{SIT, GET_UP})` | `[B ∧ (S=SIT)] ∨ [D ∧ (S=GET_UP)]` |
| GET_UP | `D ∧ (S=GET_UP)` | `B ∧ (S=GET_UP)` |
| KICK | `D ∧ (S∈{KICK, PUNCH, HALT})` | `[B ∧ (S∈{KICK, PUNCH})] ∨ [D ∧ (S=HALT)]` |
| PUNCH | 与 KICK **完全相同** | 与 KICK **完全相同** |

这张表逐项对应[原始 `state_resets`、`state_success` 分支](../../closd/learning/im_amp_players.py#L73-L85)。特别注意：

- **SIT 的 GET_UP 分支和击打的 HALT 分支没有 `¬T` 或 `Q` 限制。** 进入对应后续状态后，哪怕随后跌倒导致 `terminate=1`，原先的坐下或击打仍计为成功。
- **KICK 与 PUNCH 不是独立成功率。** 代码注释说明进入 HALT 后丢失了原先的击打类别，于是两个日志项都把两种击打及 HALT 的结束回合计入同一分母和分子，故两项数值相同。[源码注释](../../closd/learning/im_amp_players.py#L77-L80)
- **GET_UP 是条件化的统计。** 只有回合结束时已进入 GET_UP 的样本进入它的分母；没有先完成 SIT、因此从未切换到 GET_UP 的样本不计入 GET_UP 分母。一个结束于 GET_UP 的回合同时进入 SIT 与 GET_UP 的分母。

## 3. 各动作的目标条件与状态转换

### REACH：到达目标

REACH 的目标关节是 `traj`，代码将其映射到骨盆，并把目标与当前位置的高度坐标都置零，所以检查的是**骨盆水平位置到目标的距离 ≤ `done_dist=0.3 m`**。[REACH 条件](../../closd/env/tasks/closd_task.py#L34-L37) · [距离判定](../../closd/env/tasks/closd_multitask.py#L456-L470) 首次达到后 `task_done` 保留，REACH 状态机不作切换；直到回合结束且没有跌倒终止，才进入 REACH 成功数。若先到目标、后来跌倒，REACH **失败**。[不切换状态](../../closd/env/tasks/closd_multitask.py#L330-L332) · [计数器](../../closd/learning/im_amp_players.py#L81-L85)

### SIT：坐到沙发位置

SIT 的局部完成条件是：根部/骨盆的水平位置落入以沙发为中心计算出的矩形，且高度在 **`0.30–0.50 m`** 之间。矩形尺寸参数是 `1.2 × 2.5 m`；代码没有额外要求骨盆与沙发发生接触、坐姿保持若干秒，或重心持续稳定。[矩形与高度参数](../../closd/env/tasks/closd_multitask.py#L89-L105) · [实际判定](../../closd/env/tasks/closd_multitask.py#L415-L433)

首次满足条件后，`is_done` 保留；在原始成功触发后的 **60 个控制步，约 2 秒**，状态机尝试从 SIT 转到 GET_UP。当前多任务配置 `getup_prob=1`，因此只要到达转换时点且回合还在运行，就会转入 GET_UP。**保留的完成标志不要求 60 步内始终保持坐姿。**[延迟与状态转换](../../closd/env/tasks/closd_multitask.py#L101-L107) · [转换实现](../../closd/env/tasks/closd_multitask.py#L304-L326) · [概率配置](../../closd/data/cfg/env/closd_multitask.yaml#L1-L6)

统计 SIT 成功有两种情况：结束时仍在 SIT 且 `B` 为真；或者结束时已在 GET_UP，此时**不管 GET_UP 是否完成、后来是否跌倒**，SIT 都成功。因此“曾坐到位置”本身还不必然计入成功：若在状态转换前跌倒且未能无跌倒运行到时限，SIT 失败。[SIT 计数公式](../../closd/learning/im_amp_players.py#L73-L76)

### GET_UP：从沙发起身到目标点

进入 GET_UP 时，目标由沙发目标平移：水平约 `1 m`，高度从沙发目标的 `0.4 m` 再加 `0.53 m`，约为 `0.93 m`。局部完成条件是**骨盆三维位置到该目标的欧氏距离 ≤ `0.3 m`**。[目标更新](../../closd/env/tasks/closd_multitask.py#L321-L326) · [三维距离判定](../../closd/env/tasks/closd_multitask.py#L424-L433) 最终只有在回合结束时仍是 GET_UP、任务完成标志为真、且没有跌倒终止，才计为 GET_UP 成功；先起身达标而后跌倒，GET_UP **失败**。[GET_UP 计数公式](../../closd/learning/im_amp_players.py#L81-L85)

### KICK / PUNCH：击倒箱子

两种击打使用相同的**箱体姿态**完成条件：将箱体局部向上方向旋转到世界坐标，计算它与世界向上方向的点积；当点积 `≤ 0.25` 时触发击打完成。若按两根单位向量解释，相当于箱体上方向相对竖直偏转至少约 `75.5°`。源码注释写“箱子在地上”，但实际代码只查方向，**没有检查箱底是否触地、由哪个身体部位击倒、是否严格符合踢/拳动作形式**。[角度阈值](../../closd/env/tasks/closd_multitask.py#L80-L85) · [姿态条件](../../closd/env/tasks/closd_multitask.py#L435-L453)

检测到箱子达标后，状态机把 KICK/PUNCH 切到 **HALT**，清除当前子任务的 `is_done`，并把 HALT 的目标设为此时骨盆的水平位置及 `z=0.93 m`。HALT 的生成条件文本是 “A person is standing still.”，可理解为击打后的原地站立子任务。HALT 自身的局部完成条件仅为**骨盆水平位置距 HALT 目标 ≤ `0.3 m`**，但**击打成功率并不要求完成 HALT 局部目标或持续站稳**。[切换与目标更新](../../closd/env/tasks/closd_multitask.py#L280-L301) · [HALT 条件文本](../../closd/env/tasks/closd_task.py#L46-L48) · [HALT 距离条件](../../closd/env/tasks/closd_multitask.py#L447-L453)

因此，**箱子先达倾倒阈值、环境确实切入 HALT、角色再跌倒**，结束时状态仍是 HALT，这次击打仍计为成功。若箱子倾倒与跌倒发生在同一仿真步，可能在下一次状态更新前就结束，此时未切入 HALT，不能一概认为成功。状态更新在控制步的物理模拟之前执行；它读取先前物理步的结果，而评测在 `done` 步统计状态。[更新调用顺序](../../closd/env/tasks/humanoid_amp_task.py#L58-L62) · [清除终止回合的完成标志](../../closd/env/tasks/closd_task.py#L196-L200) · [评测调用](../../closd/learning/im_amp_players.py#L294-L300)

## 4. “物理可行”的实际覆盖范围

这些动作由 PhysX 模拟的人体与场景物体执行：有刚体状态、接触和地面摩擦；当前地面静/动摩擦系数均设为 `1.0`、恢复系数为 `0.0`。人体当前使用 `isaac_pd` 位置目标控制，控制器每 2 个 `1/60 s` 仿真步更新一次。[模拟配置](../../closd/data/cfg/sim/default_sim.yaml) · [摩擦与控制参数](../../closd/data/cfg/env/im_single_prim.yaml#L28-L35) · [地面参数应用](../../closd/env/tasks/humanoid.py#L589-L600) · [关节目标应用](../../closd/env/tasks/humanoid.py#L1204-L1230)

这比只看离线动画是否自然更接近**特定仿真环境中的可执行性**；但成功率只检查前述目标事件与有限的跌倒条件。它**没有显式检查**质心或支撑多边形的稳定裕度、接触力/力矩平衡、持续站稳、抗扰动性能、真实硬件的关节力矩与速度限制或仿真到真实世界的误差。因此不能把该成功率解读为“符合真机动力学、可以部署”的比例；击打和坐下的状态转换例外尤其可能把后续跌倒的回合计为成功。[目标与状态条件](../../closd/env/tasks/closd_multitask.py#L405-L470) · [终止条件](../../closd/env/tasks/humanoid.py#L1555-L1590) · [评测统计例外](../../closd/learning/im_amp_players.py#L73-L80)

## 5. 读成功率时应保留的限定

- 成功率是**结束回合数作分母**的比例，不是每一帧的命中率，也不是动作视频的主观评分；目标达到后通常还要运行到跌倒或回合时限。[统计代码](../../closd/learning/im_amp_players.py#L63-L92)
- REACH/GET_UP 要求结束时没有提前终止；SIT 进入 GET_UP、击打进入 HALT 后，对应原任务的成功分支则允许后续跌倒。它们衡量的不是完全相同的“全过程安全完成”标准。[分支公式](../../closd/learning/im_amp_players.py#L73-L83)
- 当前 KICK/PUNCH 两列是同一个混合击打统计，不应把两列相等解释为踢和拳各自性能恰好相同；GET_UP 也只在已经进入 GET_UP 的回合上计算。[评测计数器](../../closd/learning/im_amp_players.py#L73-L85)
- 环境在每个控制步**开始前**根据上一物理步状态更新 `task_done` 与状态机，评测则用回合结束时报告的状态与完成标志。因此恰在终止物理步首次达到目标、尚未来得及更新标志的情况，不能简单归入“已完成后又失败”。[任务更新时点](../../closd/env/tasks/humanoid_amp_task.py#L58-L62) · [信息上报](../../closd/env/tasks/closd_task.py#L168-L172) · [评测时点](../../closd/learning/im_amp_players.py#L294-L300)
