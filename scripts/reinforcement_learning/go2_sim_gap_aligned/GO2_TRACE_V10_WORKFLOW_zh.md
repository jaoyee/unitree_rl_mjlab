# Go2 TRACE V10 中文流程与代码说明

本文档说明当前分支 `zkq/go2-trace-v10-latest` 中 Go2 TRACE V10 的完整实现、运行顺序、关键文件、产物协议，以及相对 V9 所做的修复。

本文档对应的代码提交为：

```text
69c6424  feat(go2): add trajectory diagnostics and hierarchical reward
58862e4  fix(trace): harden scorer labels and replay continuity
31a03f5  feat(trace): add complete Go2 TRACE V10 workflow
```

正式 V10 的冻结参数以以下文件为唯一依据：

```text
scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json
```

文档中的数值如果和运行时环境变量冲突，以协议文件和 fail-closed 检查结果为准。服务器上的数据集、RWM、policy checkpoint、标签缓存、replay shard 和日志不属于代码仓库。

## 1. V10 要解决的问题

V10 不改变 TRACE 的基本算法路线。它要解决的是 V8/V9 中已经确认的实现不一致、数据分布失控、scorer 训练不可靠、replay 连续性不足和实验不可复现问题。

正式路线仍然是：

```text
condition-specific 25K dataset
        |
        +--> frozen condition-specific RWM
        |
        +--> 从 dataset state reset 到 nominal simulator
                |
                +--> 当前 Policy 在 simulator 中 rollout
                +--> LLM preference labels
                +--> condition-specific scorer
                +--> 分布约束下的全局 Top-25%
                +--> simbuffer

持续恢复的 RWMbuffer + 固定 10% simbuffer
        |
        +--> 原 FlashSAC update
        +--> 下一轮 Policy
```

V10 明确保留：

- frozen RWM；
- nominal simulator；
- LLM preference 和 condition-specific scorer；
- FlashSAC 原有 actor/critic update；
- 固定 `r10`，每个 batch 中 90% RWMbuffer 和 10% simbuffer；
- 8 个 refresh，每轮 5M environment steps；
- sim/real 两侧相同的算法流程；
- 同 condition、同 gap 的 baseline 对比。

V10 没有引入：

- condition-specific 人工控制规则；
- 针对 RR 或 payload 的专用 reward；
- 动态 replay ratio；
- r25 正式分支；
- 用 fallback 或 stand 样本补齐 locomotion 配额；
- 为每个 gap 修改 nominal simulator 的动力学。

## 2. 三条必须区分的实验线

### 2.1 正式 TRACE V10

入口：

```text
launch_go2_trace_v10_formal.sh
```

正式 V10 使用 LLM scorer、分布约束全局 Top-25%、固定 r10 和 8 个 refresh。当前 formal 默认仍使用 reward V1，除非后续 Reward V2 pilot 通过并形成新的冻结协议。

### 2.2 Scorer-free metric 诊断

入口：

```text
launch_go2_trace_v10_metric_pilot.sh
```

该分支从同一批候选中比较：

- tracking 70% + stability 30% 的 metric Top-25%；
- random Top-25%；
- no-TRACE control。

它不使用 LLM scorer，目的是判断问题来自 scorer 还是最终优化目标。它不属于 formal TRACE 结果。

### 2.3 Reward V2 的 RWM-only 验证

入口：

```text
launch_go2_reward_v2_rr05_pilot.sh
launch_go2_reward_v2_rr05_critic_warmup_pilot.sh
launch_go2_reward_v2_rr05_staged_pilot.sh
```

该分支只验证层级 reward 能否先恢复正确 locomotion，再讨论 TRACE replay。它不使用 scorer，也不等同于正式 V10。

其中：

- paired pilot 用相同 actor、seed、5M budget 和评测指令比较 V1/V2；
- critic-warmup pilot 在单次 5M 中先冻结 actor/temperature 3,000 个 critic updates；
- staged pilot 先做 5M critic-only，再完整恢复 critic、replay buffer 和 reward normalizer 做 5M actor finetune。

Reward V2 在验证通过之前不能作为 formal 默认，也不能只改 simbuffer 而不改 RWMbuffer。两路 reward 必须一致。

## 3. Formal 实验矩阵

正式 V10 共 10 个 TRACE policy：

| Side | Conditions |
|---|---|
| sim | `g0`, `rr05`, `rr03`, `p5`, `p75` |
| real | `g0`, `rr05`, `rr03`, `p5`, `p75` |

含义：

- `sim`：RWM 和 25K dataset 来自仿真采集；
- `real`：RWM 和 25K dataset 来自真机采集；
- 两侧 candidate rollout 都在无 gap、无 DR 的 nominal simulator 中进行；
- condition 只决定 dataset、frozen RWM、scorer 和评测 gap；
- `real/g0` 只允许使用 `go2sun_recollect_20260719`，旧 go2-g0 数据会被 launcher 拒绝。

这里的 `real` 不是说训练过程直接在真机在线交互。它表示离线输入数据来自真机，最终 policy 的统一离线评测仍在对应仿真 gap 中完成，之后再选择策略上真机。

## 4. 唯一冻结协议

协议文件：

```text
scripts/reinforcement_learning/rwm_trace/trace_v10_protocol.json
```

关键参数：

| 模块 | V10 参数 |
|---|---|
| Candidate temperature | `T=1.0` |
| Rollout horizon | `H=100` |
| Source starts | `1024` |
| Branches per start | `16` |
| Candidate trajectories | `16,384` |
| Selected trajectories | `4,096` |
| Selection ratio | `25%` |
| Replay ratio | `0.10` |
| Batch size | `2048`，其中 TRACE 固定 `205` 条 |
| Refresh cycles | `8` |
| Steps per refresh | `5M` |
| RWMbuffer maximum | `10M` transitions |
| Retained simbuffer refreshes | 最近 `5` 轮 |
| Initial scorer window | `40` steps，stride `10` |
| Scorer hidden dimension | `32` |
| Scorer seeds | `42,43,44,45,46` |

launcher 只能覆盖：

- 仓库路径；
- 数据、模型和 checkpoint 路径；
- GPU pool；
- run root。

正式冻结参数不应再由不同 shell 环境变量分别覆盖。

## 5. 输入产物与 artifact manifest

模板：

```text
trace_v10_artifacts.example.env
```

每个 `side/condition` 需要：

```text
<SIDE>_<CONDITION>_DATASET
<SIDE>_<CONDITION>_MODEL
<SIDE>_<CONDITION>_WARMUP
```

例如：

```text
REAL_RR05_DATASET=/path/to/real_rr05_25k.pt
REAL_RR05_MODEL=/path/to/model_5000.pt
REAL_RR05_WARMUP=/path/to/common_warmup_checkpoint
```

warmup 至少需要：

```text
actor.pt
agent_state.pt
reward_normalizer.pt
replay_buffer.pt
```

可选 baseline manifest 只有在以下内容全部匹配时才允许复用：

- V10 protocol SHA；
- side/condition；
- dataset SHA；
- frozen RWM SHA；
- warmup actor/replay SHA；
- reward source SHA；
- seed；
- 40M 训练预算。

不满足时自动启动同预算 no-TRACE control，避免将旧 baseline 和新 TRACE 混比。

## 6. 启动前的 pilot 门禁

正式 launcher 要求：

```text
V10_PILOT_GATE=/path/to/pilots_passed.env
```

文件必须包含：

```text
passed=true
```

pilot 顺序：

1. 对 `real/rr05` 和 `real/p5` 做 candidate-only preflight；
2. 检查 candidate 固定配额、动作饱和和 action OOD；
3. preflight 都通过后启动 TRACE 单 refresh pilot；
4. 同时启动相同预算的 no-TRACE control；
5. 对 realization、方向正确率、tracking 和 survival 做配对检查；
6. 两组都通过后才能启动 10 个 formal 条件。

preflight 的冻结上限：

| 指标 | 上限 |
|---|---:|
| selected mean any-joint saturation | `0.35` |
| selected P95 any-joint saturation | `0.50` |
| single action element saturation | `0.05` |
| per-mode action OOD fraction | `0.10` |

preflight 失败后脚本停止，不自动改 T，也不偷偷放松阈值。

## 7. Initial scorer 的构建

入口：

```text
prepare_go2_trace_v10_initial_scorer.sh
```

流程：

```text
condition 25K dataset
  -> 40-step windows, stride 10
  -> 稳定 source group 的 train/validation partition
  -> 同 partition、同 command mode 的 pair pool
  -> Codex/LLM preference labels
  -> 5 个 scorer seeds
  -> 质量门禁后选择最佳 checkpoint
  -> ready.env 原子提交
```

固定要求：

- initial pair pool 总量 `800`；
- 每个 partition/mode 至少 `32` 对；
- 高置信标签至少 `600`；
- 每个 partition/mode 至少 `12` 条高置信标签；
- i/j 每侧至少 `3` 条；
- confidence threshold `0.70`；
- global balanced accuracy 至少 `0.60`；
- 每个 command mode balanced accuracy 至少 `0.55`；
- 每个 mode validation 至少 `12` 对；
- 7 个非 stand mode 的 anti-collapse 固定对必须选择“正确运动”，不能选择“稳定原地扭动”。

V10 先划分稳定 source partition，再在 partition 内构造 pair。这样不会通过大量组合少数窗口来伪造独立样本，也不会让同一 source state 跨 refresh 同时进入 train 和 validation。

标签缓存绑定以下 SHA：

- prompt；
- pair；
- trajectory；
- schema。

任一内容变化时，旧 response 不可复用。

对于 initial dataset 无法提供的 simulator-only privileged features：

- summary 显式写 missing mask；
- scorer 对始终缺失的输入列将首层权重置零；
- 不允许随机初始化的 privileged 权重影响 initial scorer。

最终 `ready.env` 记录 protocol、dataset、condition 和 scorer SHA。condition 或 SHA 不一致时 branch 立即停止。

## 8. 每个 refresh 的完整流程

主循环：

```text
run_go2_trace_v10_branch.sh
```

formal 执行 8 轮，pilot/preflight 只执行 1 轮。

### 8.1 恢复已提交状态

每轮开始前扫描 `refresh_XX/cycle_commit.json`。只有同时验证以下内容后才允许恢复：

- policy checkpoint；
- scorer checkpoint；
- committed replay manifest；
- protocol SHA；
- cycle id。

未提交的部分 policy 目录不会被当成成功结果。检测到非空的 partial output 时 fail-closed，要求人工隔离，避免重启后跳过门禁。

### 8.2 选择 1024 个 source starts

文件：

```text
select_v10_source_starts.py
```

固定 mode 数量：

| Mode | Source starts |
|---|---:|
| stand | 82 |
| pure_x | 256 |
| pure_y | 103 |
| pure_yaw | 82 |
| xy | 143 |
| x_yaw | 174 |
| y_yaw | 51 |
| xy_yaw | 133 |

每个 start 满足：

- 来自当前 condition 的 25K dataset；
- source timestep 至少 `32`；
- episode/history 连续；
- source id 唯一；
- 按固定 seed 可复现。

1024 个 start 分成 4 个 batch，每批 256 个。

### 8.3 在 nominal simulator 一次生成候选

文件：

```text
collect_go2_trace_v10_candidates.sh
```

每个 source start 一次生成 16 个 branches：

```text
1024 x 16 = 16,384 trajectories
```

固定条件：

- `T=1`；
- `H=100`；
- 只使用当前 Policy；
- 无 domain randomization；
- 无 push；
- 无 observation noise；
- payload 固定 0 kg；
- RR calf strength 固定 1.0；
- action delay 为 0；
- simulator action noise 为 0；
- 不使用 append/retry 式补采。

reset 模式：

- sim dataset：`exact_snapshot`；
- real dataset：`canonical_real_projection`。

真实数据没有完整 simulator hidden state，因此 real reset 是可观测状态到 canonical simulator state 的投影，不应宣称和真实下一状态严格一致。

V10 将两类验证分开：

1. `validate_go2_source_reset.py` 用同一环境、同一 action 重放 `(s,a,s')`，验证 reset 和一/多步 transition replay；
2. 正式 TRACE 从 gap dataset 的 `s` reset 到 nominal simulator 后进行反事实 rollout，允许后续轨迹与 gap dataset 分化。

也就是说，reset 验证是为了证明“状态恢复函数正确”；nominal rollout 的分化才是 TRACE 希望利用的 imperfect simulator 信息。

### 8.4 构建 frozen RWM reference

文件：

```text
build_go2_rwm_reference_summaries.py
```

它从相同 source starts 构造 frozen RWM 的 closed-loop reference：

- 使用相同当前 Policy；
- `T=1`、`H=100`；
- 每个实际 batch 重置 FlashSAC action-noise state；
- history 按 episode/timestep 连续恢复；
- 输出与 simulator candidate 可对齐的 summary。

当前 reference 是同起点的 closed-loop reference，不应解释成“整个 100 步强制使用完全相同的开环 action sequence”。

### 8.5 构建 refresh feedback

先将 candidate 转换为包含 tracking、稳定性、位移、摆动、接触、动作饱和和 RWM delta 的 trajectory summaries。

每轮 query budget：

```text
200, 160, 128, 102, 81, 65, 64, 64
```

先生成预算 2 倍的 pair pool，再按 partition/mode 调度当前轮 LLM 请求。每个 partition/mode 至少需要 4 个高置信标签，且 i/j 两侧都必须存在。

标签按 refresh 累积：

```text
initial labels + refresh 1 labels + ... + current refresh labels
```

source group key 在不同 refresh 间保持稳定，避免同一个 dataset state 泄漏到 train 和 validation 两侧。

### 8.6 训练并选择 scorer

每轮使用 seeds `42-46` 训练 5 个 hidden size 32 的 scorer。然后：

1. 检查 global balanced accuracy；
2. 检查 8 个 command modes 的样本数和 balanced accuracy；
3. 检查验证集中 i/j 两类均存在；
4. 运行 7 个 locomotion anti-collapse 固定对；
5. 在通过门禁的 checkpoint 中选择最佳 scorer。

scorer 的职责是给有效候选排序，不是替代 Policy reward，也不是人工编码 RR/payload 控制器。

### 8.7 Validity 和分布约束 Top-25%

文件：

```text
build_trace_replay_v10.py
select_v10_trajectories.py
```

Validity 只拒绝明显无效轨迹：

- NaN/Inf；
- reset reconstruction error 超过 `0.005`；
- terminal length 小于 `20`；
- 非 stand 轨迹净位移小于 `0.001`；
- 激活 linear/yaw 分量 realization 小于 `0.10`；
- 方向错误率大于 `0.50`；
- 任一关节 `|a|>=0.95` 的 timestep 比例大于 `0.50`。

它不是 condition-specific 性能 gate，不根据 RR/payload 手写“哪条轨迹更优”。

最终固定选择 4096 条：

| Mode | Selected trajectories |
|---|---:|
| stand | 328 |
| pure_x | 1024 |
| pure_y | 410 |
| pure_yaw | 328 |
| xy | 573 |
| x_yaw | 696 |
| y_yaw | 205 |
| xy_yaw | 532 |

选择是两阶段约束优化：

1. 在精确 mode 配额下最小化 signed-magnitude marginal 误差；
2. 在最优分布误差内最大化 scorer 排名。

signed magnitude 使用 dataset 已有的正负号 x 3 档幅值：

- x edges：`0.05, 0.20, 0.35, 0.50`；
- y edges：`0.03, 0.087, 0.143, 0.20`；
- yaw edges：`0.05, 0.167, 0.283, 0.40`。

V10 不设 per-start 最小/最大入选数。某个 start 可以不入选，也可以入选多个 branches。只约束总体 command/magnitude 分布和固定总数。

如果任一 mode 无法达到固定配额，流程停止，不允许：

- fallback；
- stand 补 locomotion；
- 缩小 simbuffer 后仍按 r10 高频重复采样；
- 自动追加候选并改变本轮统计分布。

### 8.8 Replay OOD 审计

文件：

```text
audit_trace_replay_ood.py
```

审计使用 mmap/分块读取和 reservoir sampling，检查：

- observation；
- action；
- next observation；
- 每个 feature 的最大偏离；
- command mode 分层分布；
- NaN/Inf。

它不会只用 48 维整体均值掩盖单个关节或 action 维度的严重 OOD。

### 8.9 Simbuffer manifest 与最近 5 轮保留

文件：

```text
v10_replay_manifest.py
```

每个 cycle 先写独立 shard，再生成 staged manifest。Policy 训练完成且 checkpoint 存在后才 commit manifest。

manifest 记录：

- protocol SHA；
- side/condition；
- refresh id；
- shard SHA；
- policy/scorer SHA；
- command mode counts；
- signed bucket counts；
- sampling strategy。

只保留最近 5 个已提交 refresh。例如 cycle 6 使用 cycle 2-6 的 simbuffer shards。过期 shard 只在新 policy 和 manifest 成功提交后滚动删除。

### 8.10 FlashSAC 更新与 RWMbuffer 连续性

每个 2048 batch 固定替换 205 条 TRACE transitions，其余来自持续恢复的 RWMbuffer。

TRACE sampler：

- 按目标 command mode 比例分层；
- 桶内再按 dataset signed-magnitude marginal 校准；
- 不把少量 simbuffer 反复采样成超出协议的实际比例。

RWMbuffer：

- warmup 的最终 buffer 是 cycle 1 起点；
- 每轮从上一轮最终 checkpoint 恢复；
- 最大长度 10M；
- 中间 checkpoint 不重复保存大 buffer；
- 每轮最终 checkpoint 只保存一个 `replay_buffer.pt`；
- 新 cycle 成功提交后才删除上一 cycle 的 buffer；
- common warmup buffer 不删除。

检测到上一轮 final buffer 缺失时立即停止，不能静默清空后重新训练。

### 8.11 每轮行为诊断与原子提交

每轮 Policy 完成后运行 8 类指令行为诊断。诊断记录 locomotion、tracking、survival 和动作表现。

V10 formal 中，普通行为退化只记录曲线，不自动中断整组实验；以下错误会停止：

- NaN/Inf；
- buffer 不连续；
- fixed quota 错误；
- reset 验证错误；
- artifact/protocol/hash 不一致；
- scorer 质量门禁失败；
- replay OOD fail-closed；
- checkpoint 或 manifest 不完整。

最终生成 `cycle_commit.json`，把 policy、scorer、manifest、protocol 和 cycle 绑定起来。只有 commit 完整的 refresh 才能被 resume。

## 9. 最终统一评测

入口：

```text
launch_go2_trace_v10_final_eval.sh
```

固定设置：

- 256 parallel envs；
- 1000 steps；
- 每 50 steps 切换一次 command；
- 20 段 random50 command；
- seeds `400,401,402`；
- baseline 和 TRACE 使用相同 seed 下的相同 command sequence；
- 不跨 gap，只做 paired same-gap comparison。

主要指标：

- survival/termination；
- episode length 和 right-censoring 指标；
- return；
- XY tracking error；
- yaw tracking error；
- uncertainty；
- 按 command mode 的行为表现。

生存显著退化时不能只用 return 判断策略优劣。

## 10. GPU 调度

文件：

```text
run_v10_gpu_stage.sh
```

GPU stage 使用：

- `flock` 操作级独占锁；
- `nvidia-smi` 显存检查；
- 物理 GPU pool；
- GPU 阶段结束立即释放锁；
- LLM labeling、JSON 处理和 scorer CPU 阶段不长期占卡。

默认只在显存占用不超过 2048 MiB 时领取 GPU。领取失败时轮询等待，不和同一 V10 pool 中的其他 stage 抢卡。

## 11. V9 到 V10 的主要工作

这里的 V9 指此前服务器上经过多轮临时修补的实现。V10 从干净 Git 基线重建，没有复制 V8/V9 的 run root、label cache、scorer checkpoint 或 replay shard。

| V9 已确认问题 | V10 处理 |
|---|---|
| 多份脚本和环境变量互相覆盖，实际参数难确认 | 建立 `trace_v10_protocol.json` 作为唯一冻结参数源，并对 protocol SHA 做全链路校验 |
| `T=2` 候选动作大量饱和 | 正式协议改为 `T=1`，并加入 mean/P95/element saturation preflight |
| 每个 start 候选不足后反复 append/retry，耗时且改变样本统计 | 固定 1024 starts x 16 branches，一次生成 16,384 条，不做自动补采 |
| global Top-25% 容易偏向 stand、低速和简单 command | 保留全局 scorer 排序，但增加精确 mode 配额和 dataset signed-magnitude marginal 约束 |
| mode quota 在筛选后执行，可能只删除不补位，最终数量缩水 | V10 selector 在 eligible pool 上做固定 4096 条约束优化，数量或任一 mode 不足即停止 |
| fallback 会把明显不运动、原地扭动的轨迹补入 replay | 删除 fallback，只保留极简 validity；不足时 fail-closed |
| initial scorer 使用长窗口且稀有 mode pair 不足 | initial window 改为 40、stride 10；pair pool 800；partition/mode 固定最低配额 |
| 通过少量窗口两两组合伪造大量 pair，train/val 无法独立 | 先按稳定 source group 划分 partition，再在 partition/mode 内构造 pair |
| scorer 只检查总体 accuracy 或少数 mode | 检查 8 个 mode 的 count、i/j 双类别和 balanced accuracy，并增加 7 个 anti-collapse 固定对 |
| scorer hidden size 过大，相对标签量容易过拟合 | hidden dimension 固定为 32，并训练 5 个 seeds 后门禁选择 |
| initial scorer 看不到新增 privileged 特征，随机权重影响排序 | 加 missing mask，始终缺失的输入列首层权重显式置零 |
| LLM 标签缓存只按 pair id 复用，pair 内容变化后可能错配 | 缓存绑定 prompt/pair/trajectory/schema SHA |
| 标签不足时无限 repair 或一直等待 | 固定 pair pool、标签预算和 relabel 上限；reserve 用尽仍不足则停止 |
| refresh source key 随 candidate SHA 改变，跨轮 train/val 泄漏 | source group 基于稳定 dataset/source state，跨 cycle 保持一致 |
| RR calf 索引一度错误地读到左后腿 | 统一使用正确的 RR calf state index 20，并纳入 trajectory diagnostics |
| action-noise state 在不同 reference batch 间残留，最后小 batch 还可能维度不匹配 | 每个实际 batch 按真实 batch size 调用 `reset_action_noise()` |
| simulator reset 曾遗漏 actuator force/forward，起点误差很大 | collector 补全 action/actuator state 写入和 simulator forward，并提供专门 reset replay validator |
| simbuffer 额外覆盖 terminal reward、failure backprop、action delta 和 saturation penalty | formal V10 replay 协议将这些额外项全部关闭，保留自然 terminal reward |
| `failure_transition_ratio=0` 被实现成排除所有自然 failure | V10 不使用强制 failure quota；自然 termination 可由 scorer 正常排序 |
| RWMbuffer 在 refresh 间可能丢失，或每个中间 checkpoint 重复保存大 buffer | final-only buffer、连续恢复、原子提交、成功后滚动删除上一轮 buffer |
| replay 只保留单轮或来源不清 | 独立 cycle shards、原子 manifest、最近 5 轮 retention、policy/scorer/protocol SHA |
| OOD 只看 observation 整体均值，单维 action/next-state 偏离可被掩盖 | mmap/流式审计 observation、action、next observation、per-feature 和 per-mode 分布 |
| behavior gate 失败后重启可能被 completed marker 绕过 | `cycle_commit.json` 必须绑定 policy/scorer/manifest SHA，resume 前重新 verify |
| formal launcher 可能仍指向旧 V7/V8/V9 目录 | V10 launcher 使用相对仓库路径和显式 artifact manifest，不加载旧 protocol 产物 |
| 旧 real/g0 来自另一台 go2，接触监督明显不对称 | formal 明确拒绝 legacy go2-g0，只接受新 go2sun recollection |
| 只有 TRACE 继续训练，没有严格同预算 no-TRACE control | baseline manifest 严格校验，不一致就重跑同 seed、同 budget、同 reward control |
| 任务并发时会抢 GPU | 增加显存门限和 `flock` GPU stage 锁 |
| 脚本错误经常运行数小时后才暴露 | 增加 Python compile、shell heredoc compile、`bash -n`、协议、selector、manifest、resume、OOD 和 reward 回归测试 |

## 12. Reward 方面新增但尚未升级 formal 的工作

V9/V10 诊断发现旧 tracking reward 的指数核过宽，非零 command 下静止仍可获得很高分，再叠加 effort、smoothness 和 uncertainty penalty，Policy 容易学成“稳定但不 locomotion”。

因此代码新增 `v2_hierarchical`：

```text
先产生正确方向的有效响应
  -> 再优化速度 tracking
  -> 动起来后再主要优化稳定、平滑和能耗
```

V2 包含：

- command response；
- yaw response；
- wrong-direction penalty；
- response shortfall；
- active-command bias；
- continuous motion gate；
- gated effort penalties；
- stand 和 active command 分开处理。

但当前状态必须明确：

- V2 已有代码和单元测试；
- 已有 paired/critic-warmup/staged pilot 入口；
- V2 尚未成为 formal V10 默认；
- 只有 RWM-only control 证明 locomotion、tracking 和 survival 均合理后，才能建立新协议让 RWMbuffer/simbuffer 同时切换 V2；
- 不能拿 V1 replay buffer 或 V1 reward normalizer 直接续训 V2。

## 13. 推荐代码阅读顺序

### 第一层：理解协议和总流程

1. `GO2_TRACE_V10_WORKFLOW_zh.md`
2. `trace_v10_protocol.json`
3. `launch_go2_trace_v10_pilots.sh`
4. `launch_go2_trace_v10_formal.sh`
5. `run_go2_trace_v10_condition.sh`
6. `run_go2_trace_v10_branch.sh`

### 第二层：理解 candidate 与 reset

1. `select_v10_source_starts.py`
2. `collect_go2_trace_v10_candidates.sh`
3. `collect_go2_expert_command_coverage_dataset.py`
4. `validate_go2_source_reset.py`
5. `audit_v10_candidate_set.py`
6. `build_go2_rwm_reference_summaries.py`

### 第三层：理解 scorer

1. `build_go2_dataset_window_summaries.py`
2. `build_go2_partitioned_feedback_pairs.py`
3. `label_feedback_with_codex.py`
4. `train_go2_trace_scorer.py`
5. `select_go2_scorer_checkpoint.py`
6. `validate_v10_scorer.py`

### 第四层：理解 replay 和 Policy update

1. `build_trace_replay_v10.py`
2. `select_v10_trajectories.py`
3. `audit_trace_replay_ood.py`
4. `v10_replay_manifest.py`
5. `flash_rl/agents/flashSAC/agent.py`
6. `scripts/reinforcement_learning/rwm_trace/replay.py`
7. `04_train_normal_policy.sh`
8. `train_flashsac_world_model_go2.py`
9. `v10_cycle_commit.py`

### 第五层：理解评测和 Reward V2

1. `launch_go2_trace_v10_final_eval.sh`
2. `run_go2_command_eval_v4.sh`
3. `summarize_command_eval_v4.py`
4. `src/tasks/rwm_velocity/mdp/rewards.py`
5. `trace_v10_reward_v2_pilot.json`
6. `launch_go2_reward_v2_rr05_pilot.sh`

## 14. 典型 run root

```text
<V10_RUN_BASE>/
  initial_scorers/
    real_rr05/
      dataset_windows.jsonl
      pairs.jsonl
      labels/
      seeds/
      go2_trace_scorer.pt
      scorer_validation.json
      ready.env
  sim/
    rr05/
      state.txt
      run.log
      replay_manifest.json
      refresh_01/
        source_ids.pt
        candidate_audit.json
        rwm_reference.jsonl
        summaries.jsonl
        feedback/
        replay/
          trace_shard.pt
          ood_audit.json
          committed_manifest.json
        policy/
          stage/
          run/
        behavior/
        cycle_commit.json
      ...
      refresh_08/
      completed.txt
  baselines/
    sim/rr05/
      v10_baseline_manifest.json
```

## 15. 启动示例

### Pilot

```bash
export V10_ARTIFACT_ENV=/absolute/path/trace_v10_artifacts.env
export V10_RUN_BASE=/absolute/path/go2_trace_v10_pilots
export V10_GPU_POOL=1,2,3,4,5,6,7

bash scripts/reinforcement_learning/go2_sim_gap_aligned/launch_go2_trace_v10_pilots.sh
```

### Formal

```bash
export V10_ARTIFACT_ENV=/absolute/path/trace_v10_artifacts.env
export V10_RUN_BASE=/absolute/path/go2_trace_v10_formal
export V10_GPU_POOL=1,2,3,4,5,6,7
export V10_PILOT_GATE=/absolute/path/pilots_passed.env

bash scripts/reinforcement_learning/go2_sim_gap_aligned/launch_go2_trace_v10_formal.sh
```

### 最终评测

```bash
export V10_ARTIFACT_ENV=/absolute/path/trace_v10_artifacts.env
export V10_RUN_BASE=/absolute/path/go2_trace_v10_formal
export V10_EVAL_ROOT=/absolute/path/go2_trace_v10_eval
export V10_GPU_POOL=1,2,3,4,5,6,7

bash scripts/reinforcement_learning/go2_sim_gap_aligned/launch_go2_trace_v10_final_eval.sh
```

## 16. 查看进度与日志

查看 session：

```bash
tmux list-sessions
```

不进入 tmux 查看日志：

```bash
tail -f <RUN_ROOT>/run.log
```

查看结构化进度：

```bash
cat <RUN_ROOT>/state.txt
```

`state.txt` 包含：

```text
protocol
side
condition
cycle
stage
status
progress_percent
gpu_pool
```

进度百分比按已提交 refresh 计算，不把“已经生成部分文件但尚未 commit”当成完成。

## 17. 提交前验证

至少执行：

```bash
python -m py_compile \
  src/tasks/rwm_velocity/mdp/rewards.py \
  scripts/reinforcement_learning/rwm_trace/build_trace_replay_v10.py

python -m unittest discover -s tests -p 'test_go2_reward_v2.py' -v
python -m unittest discover -s tests -p 'test_trace_v10_protocol.py' -v
python -m unittest discover -s tests -p 'test_trace_ood_audit.py' -v
python -m unittest discover -s tests -p 'test_flashsac_actor_warmup.py' -v
```

V10 protocol tests还覆盖：

- shell heredoc Python 编译；
- fixed protocol values；
- source starts 固定分批；
- selector 精确 mode quota；
- global metric pilot 不恢复 command quota；
- last-five replay manifest；
- 每 batch 205 条 TRACE transitions；
- 跨 refresh partition 稳定；
- 旧 protocol/replay 拒绝加载。

## 18. 当前结论与边界

V10 代码已经完成的是“协议重建和实现收敛”，不是“已经证明 TRACE 一定超过 RWM baseline”。

当前可以确认：

- formal、pilot、baseline、metric diagnostic 和 reward diagnostic 已分离；
- V9 已确认的主要实现错误已在 V10 中有对应防护；
- 关键 artifact 通过 SHA 和原子 commit 绑定；
- 旧 V8/V9 缓存不能静默进入 V10；
- V10 能产生可审计、可恢复、可配对比较的实验结果。

当前仍需用实验回答：

- Reward V2 是否能消除 RWM-only control 的少动/站桩解；
- scorer 在 8 个 command modes 上是否稳定优于 random selection；
- nominal simulator 轨迹在 RR 和 payload gap 中是否提供真正互补信息；
- TRACE r10 是否在 sim 和真机部署中稳定超过配对 RWM baseline；
- 哪些 condition 的提升具有多训练 seed 的统计可靠性。

因此后续结果报告必须分别说明：

1. 使用的 protocol SHA；
2. reward version；
3. dataset/RWM/warmup SHA；
4. baseline 是否严格匹配；
5. scorer-free、Reward V2 还是 formal TRACE；
6. sim gap-only 评测还是真机部署结果。
