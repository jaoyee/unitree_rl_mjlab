# wty-yy Go2 Full DR 配置与实现分析

## 1. 分析范围

本文分析以下仓库和分支中的 Go2 域随机化实现：

- 仓库：`https://github.com/wty-yy/go2_rl_gym`
- 分支：`vanilla_train`
- 本地分析 commit：`34c237ba7d34cc8ff4e23e2952e1f420aecd901b`
- 主要配置：`legged_gym/envs/go2/go2_config.py`
- 基类实现：`legged_gym/envs/base/legged_robot.py`
- 基类默认值：`legged_gym/envs/base/legged_robot_config.py`
- Go2 观测实现：`legged_gym/envs/go2/go2_env.py`

分析时必须同时阅读配置和执行代码。配置中的 `randomize_xxx = True` 只能说明功能被打开，具体随机变量是否真正作用于动力学、在什么时候重新采样、是否存在裁剪顺序问题，取决于 `legged_robot.py` 中的实现。

## 2. 总体结论

这套配置不是单一的“物理参数随机化”，而是一套面向直接 sim-to-real locomotion policy 训练的综合随机化体系：

```text
朋友 Full DR
= 物理参数 DR
+ 执行器 DR
+ 控制延迟 DR
+ Observation DR
+ 外部扰动
+ 初始状态随机化
+ 地形和任务分布扩展
```

其主要特点是：

1. 覆盖范围完整，既考虑机器人动力学误差，也考虑电机、控制器、传感器和通信误差。
2. 部分范围很强，例如摩擦系数下界为 0、motor strength 为 0.8--1.2、质心偏移可达 3 cm。
3. 原项目使用 8192 个并行环境，因此即使部分物理参数只在环境创建时采样，也能获得较广的分布覆盖。
4. 该配置面向“policy 直接与物理仿真器交互”的训练方式，不能不加区分地复制到 RWM pipeline 的所有阶段。
5. 原项目最终鲁棒性不只来自 `domain_rand`，还来自复杂地形、初始状态随机化、广指令分布和大量并行环境。

## 3. DR 参数总览

`GO2Cfg.domain_rand` 中显式打开的随机化如下：

| 类别 | 参数 | Full 范围 | 采样时机 |
| --- | --- | --- | --- |
| 接触 | friction | `[0.0, 2.0]` | 环境创建 |
| 刚体 | base mass | 标称值加 `[-1, 1] kg` | 环境创建 |
| 刚体 | link mass | 每个 link 乘 `[0.9, 1.1]` | 环境创建 |
| 刚体 | base COM | x/y/z 各加 `[-0.03, 0.03] m` | 环境创建 |
| 接触 | restitution | `[0.0, 0.5]` | 环境创建 |
| 执行器 | Kp multiplier | `[0.9, 1.1]` | 每次 episode reset |
| 执行器 | Kd multiplier | `[0.9, 1.1]` | 每次 episode reset |
| 执行器 | motor zero offset | `[-0.035, 0.035] rad` | 每次 episode reset |
| 执行器 | motor strength | `[0.8, 1.2]` | 每次 episode reset |
| 外扰 | base push linear velocity | x/y 为 `[-0.4, 0.4] m/s` | 每 4 s |
| 外扰 | base push angular velocity | x/y/z 为 `[-0.6, 0.6] rad/s` | 每 4 s |
| 控制 | action delay | `0/5/10/15/20 ms` | 每个 policy step |
| 观测 | actor observation noise | 分量相关 | 每个 observation step |

## 4. 参数生效周期

理解采样周期对迁移到数据采集任务非常重要。

### 4.1 环境创建时固定

以下参数在 `_create_envs()` 调用属性处理函数时采样：

- friction
- restitution
- base mass
- link mass
- base COM

同一个并行环境中的这些参数在后续 episode reset 时不会重新采样。原项目依靠 8192 个并行环境产生大量固定 domain。对于并行环境数量较少的数据采集器，如果仍沿用该实现，可能出现物理 domain 覆盖不足。

### 4.2 每个 episode 重新采样

以下参数在 `reset_idx(env_ids)` 中对发生 reset 的环境重新采样：

- motor strength
- motor zero offset
- Kp multiplier
- Kd multiplier

这些值均按环境、按关节独立采样，并在该 episode 内保持不变。这符合电机标定、硬件差异和控制器参数通常在短轨迹内相对固定的事实。

### 4.3 每个 policy step 重新采样

- action delay
- observation noise

Action delay 每个 20 ms policy step 重新采样；observation noise 每次构造 actor observation 时重新采样。

### 4.4 固定间隔触发

- push 每 4 s 触发一次。

Push 不是施加短时间的物理力，而是直接重设 base 的线速度和角速度。

## 5. 物理 DR 详细分析

### 5.1 Friction randomization

配置：

```python
randomize_friction = True
friction_range = [0.0, 2.0]
```

实际实现：

1. 创建 64 个从 `[0, 2]` 均匀采样的 friction bucket。
2. 每个并行环境随机选择一个 bucket。
3. 将该摩擦值写入机器人所有 rigid shape 的 `friction` 属性。
4. 该值在环境创建后保持固定。

设计目的：

- 模拟不同地面材料。
- 模拟脚垫磨损、灰尘和表面污染。
- 覆盖 Isaac Gym 接触模型与真实接触之间的误差。
- 防止策略过度依赖单一摩擦系数形成特定步态。

注意事项：

- `[0, 2]` 是很强的范围，下界 0 接近完全无摩擦。
- 对平地 Go2 实机部署，极低摩擦样本可能降低正常地面上的训练效率。
- 代码随机化的是机器人碰撞 shape 的摩擦值；最终接触摩擦还与地面参数和 PhysX 的 friction combine 行为有关。
- 在我们的平地第一轮实验中，不建议直接使用 0 作为下界。

### 5.2 Base mass randomization

配置：

```python
randomize_base_mass = True
added_mass_range = [-1.0, 1.0]
```

实际实现：

```text
base_mass_randomized = base_mass_nominal + Uniform(-1, 1) kg
```

设计目的：

- 模拟电池电量、不同电池型号和安装差异。
- 模拟计算平台、相机、雷达、线缆等附加载荷。
- 覆盖 URDF 机身质量标定误差。

对于约 15 kg 的 Go2，`+-1 kg` 已经是明显扰动，但通常仍处在可训练范围。

### 5.3 Link mass randomization

配置：

```python
randomize_link_mass = True
multiplied_link_mass_range = [0.9, 1.1]
```

实际实现：

- base link 不使用该 multiplier。
- 除 base 外，每个 rigid body 独立乘一个 `[0.9, 1.1]` 随机系数。

设计目的：

- 覆盖腿部零件、连杆、关节壳体的质量误差。
- 改变各腿的惯性分布，防止策略只适应完全对称的模型。
- 模拟线缆和装配件造成的局部质量差异。

独立按 link 采样比整机统一缩放更强，因为它会破坏四腿之间的完全对称性。

### 5.4 Base COM randomization

配置：

```python
randomize_base_com = True
added_base_com_range = [-0.03, 0.03]
```

实际实现：

- base COM 的 x、y、z 三个方向分别独立增加 `[-3, 3] cm`。
- 只修改 base COM，不直接修改每条腿 link 的 COM。

设计目的：

- 模拟电池、计算平台和传感器安装位置变化。
- 覆盖 URDF 质心标定误差。
- 迫使策略学习更强的姿态稳定和负载转移能力。

`3 cm` 对四足机器人属于明显扰动，特别是横向 COM 偏移会持续改变左右腿负载。

### 5.5 Restitution randomization

配置：

```python
randomize_restitution = True
restitution_range = [0.0, 0.5]
```

实际实现：

- 每个环境采样一个 restitution。
- 同一环境内机器人所有 rigid shape 使用同一个 restitution。

设计目的：

- 模拟硬地、软垫、橡胶脚垫等不同碰撞回弹特性。
- 覆盖仿真接触模型与真实脚地碰撞之间的差异。

该范围上界 0.5 较大，可能增加脚端触地后的弹跳和高频运动。平地部署实验可先从更窄范围开始。

## 6. 执行器 DR 详细分析

### 6.1 标称 PD 控制器

朋友项目中的标称控制配置为：

```python
control_type = "P"
stiffness = {"joint": 20.0}
damping = {"joint": 0.5}
action_scale = 0.25
decimation = 4
```

仿真 timestep 为 `5 ms`，4 个 physics step 对应一个 `20 ms` policy step，即 policy 频率为 50 Hz。

Policy action 被解释为目标关节位置增量：

```text
q_target = default_joint_position + 0.25 * action + motor_zero_offset
```

### 6.2 PD gain randomization

配置：

```python
randomize_pd_gains = True
stiffness_multiplier_range = [0.9, 1.1]
damping_multiplier_range = [0.9, 1.1]
```

每个 episode、每个关节分别采样 Kp 和 Kd multiplier：

```text
Kp_effective = Kp_nominal * lambda_p
Kd_effective = Kd_nominal * lambda_d
```

设计目的：

- 模拟实机控制增益误差。
- 模拟电机、减速器和速度估计对闭环响应的影响。
- 提高 policy 对“同一目标角度产生不同动态响应”的鲁棒性。

Kp/Kd 按关节独立采样能够模拟关节间差异，但比整机统一变化更难。

### 6.3 Motor zero offset randomization

配置：

```python
randomize_motor_zero_offset = True
motor_zero_offset_range = [-0.035, 0.035]
```

`0.035 rad` 约等于 2 度。偏差直接加入 PD 控制器的目标位置误差项。

设计目的：

- 模拟编码器零点标定误差。
- 模拟机械装配和关节安装角偏差。
- 模拟仿真默认姿态与实机实际零位之间的差异。

该偏差在 episode 内固定，而不是每步变化，因此其物理含义是 calibration bias，不是传感器白噪声。

### 6.4 Motor strength randomization

配置：

```python
randomize_motor_strength = True
motor_strength_range = [0.8, 1.2]
```

每个 episode、每个关节独立采样 strength multiplier。

设计目的：

- 模拟电池电压变化。
- 模拟不同电机与减速器效率。
- 模拟关节摩擦、传动损耗和温度影响。
- 模拟单个关节弱于或强于标称模型的情况。

实际力矩处理顺序是：

```text
1. 根据随机后的 Kp/Kd 计算 PD torque
2. clip 到 URDF torque limits
3. torque *= motor_strength
4. 将 torque 发送给仿真器
```

这里存在一个实现细节：motor strength 在 torque clipping 之后相乘。当 strength 为 1.2 时，最终发送力矩可能达到 URDF torque limit 的 1.2 倍。因此它不完全等价于真实电机能力变化，更像“裁剪后力矩缩放”。

更严格的实现可以选择：

```text
torque = clip(motor_strength * torque_raw, -torque_limit, torque_limit)
```

或者同时随机化有效 torque limit。采用哪种方式必须在实验配置中明确记录。

### 6.5 Action delay randomization

配置：

```python
randomize_action_delay = True
```

具体实现不是 FIFO action queue，而是在每个 20 ms policy interval 内随机选择从哪个 physics substep 开始使用新 action。

```text
start_decimation = UniformInteger{0, 1, 2, 3, 4}

0: 4 个 substep 全部使用新 action，0 ms delay
1: 第 1 个 substep 使用旧 action，约 5 ms delay
2: 前 2 个 substep 使用旧 action，约 10 ms delay
3: 前 3 个 substep 使用旧 action，约 15 ms delay
4: 4 个 substep 全部使用旧 action，约 20 ms delay
```

设计目的：

- 模拟 policy 推理延迟。
- 模拟通信、SDK 和控制线程抖动。
- 模拟 action 从上层策略到关节执行器之间的延迟。

注意：延迟值每个 policy step 重新采样，因此模拟的是强时变 jitter，不是每台机器人固定的延迟。

## 7. 外部扰动

配置：

```python
push_robots = True
push_interval_s = 4
max_push_vel_xy = 0.4
max_push_ang_vel = 0.6
```

实际实现：

```text
base linear velocity x/y <- Uniform(-0.4, 0.4) m/s
base angular velocity x/y/z <- Uniform(-0.6, 0.6) rad/s
```

每 4 秒触发一次。

设计目的：

- 训练机器人在碰撞和姿态受扰后的恢复能力。
- 防止 policy 只在平稳、无干扰状态下工作。
- 提升实机被触碰、地面打滑或落脚不稳定时的生存率。

实现上的重要区别：它直接覆盖 base velocity，而不是对 base 施加一个持续若干毫秒的物理 force/impulse。因此它是一种有效但较粗略的扰动模型。

## 8. Observation DR 详细分析

Go2 配置只显式设置：

```python
class noise(LeggedRobotCfg.noise):
    add_noise = True
```

噪声强度继承自 `LeggedRobotCfg.noise`：

```python
noise_level = 1.0
dof_pos = 0.01
dof_vel = 1.5
lin_vel = 0.1
ang_vel = 0.2
gravity = 0.05
height_measurements = 0.1
```

Go2 actor observation 为 45 维：

```text
base_ang_vel[3]
projected_gravity[3]
velocity_commands[3]
joint_pos_rel[12]
joint_vel[12]
previous_action[12]
```

噪声通过下面的形式逐步独立采样：

```text
observation += Uniform(-1, 1) * noise_scale_vector
```

换算到原始物理量后的范围为：

| Actor 输入 | 原始物理量噪声 | 是否加噪 |
| --- | --- | --- |
| base angular velocity | `+-0.2 rad/s` | 是 |
| projected gravity | `+-0.05` | 是 |
| velocity commands | 0 | 否 |
| joint position | `+-0.01 rad` | 是 |
| joint velocity | `+-1.5 rad/s` | 是 |
| previous action | 0 | 否 |

设计目的：

- base angular velocity：模拟 IMU 陀螺仪噪声和振动。
- projected gravity：模拟姿态估计误差。
- joint position：模拟编码器量化和标定误差。
- joint velocity：模拟差分速度估计中的高频噪声。
- commands 不加噪：command 是控制器内部已知目标，不是外部传感器测量。
- previous action 不加噪：它是策略端已知的上一条命令。

Joint velocity 的 `+-1.5 rad/s` 强度较大。对朋友项目的直接物理训练可能合理，但如果原样放入 frozen RWM-policy rollout，可能显著放大 model exploitation 和状态分布偏移。

## 9. 不在 domain_rand 中但实际影响鲁棒性的随机化

### 9.1 关节初始位置

每次 reset 时：

```text
q_initial = q_default * Uniform(0.5, 1.5)
```

这会显著扩展初始姿态分布，训练 policy 从不同关节姿态恢复。

注意：这是对默认关节角做乘法，而不是围绕默认角增加固定弧度范围。对于正负角度和接近 0 的 hip 角，其实际扰动幅度不同。

### 9.2 Base 初始速度

每次 reset 时，base 的 3 维线速度和 3 维角速度均从 `[-0.5, 0.5]` 采样。

这使机器人不会总从完全静止状态开始，增强起步和扰动恢复能力。

### 9.3 Terrain distribution

基类使用 trimesh terrain 并开启 terrain curriculum。Go2 配置的地形比例为：

| 地形 | 比例 |
| --- | ---: |
| wave | 5% |
| slope | 20% |
| rough slope | 5% |
| stairs up | 25% |
| stairs down | 10% |
| obstacles | 20% |
| stepping stones | 0% |
| gap | 0% |
| flat | 15% |

`max_init_terrain_level = 5`，且基类 `terrain.curriculum = True`。因此该项目不是主要面向平地训练，85% 的地形采样来自非 flat 类型。

复杂地形会显著扩大状态、接触和恢复动作分布。朋友策略的实机鲁棒性不能全部归因于显式 DR 参数。

### 9.4 Command distribution

指令每 5 秒重采样，范围为：

```text
lin_vel_x: [-2.0, 2.0] m/s
lin_vel_y: [-1.0, 1.0] m/s
ang_vel_yaw: [-2.0, 2.0] rad/s
```

不同地形还有独立的最大指令范围。零指令概率从 0 逐步增加到 0.1。

它不是动力学 DR，但会决定 policy 和 dataset 覆盖的任务状态分布。

## 10. 为什么这套 DR 强度足够高

从单项范围和组合效果看，这套配置属于较强 DR：

1. 摩擦允许下降到 0。
2. Base COM 可在三个方向偏移 3 cm。
3. 每条腿 link mass、PD gains、motor strength 和 zero offset 都可以独立变化。
4. Motor strength 可产生 `+-20%` 力矩缩放。
5. 每个 action 都可能经历 0--20 ms 的随机 delay。
6. Joint velocity observation noise 达到 `+-1.5 rad/s`。
7. 每 4 秒直接修改一次 base velocity。
8. 训练中只有 15% flat terrain。
9. 初始关节角和 base velocity 也被随机化。

这些扰动同时出现时，难度不是各项难度的简单相加。尤其是低摩擦、弱 motor、COM 偏移、action delay 和 observation noise 同时出现时，训练早期容易频繁跌倒。

## 11. 实现中的关键风险和限制

### 11.1 Full DR 不等于适合所有 pipeline

原项目中的 policy 直接与真实物理仿真器交互。即使 DR 较强，policy 仍能从物理环境获得一致的 next state 和 reward。

在 RWM pipeline 中，强 DR 会先影响 dataset，再影响 world model 的条件分布。如果 RWM 没有显式 domain context，相同 `(state, action)` 在不同物理 domain 下可能对应不同 next state，RWM 可能学习到平均化转移。

### 11.2 物理参数不在每个 episode 重采样

Base mass、link mass、COM、friction 和 restitution 只在环境创建时采样。原项目依靠 8192 envs 覆盖分布。我们的 collector 如果环境数较少，必须验证是否获得足够多不同物理 domain。

### 11.3 Motor strength 的 clipping 顺序

代码先 clip torque，再乘 motor strength。强度大于 1 时可能突破 URDF torque limit。迁移时应决定是否保留原始行为，不能无意改变实验定义。

### 11.4 Push 不是严格物理冲击

Push 直接重写速度，不能等价为特定大小和持续时间的外力。它适合训练恢复能力，但不能直接解释为真实推力标定。

### 11.5 Action delay 是 step-level jitter

每个 policy step 都重新采样 delay。真实系统可能同时包含固定延迟和时变 jitter。若面向实机，可以拆成：

```text
episode-level base delay + step-level small jitter
```

### 11.6 Per-joint 独立采样可能偏强

真实电池电压通常同时影响全部电机，而代码中的 motor strength 每个关节独立。它能提高鲁棒性，但不完全符合硬件误差的相关结构。

## 12. 映射到我们的三阶段 RWM pipeline

我们的流程为：

```text
Expert training
-> Expert dataset collection
-> RWM training
-> Frozen RWM and final policy interaction
-> Final policy
```

### 12.1 Expert training 阶段

MJLab 中存在物理引擎和电机模型，因此可以使用：

- friction、mass、COM、restitution 等物理 DR
- PD gains、motor strength、zero offset 等执行 DR
- action delay
- observation noise
- push

但如果当前目标是研究 RWM 和最终 policy，而不是改进 expert，本阶段应优先保证 expert 的性能和数据采集能力。强 Full DR expert 可能降低数据质量，并引入额外实验变量。

### 12.2 Dataset collection 阶段

该阶段仍在 MJLab 物理环境中，因此是最适合完整应用朋友 DR 的阶段：

- 物理 DR 改变真实 next-state transition。
- 执行器 DR 改变 policy action 到实际 torque 的映射。
- Action delay 改变有效执行动作。
- Observation noise 改变 expert 实际看到的状态。

建议同时保存：

```text
clean simulator state
noisy policy observation
raw policy action
effective delayed/scaled action
episode-level DR parameters
```

其中 RWM 的监督标签优先使用 clean state/next state；否则传感器噪声可能被 RWM 当作随机动力学学习。

### 12.3 Frozen RWM-policy 阶段

这里没有 MJLab 物理引擎，因此不能真正改变：

- 质量和质心
- 地面摩擦和 restitution
- link inertia
- 真实 PD controller
- 接触模型

可以在接口层加入：

```text
policy action
-> delay
-> multiplicative scale
-> per-joint bias
-> additive noise
-> low-pass filter
-> clipping
-> frozen RWM
```

这属于“等效执行 DR”，不是完整物理执行器 DR。

Observation 侧可以加入分量相关的 noise、bias、delay 或 dropout，但不能期望 frozen RWM 凭空预测 dataset 中没有覆盖的物理 domain。

## 13. Full 与 Half DR 的建议定义

Half DR 应围绕正常标称值缩小范围，而不是把所有上下界直接除以 2。

| 参数 | 朋友 Full | 建议 Half |
| --- | ---: | ---: |
| friction | `[0.0, 2.0]` | 建议围绕标称摩擦重新设定，例如 `[0.5, 1.5]` |
| base added mass | `[-1, 1] kg` | `[-0.5, 0.5] kg` |
| link mass multiplier | `[0.9, 1.1]` | `[0.95, 1.05]` |
| base COM | `[-0.03, 0.03] m` | `[-0.015, 0.015] m` |
| restitution | `[0.0, 0.5]` | `[0.0, 0.25]` |
| Kp multiplier | `[0.9, 1.1]` | `[0.95, 1.05]` |
| Kd multiplier | `[0.9, 1.1]` | `[0.95, 1.05]` |
| motor zero offset | `[-0.035, 0.035] rad` | `[-0.0175, 0.0175] rad` |
| motor strength | `[0.8, 1.2]` | `[0.9, 1.1]` |
| push linear velocity | `+-0.4 m/s` | `+-0.2 m/s` |
| push angular velocity | `+-0.6 rad/s` | `+-0.3 rad/s` |
| action delay | `0--20 ms` | 保留 `0--20 ms` 但降低非零延迟概率，或固定 `0/10 ms` |
| observation noise | 原始幅度 | 各分量幅度乘 0.5 |

Action delay 是离散时序结构，不能简单把“20 ms 除以 2”视为唯一正确的 Half 定义。需要明确是减小最大 delay，还是降低 delay 发生概率。

## 14. 对 36 组全因子实验的意义

计划中的实验为：

```text
Expert DR:  default / friend-half / friend-full
Dataset DR: default / friend-half / friend-full
Final interface: clean / action-noise / obs-noise / action+obs-noise
```

因此：

```text
3 x 3 x 4 = 36 final policies
```

该设计可以分析：

- 强 expert DR 是否提升后续数据质量，还是降低 expert 本身性能。
- Dataset DR 强度是否决定 RWM 的跨 domain 泛化。
- Final action/obs noise 是否只在对应 dataset 已覆盖扰动时有效。
- Expert DR、Dataset DR 和 final interface noise 是否存在交互效应。

完整任务数实际上为：

```text
3 experts
9 datasets
9 RWMs
36 final policies
```

实验必须复用上游结果，不能为每个 final policy 重复训练 expert 和 RWM。

## 15. 建议记录的 DR 元数据

为了让 RWM 数据可分析，每个 transition 或 episode 至少应能追溯：

- expert checkpoint 和 expert DR profile
- dataset DR profile
- environment id 和 episode id
- friction
- base added mass
- link mass multipliers
- base COM offset
- restitution
- Kp/Kd multipliers
- motor zero offsets
- motor strengths
- action delay
- raw action
- effective action
- clean observation/state
- noisy policy observation
- clean next state
- termination reason

即使第一版 RWM 不把 DR 参数作为输入，也应保存这些元数据，便于后续分析 RWM 在不同 domain 下的误差。

## 16. 最终评价建议

不能只用训练 reward 判断 DR 是否更好。至少需要比较：

- clean environment reward
- default DR reward
- friend Half DR reward
- friend Full DR reward
- holdout DR reward
- episode length
- fall rate
- zero-command standing stability
- linear/angular velocity tracking error
- roll/pitch 分布和峰值
- action rate 和 action smoothness
- joint tracking error `q_des - q`
- torque saturation ratio
- 不同 seed 下均值和方差

对于面向真机的策略，应优先选择：

```text
clean 性能不过度下降
+ DR 测试均值较高
+ 最差 domain 表现较好
+ fall rate 较低
+ action/torque 平滑
+ 多 seed 方差较小
```

而不是只选择单一环境中最高 reward 的模型。

## 17. 核心结论

1. 朋友配置是一套完整且偏强的 sim-to-real DR。
2. 其 DR 包含物理、执行器、控制延迟、观测、外扰和任务分布多个层面。
3. Full DR 的效果依赖 8192 并行环境和直接物理交互，不能只复制数值。
4. 对我们的平地 RWM pipeline，应关闭复杂地形，但保留合理的物理和执行器随机化。
5. Dataset 阶段是注入完整物理/执行 DR 的关键位置。
6. Frozen RWM 阶段只能加入 action/observation 接口上的等效 DR。
7. Full 与 Half 必须严格定义，并记录每项参数的范围、采样频率和相关结构。
8. Motor strength 的裁剪顺序、物理参数只在环境创建时采样等实现细节，必须在迁移前确认。
9. DR 的目标是提升对误差的鲁棒性，不会替代正确的实机 PD gains、action scale、joint mapping 和 torque limits。

## 18. 代码来源

- Go2 配置：<https://github.com/wty-yy/go2_rl_gym/blob/vanilla_train/legged_gym/envs/go2/go2_config.py>
- Go2 环境：<https://github.com/wty-yy/go2_rl_gym/blob/vanilla_train/legged_gym/envs/go2/go2_env.py>
- 基础机器人实现：<https://github.com/wty-yy/go2_rl_gym/blob/vanilla_train/legged_gym/envs/base/legged_robot.py>
- 基础配置：<https://github.com/wty-yy/go2_rl_gym/blob/vanilla_train/legged_gym/envs/base/legged_robot_config.py>
