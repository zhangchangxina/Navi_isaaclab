# 统一配置覆盖逻辑示例

## 修改后的配置逻辑

现在所有参数都使用统一的智能覆盖逻辑：

```python
# 智能覆盖逻辑模式：
_param = getattr(args, 'param_name', default_value) if _flag_in_argv('--param_name') else getattr(task_cfg, 'param_name', default_value)
```

## 配置优先级（从高到低）

1. **命令行参数** (最高优先级)
2. **配置文件** (中等优先级)  
3. **硬编码默认值** (最低优先级)

## 使用示例

### 1. 通过命令行覆盖任何参数

```bash
# 覆盖SAC核心参数
./isaaclab.sh -p scripts/incremental_mbpo/train_mbpo.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --gamma=0.95 \
    --tau=0.01 \
    --alpha=0.1 \
    --lr=5e-4

# 覆盖MBPO参数
./isaaclab.sh -p scripts/incremental_mbpo/train_mbpo.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --num_networks=3 \
    --rollout_batch_size=512 \
    --replay_size=2000000

# 覆盖训练控制参数
./isaaclab.sh -p scripts/incremental_mbpo/train_mbpo.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --max_iterations=5000 \
    --updates_per_cycle=300 \
    --batch_size=512
```

### 2. 混合使用（部分命令行，部分配置文件）

```bash
# 只覆盖关键参数，其他使用配置文件默认值
./isaaclab.sh -p scripts/incremental_mbpo/train_mbpo.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --max_iterations=10000 \
    --num_envs=256 \
    --updates_per_cycle=200
    # gamma, tau, alpha等使用mbpo_cfg.py中的默认值
```

## 新增的命令行参数

### Core SAC参数
- `--gamma`: 折扣因子
- `--tau`: 软更新系数
- `--alpha`: 熵系数
- `--lr`: 通用学习率
- `--critic_lr`: Critic学习率
- `--policy_lr`: Policy学习率
- `--alpha_lr`: Alpha学习率
- `--dynamics_lr`: 动力学模型学习率
- `--automatic_entropy_tuning`: 自动熵调节
- `--target_update_interval`: 目标网络更新间隔

### MBPO参数
- `--num_networks`: 动力学模型数量
- `--num_elites`: 精英模型数量
- `--rollout_batch_size`: 模型rollout批次大小
- `--rollout_min_length`: 最小rollout长度
- `--rollout_max_length`: 最大rollout长度
- `--replay_size`: 回放缓冲区大小
- `--min_pool_size`: 最小池大小
- `--real_ratio`: 真实数据比例
- `--real_ratio_min`: 最小真实数据比例
- `--batch_size`: 训练批次大小
- `--updates_per_cycle`: 每周期更新次数

### 增量动力学参数
- `--dyn_w_dyn`: 动力学损失权重
- `--dyn_w_r`: 奖励损失权重
- `--dyn_w_d`: 终止损失权重
- `--dyn_done_threshold`: 终止阈值

### 其他配置开关
- `--use_cumulative_action`: 使用累积动作
- `--use_lqr_baseline`: 使用LQR基线
- `--lqr_model_path`: LQR模型路径
- `--formulation`: 公式类型

## 优势

1. **完全灵活**: 任何参数都可以通过命令行覆盖
2. **向后兼容**: 不提供命令行参数时使用配置文件默认值
3. **统一逻辑**: 所有参数使用相同的覆盖机制
4. **易于调试**: 可以快速调整任何参数进行实验
