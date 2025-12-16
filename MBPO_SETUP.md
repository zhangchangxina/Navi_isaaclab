# RSL-RL Style MBPO Implementation for Isaac Lab

本文档说明了如何使用新实现的 RSL-RL 风格的 MBPO (Model-Based Policy Optimization) 训练框架。

## 概述

我们已经将原始的 `train_mbpo.py` 重构为类似 RSL-RL 的模块化结构，使其更易于集成和使用。

## 文件结构

### 训练脚本 (scripts/reinforcement_learning/rsl_rl_mbpo/)

```
scripts/reinforcement_learning/rsl_rl_mbpo/
├── __init__.py              # 包初始化
├── cli_args.py              # 命令行参数处理
├── train.py                 # 主训练脚本
├── play.py                  # 测试/播放脚本
├── example_config.py        # 配置示例
└── README.md                # 详细文档
```

### 核心库 (source/isaaclab_rl/isaaclab_rl/rsl_rl_mbpo/)

```
source/isaaclab_rl/isaaclab_rl/rsl_rl_mbpo/
├── __init__.py              # 模块导出
├── mbpo_cfg.py              # 配置类定义
├── vecenv_wrapper.py        # 环境包装器
└── runner.py                # MBPO训练器
```

## 主要特性

1. **RSL-RL 风格接口**: 与 RSL-RL 的 PPO 训练保持一致的API和工作流
2. **模块化设计**: 清晰的配置、包装器和训练器分离
3. **灵活配置**: 支持命令行参数覆盖配置文件
4. **多种日志**: 支持 TensorBoard、WandB 和 Neptune
5. **RSL-RL 更新模式**: 支持类似 PPO 的多 epoch 和 mini-batch 更新
6. **模型集成**: 支持动力学模型集成学习
7. **稳定性奖励**: 可选的基于 LQR 的稳定性奖励塑形

## 快速开始

### 1. 基本训练

```bash
# 使用默认参数训练
python scripts/reinforcement_learning/rsl_rl_mbpo/train.py \
    --task Isaac-Velocity-Flat-Drone-v0 \
    --num_envs 4096 \
    --max_iterations 10000
```

### 2. 使用启动脚本

```bash
# 修改 run_mbpo_train.sh 中的参数
./run_mbpo_train.sh
```

### 3. 自定义配置训练

```bash
python scripts/reinforcement_learning/rsl_rl_mbpo/train.py \
    --task Isaac-Velocity-Flat-Drone-v0 \
    --num_envs 4096 \
    --max_iterations 10000 \
    --num_steps_per_env 24 \
    --num_learning_epochs 5 \
    --num_mini_batches 4 \
    --batch_size 256 \
    --gamma 0.99 \
    --tau 0.005 \
    --alpha 0.2 \
    --logger wandb \
    --log_project_name UAV_Navigation \
    --experiment_name my_mbpo_experiment \
    --run_name test_run_1
```

### 4. 测试训练好的模型

```bash
python scripts/reinforcement_learning/rsl_rl_mbpo/play.py \
    --task Isaac-Velocity-Flat-Drone-v0 \
    --num_envs 32 \
    --checkpoint logs/rsl_rl_mbpo/experiment_name/timestamp/model_1000.pt
```

## 关键参数说明

### 环境参数
- `--task`: 任务名称 (例如: Isaac-Velocity-Flat-Drone-v0)
- `--num_envs`: 并行环境数量
- `--seed`: 随机种子

### 训练参数
- `--max_iterations`: 最大训练迭代次数
- `--num_steps_per_env`: 每个环境每次迭代收集的步数
- `--batch_size`: 训练批次大小

### RSL-RL 风格参数
- `--num_learning_epochs`: 每次迭代的学习轮数 (类似 PPO)
- `--num_mini_batches`: 每轮的 mini-batch 数量

### SAC 参数
- `--gamma`: 折扣因子 (default: 0.99)
- `--tau`: 目标网络软更新系数 (default: 0.005)
- `--alpha`: 熵系数 (default: 0.2)

### 日志参数
- `--logger`: 日志类型 (tensorboard/wandb/neptune)
- `--log_project_name`: 项目名称 (用于 wandb/neptune)
- `--experiment_name`: 实验名称
- `--run_name`: 运行名称后缀

### 可选特性
- `--use_stability_reward`: 启用稳定性奖励塑形
- `--use_cumulative_action`: 启用累积动作模式

## 与原始 train_mbpo.py 的区别

### 优点
1. **更好的模块化**: 代码组织更清晰，易于维护
2. **一致的接口**: 与 RSL-RL 的使用方式保持一致
3. **配置管理**: 使用 @configclass 而不是字典
4. **类型安全**: 更好的类型提示和检查
5. **易于扩展**: 可以轻松添加新功能或算法变体

### 保留的功能
- SAC 算法核心
- 增量动力学模型
- 模型集成训练
- 稳定性奖励塑形
- 累积动作支持
- RSL-RL 风格的更新模式

## 配置任务的 MBPO

要为你的任务配置 MBPO，可以参考 `example_config.py`:

```python
from isaaclab_rl.rsl_rl_mbpo import (
    RslRlMBPORunnerCfg,
    RslRlMBPOAlgorithmCfg,
    RslRlMBPOSACPolicyCfg,
    RslRlMBPOSACCriticCfg,
    RslRlMBPODynamicsModelCfg,
)

# 在你的任务配置中
@configclass
class MyTaskRslRlMBPOCfg(RslRlMBPORunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    # ... 其他配置
```

然后在任务的 `__init__.py` 中注册:

```python
# 注册 MBPO 配置入口点
MyTask.rsl_rl_mbpo_cfg_entry_point = MyTaskRslRlMBPOCfg
```

## 日志和检查点

### 日志位置
```
logs/rsl_rl_mbpo/
└── <experiment_name>/
    └── <timestamp>_<run_name>/
        ├── params/
        │   ├── env.yaml
        │   ├── agent.yaml
        │   ├── env.pkl
        │   └── agent.pkl
        ├── model_100.pt
        ├── model_200.pt
        └── ...
```

### 恢复训练

```bash
python scripts/reinforcement_learning/rsl_rl_mbpo/train.py \
    --task Isaac-Velocity-Flat-Drone-v0 \
    --resume \
    --load_run <run_directory> \
    --checkpoint model_*.pt
```

## 性能调优建议

1. **环境数量**: 从小开始 (64-256)，逐步增加到 2048-4096
2. **学习率**: 
   - Critic: 3e-4
   - Actor: 3e-4
   - Dynamics: 1e-3
3. **批次大小**: 256-512 通常效果较好
4. **熵系数**: 根据任务调整，探索多则增大，利用多则减小
5. **动力学模型**: 简单任务用 1 个模型，复杂任务用 3-5 个模型集成

## 故障排除

### 问题: 训练不稳定
- 减小学习率
- 增加批次大小
- 检查 Q 值是否爆炸

### 问题: 收敛慢
- 增加环境数量
- 调整熵系数
- 检查奖励尺度

### 问题: 内存不足
- 减少环境数量
- 减小 replay buffer 大小
- 减少动力学模型数量

## 下一步

1. 查看 `scripts/reinforcement_learning/rsl_rl_mbpo/README.md` 获取更详细的文档
2. 查看 `example_config.py` 了解如何创建配置
3. 尝试训练你的第一个 MBPO agent!

## 参考资源

- [MBPO 论文](https://arxiv.org/abs/1906.08253)
- [SAC 论文](https://arxiv.org/abs/1801.01290)
- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL 库](https://github.com/leggedrobotics/rsl_rl)

## 许可证

BSD-3-Clause (与 Isaac Lab 相同)

