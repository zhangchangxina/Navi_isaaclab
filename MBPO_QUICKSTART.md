# MBPO 快速开始指南

## ✅ 修复完成

所有导入问题已修复！现在可以正常运行 MBPO 训练了。

## 🚀 立即开始训练

```bash
# 方式 1: 使用训练脚本 (推荐)
./run_mbpo_train.sh

# 方式 2: 直接运行
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl_mbpo/train.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --num_envs=4096 \
    --max_iterations=10000 \
    --headless
```

## 📋 当前配置

脚本 `run_mbpo_train.sh` 使用以下配置：
- **任务**: Isaac-Exploration-Rough-Drone-v0
- **环境数量**: 4096
- **GPU**: CUDA:1
- **最大迭代**: 10000
- **Steps per env**: 24
- **Learning epochs**: 5
- **Mini-batches**: 4
- **Batch size**: 256

## 🔧 修改配置

编辑 `run_mbpo_train.sh` 文件来修改训练参数：

```bash
# 修改任务
--task=Isaac-Exploration-Rough-Drone-v0

# 修改环境数量（更多环境 = 更快收集数据）
--num_envs=2048  # 或 4096, 8192

# 修改训练参数
--num_steps_per_env=24       # 每个环境每次迭代收集的步数
--num_learning_epochs=5      # 每次迭代的学习轮数
--num_mini_batches=4         # 每轮的mini-batch数量
--batch_size=256             # 批次大小

# 添加日志（可选）
--logger=wandb               # 使用 WandB
--log_project_name=UAV_MBPO  # 项目名称
```

## 📊 监控训练

### 使用 TensorBoard (默认)
```bash
tensorboard --logdir logs/rsl_rl_mbpo/
```

### 使用 WandB
在 `run_mbpo_train.sh` 中添加：
```bash
--logger=wandb \
--log_project_name=UAV_Navigation
```

## 🎮 测试训练好的模型

训练完成后，使用以下命令测试：

```bash
python scripts/reinforcement_learning/rsl_rl_mbpo/play.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --num_envs=32 \
    --checkpoint=logs/rsl_rl_mbpo/Isaac_Exploration_Rough_Drone_v0/TIMESTAMP/model_1000.pt
```

## 💡 重要说明

### 自动配置创建
脚本会自动检测任务是否有 MBPO 配置。如果没有，会创建默认配置：

```
[INFO] MBPO config not found for task, creating default config
[INFO] Created default MBPO configuration
```

这是正常的！默认配置已经可以开始训练了。

### GPU 设置
- 脚本使用 `CUDA_VISIBLE_DEVICES=1` (GPU 1)
- 可以在 `run_mbpo_train.sh` 中修改为其他 GPU

### 日志位置
训练日志和检查点保存在：
```
logs/rsl_rl_mbpo/
└── Isaac_Exploration_Rough_Drone_v0/
    └── TIMESTAMP/
        ├── params/
        ├── model_100.pt
        ├── model_200.pt
        └── ...
```

## 🐛 常见问题

### Q: 提示找不到 MBPO 配置？
A: 这是正常的！脚本会自动创建默认配置。

### Q: 训练很慢？
A: 尝试：
- 减少环境数量 (--num_envs=2048)
- 减少 mini-batches (--num_mini_batches=2)
- 使用更快的 GPU

### Q: 内存不足？
A: 尝试：
- 减少环境数量
- 减小 batch_size
- 减少 replay_size (需要修改配置)

## 📚 更多信息

- 详细文档: `scripts/reinforcement_learning/rsl_rl_mbpo/README.md`
- 完整设置指南: `MBPO_SETUP.md`
- 新旧对比: `scripts/reinforcement_learning/rsl_rl_mbpo/COMPARISON.md`
- 配置示例: `scripts/reinforcement_learning/rsl_rl_mbpo/example_config.py`

## ✨ 开始训练吧！

现在一切就绪，运行以下命令开始你的第一次 MBPO 训练：

```bash
./run_mbpo_train.sh
```

祝训练顺利！🎉

