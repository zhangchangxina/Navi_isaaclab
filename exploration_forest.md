## 安装环境

参考下面文档完成

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html

1. Isaac Sim安装
2. 环境变量设置
3. 下载IsaacLab（直接使用该代码库，不要再git clone下载一次）
4. 设置Isaac Sim的软链接
5. 设置conda环境
6. 安装依赖库
7. 确认安装完成

## 修改路径
修改source/isaaclab/isaaclab/utils/assets.py的36行，改为IsaacLab的存放路径

## Task名称
无人车：Isaac-Exploration-Rough-Turtlebot-v0

无人机：Isaac-Exploration-Rough-Drone-v0

## 动作空间
无人车：Vx, Wz

无人机：Vx, Vy, Vz

## 启动训练
### 无人车
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Exploration-Rough-Turtlebot-v0 --num_envs=16
### 无人机
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Exploration-Rough-Drone-v0 --num_envs=16


## 箭头显示
蓝色箭头：机器人当前速度
绿色箭头：目标点位置与朝向


## 任务定义相关文件
### source/isaaclab_tasks/isaaclab_tasks/manager_based/exploration/velocity/velocity_env_cfg.py
目标点设置方式：
    1.在障碍物区域边界随机发布目标点
    2.在障碍物区域内部随机发布目标点
    以上两种可选，可参照139~160行

机器人重置方式：可参照229~265行
    1.在障碍物区域边界随机重置机器人
    2.在障碍物区域内部随机重置机器人
    以上两种可选，可参照139~160行

轨迹显示：
    可自行选择显示与否
    可参照163~169行

材质设置
    可通过更换mdl文件进行材质更改
    地面材质设置：60行
    树木材质设置：82行
    参考资料：https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html


## 地形保持与加载
地形已设置好默认保持，通过在source/isaaclab/isaaclab/terrains/config/forest.py文件中设置seed值可实现对应地形加载
(地形保存在logs/terrains文件夹下，不同的seed值对应不同的地形，seed值可在具体terrain文件夹下的cfg.yaml中查看)


## 传感器设置
雷达：单线激光雷达，分辨率1deg

