# wandb配置文件 - 阻止上传模型文件
import os
import wandb

def configure_wandb():
    """配置wandb，阻止上传模型文件"""
    
    # 设置环境变量
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
    
    # 创建.wandbignore文件
    wandb_ignore_content = """
# 阻止wandb上传模型文件
*.pth
*.pt
*.ckpt
*.h5
*.hdf5
*.pkl
*.pickle
*.joblib
*.model
*.weights
*.bin
*.safetensors

# 阻止上传日志目录中的模型文件
logs/**/*.pth
logs/**/*.pt
logs/**/*.ckpt
logs/**/*.h5
logs/**/*.hdf5
logs/**/*.pkl
logs/**/*.pickle
logs/**/*.joblib
logs/**/*.model
logs/**/*.weights
logs/**/*.bin
logs/**/*.safetensors

# 阻止上传检查点目录
checkpoints/
checkpoint/
models/
model/
weights/
weight/

# 阻止上传大型数据文件
*.npy
*.npz
*.csv
*.json
*.yaml
*.yml
*.txt
*.log

# 阻止上传临时文件
*.tmp
*.temp
*.swp
*.swo
*~

# 阻止上传IDE文件
.vscode/
.idea/
*.pyc
__pycache__/
*.pyo
*.pyd
.Python
env/
venv/
.env
.venv
pip-log.txt
pip-delete-this-directory.txt

# 阻止上传git文件
.git/
.gitignore
.gitattributes

# 阻止上传其他大型文件
*.zip
*.tar
*.gz
*.rar
*.7z
"""
    
    # 写入.wandbignore文件
    with open(".wandbignore", "w") as f:
        f.write(wandb_ignore_content.strip())
    
    print("已创建.wandbignore文件，阻止wandb上传模型文件")

def init_wandb_without_models(project_name, run_name=None, **kwargs):
    """初始化wandb，但不上传模型文件"""
    
    # 配置wandb
    configure_wandb()
    
    # 初始化wandb
    wandb.init(
        project=project_name,
        name=run_name,
        mode="offline",  # 离线模式
        **kwargs
    )
    
    print(f"wandb已初始化，项目: {project_name}, 运行: {run_name}")
    print("注意: 模型文件不会被上传到wandb")

# 使用示例
if __name__ == "__main__":
    # 配置wandb
    configure_wandb()
    
    # 或者直接初始化
    # init_wandb_without_models("UAV_Navigation", "drone_experiment") 