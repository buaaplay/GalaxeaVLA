# GalaxeaVLA 快速启动指南

## 🚀 5 分钟快速上手（服务器端）

### 前置条件
- ✅ Ubuntu 20.04/22.04 + CUDA 12.8
- ✅ Python 3.10 环境已配置
- ✅ 已安装项目依赖（uv sync）
- ✅ GPU: RTX 4090 或 8GB+ VRAM

---

## 📦 一键部署

```bash
# 1. 进入项目目录
cd /root/playground/GalaxeaVLA

# 2. 运行一键部署脚本（创建目录、链接权重）
bash scripts/setup_server.sh

# 3. 加载环境变量
source .env

# 4. 下载 PaliGemma 权重（约 5GB，首次需要）
bash scripts/download_paligemma.sh
# 选择方案 1 或 2（推荐 2-国内镜像）

# 5. 运行推理测试
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt
```

**期望输出**：
```
🚀 GalaxeaVLA Server Inference Script
======================================================================
✅ Model loaded successfully!
✅ Inference successful!
📊 Results:
  Output shape: torch.Size([1, 10, 26])
  Action breakdown (26-dim):
    Left arm (6):      [...]
    Left gripper (1):  ...
    ...
✅ Inference test completed successfully!
```

---

## 📁 目录结构（自动创建）

```
/root/playground/GalaxeaVLA/
├── checkpoints/                              # 模型权重
│   ├── G0Plus_3B_base/
│   │   └── model_state_dict.pt              # 软链接 -> /root/ckpts/...
│   └── G0Plus_PP_CKPT/
│       └── model_state_dict.pt              # 软链接 -> /root/ckpts/...
│
├── data/
│   ├── google/
│   │   └── paligemma-3b-pt-224/             # PaliGemma 权重（需下载）
│   └── datasets/                             # LeRobot 数据集（可选）
│
├── outputs/                                  # 推理输出
│
├── scripts/
│   ├── setup_server.sh                      # 一键部署脚本 ⭐
│   ├── download_paligemma.sh                # PaliGemma 下载脚本 ⭐
│   ├── server_inference.py                  # 快速推理脚本 ⭐
│   ├── eval_open_loop.py                    # 离线评估（官方）
│   └── run/
│       ├── eval_open_loop.sh                # 评估启动脚本
│       └── finetune.sh                      # Fine-tuning 启动脚本
│
├── configs/                                  # 配置文件
│   ├── train.yaml                           # 训练主配置
│   ├── model/vla/g0plus.yaml                # 模型配置
│   └── task/real/*.yaml                     # 任务配置
│
├── .env                                      # 环境变量（自动生成）
├── SERVER_SETUP.md                          # 详细部署文档 ⭐
└── QUICKSTART.md                            # 本文档 ⭐
```

---

## 🎯 常用命令

### 基础推理测试
```bash
# 使用 G0Plus_3B_base（通用）
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt \
  --batch_size 1 \
  --device cuda

# 使用 Pick-and-Place 权重
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_PP_CKPT/model_state_dict.pt
```

### 离线评估（需要数据集）
```bash
# 设置环境变量
export PROJECT_ROOT=/root/playground/GalaxeaVLA
export GALAXEA_FM_OUTPUT_DIR=outputs
export HF_DATASETS_CACHE=data/datasets

# 运行评估
bash scripts/run/eval_open_loop.sh \
  configs/task/real/r1lite_g0plus_finetune_demo.yaml \
  checkpoints/G0Plus_3B_base/model_state_dict.pt \
  batch_size_val=8 \
  eval_episodes_num=10
```

### Fine-tuning（需要 A100/H100）
```bash
# 准备你的 LeRobot 格式数据集
export HF_DATASETS_CACHE=data/datasets
export GALAXEA_FM_OUTPUT_DIR=outputs
export SWANLAB_API_KEY=your_api_key

# 修改配置文件
vim configs/task/real/r1lite_g0plus_finetune_demo.yaml
# 修改 dataset_dirs 指向你的数据集

# 启动 Fine-tuning（8 GPU）
bash scripts/run/finetune.sh 8 real/r1lite_g0plus_finetune_demo
```

---

## 🔧 环境变量

自动生成的 `.env` 文件包含：

```bash
export PROJECT_ROOT=/root/playground/GalaxeaVLA
export GALAXEA_FM_OUTPUT_DIR=/root/playground/GalaxeaVLA/outputs
export HF_DATASETS_CACHE=/root/playground/GalaxeaVLA/data/datasets
export PYTHONPATH=/root/playground/GalaxeaVLA/src:$PYTHONPATH

# 可选：国内镜像加速
# export HF_ENDPOINT=https://hf-mirror.com
```

**使用前务必加载**：
```bash
source .env
```

---

## 📥 PaliGemma 权重下载方案

### 方案 1: huggingface-cli（推荐）
```bash
pip install huggingface-hub

huggingface-cli download google/paligemma-3b-pt-224 \
  --local-dir data/google/paligemma-3b-pt-224 \
  --local-dir-use-symlinks False \
  --resume-download
```

### 方案 2: 国内镜像（网络不好）
```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download google/paligemma-3b-pt-224 \
  --local-dir data/google/paligemma-3b-pt-224 \
  --local-dir-use-symlinks False
```

### 方案 3: 自动脚本
```bash
bash scripts/download_paligemma.sh
# 选择方案 2（国内镜像）
```

---

## 🐛 常见问题

### Q1: 找不到 CUDA
```bash
# 检查 CUDA
nvidia-smi

# 检查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Q2: 找不到模块
```bash
# 确保加载了环境变量
source .env

# 或手动设置
export PYTHONPATH=/root/playground/GalaxeaVLA/src:$PYTHONPATH
```

### Q3: PaliGemma 下载慢
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用迅雷等工具手动下载后放到 data/google/paligemma-3b-pt-224/
```

### Q4: 显存不足
```bash
# 减小 batch size
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt \
  --batch_size 1

# 使用 CPU（慢）
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt \
  --device cpu
```

### Q5: 权重加载失败
```bash
# 检查权重文件完整性
ls -lh checkpoints/G0Plus_3B_base/model_state_dict.pt

# 应该显示约 13GB
# 如果不是，重新链接
ln -sf /root/ckpts/G0Plus_3B_base/checkpoints/model_state_dict.pt \
       checkpoints/G0Plus_3B_base/model_state_dict.pt
```

---

## 📊 性能参考

| GPU | 显存 | Batch Size | 推理速度 | 备注 |
|-----|------|------------|----------|------|
| RTX 4090 | 24GB | 8 | ~30 FPS | 推荐 |
| RTX 3090 | 24GB | 4 | ~20 FPS | 可用 |
| A100 80GB | 80GB | 32 | ~100 FPS | Fine-tuning |

---

## 📝 下一步

1. ✅ 完成基础推理测试
2. 📊 准备你的 LeRobot 数据集
3. 🔧 运行离线评估
4. 🚀 Fine-tune 自定义任务
5. 🤖 部署到真实机器人（配合 EFMNode）

---

## 📚 相关文档

- **SERVER_SETUP.md** - 详细部署指南
- **README.md** - 项目概述
- **docs/pick_up_anything_user_guideline.md** - 真实机器人部署
- **configs/** - 配置文件说明

---

## 🆘 获取帮助

- GitHub Issues: https://github.com/OpenGalaxea/GalaxeaVLA/issues
- Discord: https://discord.gg/hB6BuUWZZA
- Email: wave.leaf27@gmail.com

---

## ✅ 验证清单

完成部署后，请验证：

- [ ] `nvidia-smi` 显示 CUDA 12.8
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 返回 True
- [ ] `ls -lh checkpoints/G0Plus_3B_base/model_state_dict.pt` 显示 13GB
- [ ] `ls data/google/paligemma-3b-pt-224/model.safetensors` 存在（约 5GB）
- [ ] `python scripts/server_inference.py --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt` 运行成功

全部完成 ✅ 即可开始使用！
