# 服务器部署文件说明

本目录包含在服务器上快速部署 GalaxeaVLA 推理的所有必要文件和脚本。

## 📦 新增文件清单

### 1. 文档
- **QUICKSTART.md** - 5分钟快速上手指南（⭐ 推荐先读）
- **SERVER_SETUP.md** - 详细部署文档和故障排查

### 2. 脚本（在 scripts/ 目录）
- **setup_server.sh** - 一键部署脚本（创建目录、链接权重）
- **download_paligemma.sh** - PaliGemma 权重下载脚本
- **server_inference.py** - 快速推理测试脚本

### 3. 目录结构（自动创建）
```
checkpoints/  - 模型权重（软链接）
data/         - PaliGemma 权重和数据集
outputs/      - 推理输出
.env          - 环境变量（自动生成）
```

---

## 🚀 快速开始（3 步）

### 在服务器上执行：

```bash
# Step 1: 一键部署
cd /root/playground/GalaxeaVLA
bash scripts/setup_server.sh

# Step 2: 下载 PaliGemma 权重（约 5GB）
bash scripts/download_paligemma.sh
# 选择方案 2（国内镜像）

# Step 3: 测试推理
source .env
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt
```

**期望输出**：
```
✅ Model loaded successfully!
✅ Inference successful!
📊 Output shape: torch.Size([1, 10, 26])
```

---

## 📂 目录结构说明

```
/root/playground/GalaxeaVLA/
│
├── QUICKSTART.md                    ⭐ 快速上手（先读这个）
├── SERVER_SETUP.md                  📚 详细文档
├── DEPLOYMENT_README.md             📝 本文档
├── .env                             🔧 环境变量（自动生成）
│
├── checkpoints/                     📦 模型权重
│   ├── G0Plus_3B_base/
│   │   └── model_state_dict.pt     🔗 软链接 -> /root/ckpts/...
│   └── G0Plus_PP_CKPT/
│       └── model_state_dict.pt     🔗 软链接 -> /root/ckpts/...
│
├── data/
│   ├── google/
│   │   └── paligemma-3b-pt-224/    ⬇️  需要下载（5GB）
│   └── datasets/                    📊 LeRobot 数据（可选）
│
├── outputs/                         📁 推理输出
│
└── scripts/
    ├── setup_server.sh              🔧 一键部署
    ├── download_paligemma.sh        ⬇️  权重下载
    └── server_inference.py          🚀 快速推理
```

---

## 🎯 使用场景

### 场景 1: 快速验证模型（1 分钟）
```bash
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt
```

### 场景 2: 离线评估（需要数据集）
```bash
bash scripts/run/eval_open_loop.sh \
  configs/task/real/r1lite_g0plus_finetune_demo.yaml \
  checkpoints/G0Plus_3B_base/model_state_dict.pt
```

### 场景 3: Fine-tuning（需要 A100）
```bash
bash scripts/run/finetune.sh 8 real/r1lite_g0plus_finetune_demo
```

---

## 🔑 关键文件说明

### scripts/setup_server.sh
**用途**：自动化部署脚本

**功能**：
- ✅ 创建标准目录结构
- ✅ 软链接已有权重（避免重复占用空间）
- ✅ 检查环境（Python、CUDA、GPU）
- ✅ 生成 .env 环境变量文件

**运行**：
```bash
bash scripts/setup_server.sh
```

---

### scripts/download_paligemma.sh
**用途**：下载 PaliGemma-3b-pt-224 基础权重

**功能**：
- 方案 1: huggingface-cli（官方）
- 方案 2: huggingface-cli + 国内镜像（推荐）
- 方案 3: git lfs
- 方案 4: 手动 wget 核心文件

**运行**：
```bash
bash scripts/download_paligemma.sh
# 根据提示选择方案
```

---

### scripts/server_inference.py
**用途**：快速推理测试脚本

**功能**：
- ✅ 检查环境和文件
- ✅ 加载模型权重
- ✅ 运行推理测试
- ✅ 输出详细结果

**运行**：
```bash
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt \
  --batch_size 1 \
  --device cuda
```

**参数说明**：
- `--ckpt_path`: 权重路径（必需）
- `--paligemma_path`: PaliGemma 路径（默认 data/google/paligemma-3b-pt-224）
- `--batch_size`: 批次大小（默认 1）
- `--device`: 设备（cuda/cpu，默认 cuda）

---

## 🔄 从 Mac 到服务器的复刻流程

### 在 Mac 本地（已完成）：
```bash
cd /Users/play/Documents/Project/GalaxeaVLA
# 所有文件已创建完毕
```

### 复刻到服务器：
```bash
# 方案 1: 使用 rsync
rsync -avz --exclude='.git' \
  /Users/play/Documents/Project/GalaxeaVLA/ \
  user@server:/root/playground/GalaxeaVLA/

# 方案 2: 使用 git（如果有远程仓库）
# 在 Mac 上
cd /Users/play/Documents/Project/GalaxeaVLA
git add QUICKSTART.md SERVER_SETUP.md DEPLOYMENT_README.md
git add scripts/setup_server.sh scripts/download_paligemma.sh scripts/server_inference.py
git commit -m "Add server deployment scripts"
git push

# 在服务器上
cd /root/playground/GalaxeaVLA
git pull

# 方案 3: 手动复制文件
# 只复制关键文件：
# - QUICKSTART.md
# - SERVER_SETUP.md
# - DEPLOYMENT_README.md
# - scripts/setup_server.sh
# - scripts/download_paligemma.sh
# - scripts/server_inference.py
```

---

## ⚙️ 环境变量说明

自动生成的 `.env` 文件：

```bash
export PROJECT_ROOT=/root/playground/GalaxeaVLA
export GALAXEA_FM_OUTPUT_DIR=/root/playground/GalaxeaVLA/outputs
export HF_DATASETS_CACHE=/root/playground/GalaxeaVLA/data/datasets
export PYTHONPATH=/root/playground/GalaxeaVLA/src:$PYTHONPATH
```

**使用前必须加载**：
```bash
source .env
```

或添加到 `~/.bashrc`：
```bash
echo "source /root/playground/GalaxeaVLA/.env" >> ~/.bashrc
```

---

## 📋 部署 Checklist

- [ ] 复制文件到服务器
- [ ] 运行 `bash scripts/setup_server.sh`
- [ ] 运行 `bash scripts/download_paligemma.sh`
- [ ] 加载环境变量 `source .env`
- [ ] 测试推理 `python scripts/server_inference.py --ckpt_path ...`
- [ ] 验证输出 ✅

---

## 🐛 故障排查

### 问题 1: 权重文件找不到
```bash
# 检查链接是否正确
ls -lh checkpoints/G0Plus_3B_base/model_state_dict.pt

# 如果链接断开，重新运行
bash scripts/setup_server.sh
```

### 问题 2: PaliGemma 下载失败
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_paligemma.sh
# 选择方案 2
```

### 问题 3: 推理时显存不足
```bash
# 减小 batch size
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt \
  --batch_size 1
```

更多问题见 **SERVER_SETUP.md** 的故障排查章节。

---

## 📚 推荐阅读顺序

1. **QUICKSTART.md** - 快速上手（5 分钟）
2. **本文档** - 了解文件结构
3. **SERVER_SETUP.md** - 详细配置和故障排查
4. **README.md** - 项目总体介绍

---

## 🎉 完成部署后

恭喜！你已经完成了 GalaxeaVLA 的服务器部署。

**验证成功的标志**：
```bash
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt

# 输出：
✅ Model loaded successfully!
✅ Inference successful!
📊 Output shape: torch.Size([1, 10, 26])
```

**下一步**：
- 📊 准备你的 LeRobot 数据集
- 🔧 运行离线评估
- 🚀 Fine-tune 自定义任务
- 🤖 部署到真实机器人

---

## 🆘 需要帮助？

- 📖 查看 **QUICKSTART.md** 和 **SERVER_SETUP.md**
- 🐛 GitHub Issues: https://github.com/OpenGalaxea/GalaxeaVLA/issues
- 💬 Discord: https://discord.gg/hB6BuUWZZA

---

**创建日期**: 2026-01-10
**适用版本**: GalaxeaVLA G0Plus_3B_base / G0Plus_PP_CKPT
**测试环境**: Ubuntu 20.04 + CUDA 12.8 + RTX 4090
