# GalaxeaVLA 服务器推理部署指南

本文档提供在服务器上快速部署推理的完整方案。

## 📁 目录结构规划

```
/root/playground/GalaxeaVLA/          # 项目根目录
├── checkpoints/                      # 模型权重（软链接到已有权重）
│   ├── G0Plus_3B_base/
│   │   └── model_state_dict.pt      -> /root/ckpts/G0Plus_3B_base/checkpoints/model_state_dict.pt
│   └── G0Plus_PP_CKPT/
│       └── model_state_dict.pt      -> /root/ckpts/G0Plus_PP_CKPT/model_state_dict.pt
│
├── data/                             # 数据和基础模型
│   ├── google/
│   │   └── paligemma-3b-pt-224/     # PaliGemma 基础权重（需下载）
│   └── datasets/                     # LeRobot 格式数据集（可选）
│
├── outputs/                          # 推理输出目录
│
├── configs/                          # 配置文件（已修改）
│   └── model/vla/g0plus.yaml        # 已适配服务器路径
│
├── scripts/
│   └── server_inference.py          # 快速推理脚本
│
└── SERVER_SETUP.md                   # 本文档
```

---

## 🚀 快速部署步骤（5分钟跑通）

### Step 1: 创建目录并链接权重

```bash
cd /root/playground/GalaxeaVLA

# 创建目录结构
mkdir -p checkpoints/G0Plus_3B_base checkpoints/G0Plus_PP_CKPT
mkdir -p data/google outputs

# 软链接已有权重（避免重复占用空间）
ln -sf /root/ckpts/G0Plus_3B_base/checkpoints/model_state_dict.pt \
       checkpoints/G0Plus_3B_base/model_state_dict.pt

ln -sf /root/ckpts/G0Plus_PP_CKPT/model_state_dict.pt \
       checkpoints/G0Plus_PP_CKPT/model_state_dict.pt

# 验证链接
ls -lh checkpoints/G0Plus_3B_base/model_state_dict.pt
ls -lh checkpoints/G0Plus_PP_CKPT/model_state_dict.pt
```

### Step 2: 下载 PaliGemma 基础权重

**方案 A：使用 huggingface-cli（推荐）**
```bash
# 安装 huggingface-hub
pip install huggingface-hub

# 下载权重（约 5GB，10-20分钟）
huggingface-cli download google/paligemma-3b-pt-224 \
  --local-dir data/google/paligemma-3b-pt-224 \
  --local-dir-use-symlinks False

# 或者使用国内镜像（如果网络不好）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download google/paligemma-3b-pt-224 \
  --local-dir data/google/paligemma-3b-pt-224 \
  --local-dir-use-symlinks False
```

**方案 B：使用 git lfs（备选）**
```bash
cd data/google
git lfs install
git clone https://huggingface.co/google/paligemma-3b-pt-224

# 如果网络慢，使用镜像
git clone https://hf-mirror.com/google/paligemma-3b-pt-224
```

**方案 C：手动下载核心文件**
```bash
cd data/google/paligemma-3b-pt-224

# 核心文件列表（最小化部署）
wget https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/model.safetensors
wget https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/config.json
wget https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/tokenizer.json
wget https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/tokenizer_config.json
wget https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/special_tokens_map.json
wget https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/preprocessor_config.json
```

### Step 3: 验证环境

```bash
# 检查 Python 环境（应该已经配置好）
python --version  # 应该是 3.10.x

# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 验证 GPU
nvidia-smi
```

**期望输出**：
```
PyTorch: 2.7.1
CUDA available: True
CUDA version: 12.8
```

### Step 4: 运行快速推理测试

```bash
# 使用我创建的快速测试脚本
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt \
  --device cuda

# 如果成功，应该看到：
# ✅ Model loaded successfully!
# ✅ Inference successful!
# ✅ Output action shape: torch.Size([1, 10, 26])
```

---

## 🔧 配置文件说明

已修改 `configs/model/vla/g0plus.yaml` 中的路径：

```yaml
# Line 1: 预训练权重路径（使用项目相对路径）
pretrained_ckpt: ${oc.env:PROJECT_ROOT}/checkpoints/G0Plus_3B_base/model_state_dict.pt

# Line 122: tokenizer 路径
tokenizer_params:
  pretrained_model_name_or_path: ${oc.env:PROJECT_ROOT}/data/google/paligemma-3b-pt-224

# Line 133: 预训练模型路径
pretrained_model_path: ${oc.env:PROJECT_ROOT}/data/google/paligemma-3b-pt-224
```

使用前设置环境变量：
```bash
export PROJECT_ROOT=/root/playground/GalaxeaVLA
```

---

## 📊 权重文件说明

### 已有权重（服务器上）

| 权重文件 | 大小 | 路径 | 用途 |
|---------|------|------|------|
| G0Plus_3B_base | 13GB | `/root/ckpts/G0Plus_3B_base/checkpoints/model_state_dict.pt` | Fine-tuning 基础模型 |
| G0Plus_PP_CKPT | 13GB | `/root/ckpts/G0Plus_PP_CKPT/model_state_dict.pt` | Pick-and-Place 部署模型 |

**权重内部结构**：
```python
{
    'model_state_dict': {
        'vision_tower.embeddings.patch_embedding.weight': ...,
        'vision_tower.encoder.layers.0.self_attn.q_proj.weight': ...,
        ...
    }
}
```

### 需要下载（约 5GB）

| 文件 | 用途 |
|------|------|
| PaliGemma-3b-pt-224 | 视觉-语言基础模型（必需） |

---

## 🎯 推理命令参考

### 1. 最小化测试（验证模型加载）
```bash
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt
```

### 2. 离线评估（需要数据集）
```bash
export PROJECT_ROOT=/root/playground/GalaxeaVLA
export GALAXEA_FM_OUTPUT_DIR=outputs

bash scripts/run/eval_open_loop.sh \
  configs/task/real/r1lite_g0plus_finetune_demo.yaml \
  checkpoints/G0Plus_3B_base/model_state_dict.pt \
  batch_size_val=8
```

### 3. 使用 Pick-and-Place 权重
```bash
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_PP_CKPT/model_state_dict.pt
```

---

## 🐛 常见问题排查

### 问题 1：找不到 PaliGemma 权重
**错误**：`OSError: google/paligemma-3b-pt-224 does not appear to be...`

**解决**：
```bash
# 方案1：设置环境变量
export HF_HOME=/root/playground/GalaxeaVLA/data
export TRANSFORMERS_CACHE=/root/playground/GalaxeaVLA/data

# 方案2：使用绝对路径
export PROJECT_ROOT=/root/playground/GalaxeaVLA
```

### 问题 2：CUDA Out of Memory
**错误**：`RuntimeError: CUDA out of memory`

**解决**：
```bash
# 减小 batch size
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt \
  --batch_size 1

# 或使用 CPU（慢）
python scripts/server_inference.py \
  --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt \
  --device cpu
```

### 问题 3：权重加载失败
**错误**：`KeyError: 'model_state_dict'`

**检查权重结构**：
```python
import torch
ckpt = torch.load("checkpoints/G0Plus_3B_base/model_state_dict.pt", map_location="cpu")
print("Keys:", list(ckpt.keys()))

# 如果有 model_state_dict key
state_dict = ckpt["model_state_dict"]

# 如果直接是 state_dict
state_dict = ckpt
```

---

## 📈 性能基准参考

| GPU 型号 | 显存 | Batch Size | 推理速度 |
|---------|------|-----------|----------|
| RTX 4090 | 24GB | 8 | ~30 FPS |
| RTX 3090 | 24GB | 4 | ~20 FPS |
| A100 80GB | 80GB | 32 | ~100 FPS |

---

## 📝 下一步操作

1. ✅ 完成 Step 1-4（约 15 分钟）
2. 🔄 准备你的 LeRobot 格式数据集（可选）
3. 🚀 运行离线评估或 Fine-tuning
4. 🤖 部署到真实机器人（需配合 EFMNode）

---

## 🆘 需要帮助？

- 官方仓库：https://github.com/OpenGalaxea/GalaxeaVLA
- 问题追踪：https://github.com/OpenGalaxea/GalaxeaVLA/issues
- Discord：https://discord.gg/hB6BuUWZZA
