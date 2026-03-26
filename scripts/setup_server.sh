#!/bin/bash
# GalaxeaVLA 服务器一键部署脚本
# 用途：在服务器上快速创建目录结构并链接权重
# 使用：bash scripts/setup_server.sh

set -e  # 遇到错误立即退出

echo "🚀 GalaxeaVLA 服务器部署脚本"
echo "=" | head -c 70; echo

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "📁 项目根目录: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查函数
check_file() {
    if [ -f "$1" ]; then
        size=$(du -h "$1" | cut -f1)
        echo -e "${GREEN}✅ 找到: $1 ($size)${NC}"
        return 0
    else
        echo -e "${RED}❌ 未找到: $1${NC}"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅ 目录存在: $1${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  目录不存在: $1${NC}"
        return 1
    fi
}

echo
echo "=" | head -c 70; echo
echo "Step 1: 创建目录结构"
echo "=" | head -c 70; echo

# 创建基础目录
mkdir -p checkpoints/G0Plus_3B_base
mkdir -p checkpoints/G0Plus_PP_CKPT
mkdir -p data/google
mkdir -p data/datasets
mkdir -p outputs

echo -e "${GREEN}✅ 目录结构创建完成${NC}"
tree -L 2 -d . 2>/dev/null || ls -la

echo
echo "=" | head -c 70; echo
echo "Step 2: 链接已有权重"
echo "=" | head -c 70; echo

# 定义源权重路径（服务器上）
SRC_G0PLUS_BASE="/root/ckpts/G0Plus_3B_base/checkpoints/model_state_dict.pt"
SRC_G0PLUS_PP="/root/ckpts/G0Plus_PP_CKPT/model_state_dict.pt"

# 目标路径（项目内）
DST_G0PLUS_BASE="$PROJECT_ROOT/checkpoints/G0Plus_3B_base/model_state_dict.pt"
DST_G0PLUS_PP="$PROJECT_ROOT/checkpoints/G0Plus_PP_CKPT/model_state_dict.pt"

# 链接 G0Plus_3B_base
if [ -f "$SRC_G0PLUS_BASE" ]; then
    if [ -L "$DST_G0PLUS_BASE" ]; then
        echo -e "${YELLOW}⚠️  符号链接已存在，删除旧链接${NC}"
        rm "$DST_G0PLUS_BASE"
    fi
    ln -sf "$SRC_G0PLUS_BASE" "$DST_G0PLUS_BASE"
    check_file "$DST_G0PLUS_BASE" && echo "  -> $SRC_G0PLUS_BASE"
else
    echo -e "${RED}❌ 源文件不存在: $SRC_G0PLUS_BASE${NC}"
    echo "   请确认服务器上的权重路径"
fi

# 链接 G0Plus_PP_CKPT
if [ -f "$SRC_G0PLUS_PP" ]; then
    if [ -L "$DST_G0PLUS_PP" ]; then
        echo -e "${YELLOW}⚠️  符号链接已存在，删除旧链接${NC}"
        rm "$DST_G0PLUS_PP"
    fi
    ln -sf "$SRC_G0PLUS_PP" "$DST_G0PLUS_PP"
    check_file "$DST_G0PLUS_PP" && echo "  -> $SRC_G0PLUS_PP"
else
    echo -e "${RED}❌ 源文件不存在: $SRC_G0PLUS_PP${NC}"
    echo "   请确认服务器上的权重路径"
fi

echo
echo "=" | head -c 70; echo
echo "Step 3: 检查 PaliGemma 权重"
echo "=" | head -c 70; echo

PALIGEMMA_DIR="$PROJECT_ROOT/data/google/paligemma-3b-pt-224"

if check_dir "$PALIGEMMA_DIR"; then
    echo "检查核心文件..."
    check_file "$PALIGEMMA_DIR/config.json" || true
    check_file "$PALIGEMMA_DIR/model.safetensors" || true
    check_file "$PALIGEMMA_DIR/tokenizer.json" || true
else
    echo -e "${YELLOW}⚠️  PaliGemma 权重未下载${NC}"
    echo
    echo "请执行以下命令下载（约 5GB，需要 10-20 分钟）："
    echo
    echo -e "${GREEN}# 方案 1: 使用 huggingface-cli（推荐）${NC}"
    echo "pip install huggingface-hub"
    echo "huggingface-cli download google/paligemma-3b-pt-224 \\"
    echo "  --local-dir $PALIGEMMA_DIR \\"
    echo "  --local-dir-use-symlinks False"
    echo
    echo -e "${GREEN}# 方案 2: 使用国内镜像（如果网络不好）${NC}"
    echo "export HF_ENDPOINT=https://hf-mirror.com"
    echo "huggingface-cli download google/paligemma-3b-pt-224 \\"
    echo "  --local-dir $PALIGEMMA_DIR \\"
    echo "  --local-dir-use-symlinks False"
    echo
    echo -e "${GREEN}# 方案 3: 使用 git lfs${NC}"
    echo "cd data/google"
    echo "git lfs install"
    echo "git clone https://huggingface.co/google/paligemma-3b-pt-224"
fi

echo
echo "=" | head -c 70; echo
echo "Step 4: 环境检查"
echo "=" | head -c 70; echo

# 检查 Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}✅ Python: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python 未安装${NC}"
fi

# 检查 PyTorch 和 CUDA
python -c "
import sys
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ CUDA version: {torch.version.cuda}')
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️  CUDA not available')
except ImportError as e:
    print(f'❌ PyTorch import error: {e}')
    sys.exit(1)
" || echo -e "${RED}PyTorch 检查失败${NC}"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo -e "${YELLOW}⚠️  nvidia-smi 未找到${NC}"
fi

echo
echo "=" | head -c 70; echo
echo "Step 5: 设置环境变量"
echo "=" | head -c 70; echo

ENV_FILE="$PROJECT_ROOT/.env"
cat > "$ENV_FILE" <<EOF
# GalaxeaVLA 环境变量配置
# 由 setup_server.sh 自动生成

export PROJECT_ROOT=$PROJECT_ROOT
export GALAXEA_FM_OUTPUT_DIR=$PROJECT_ROOT/outputs
export HF_DATASETS_CACHE=$PROJECT_ROOT/data/datasets
export PYTHONPATH=$PROJECT_ROOT/src:\$PYTHONPATH

# 可选：如果在中国，取消注释以下行
# export HF_ENDPOINT=https://hf-mirror.com
EOF

echo -e "${GREEN}✅ 环境变量已保存到: $ENV_FILE${NC}"
echo
echo "使用前请加载环境变量："
echo -e "${GREEN}source $ENV_FILE${NC}"

echo
echo "=" | head -c 70; echo
echo "✅ 部署完成！"
echo "=" | head -c 70; echo
echo
echo "📝 下一步操作："
echo
echo "1. 如果 PaliGemma 未下载，请运行下载命令（见上面 Step 3）"
echo
echo "2. 加载环境变量："
echo -e "   ${GREEN}source .env${NC}"
echo
echo "3. 运行快速测试："
echo -e "   ${GREEN}python scripts/server_inference.py \\${NC}"
echo -e "   ${GREEN}     --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt${NC}"
echo
echo "4. 查看完整文档："
echo -e "   ${GREEN}cat SERVER_SETUP.md${NC}"
echo
echo "=" | head -c 70; echo
