#!/bin/bash
# PaliGemma 权重快速下载脚本
# 用途：自动下载 PaliGemma-3b-pt-224 权重到项目目录
# 使用：bash scripts/download_paligemma.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$PROJECT_ROOT/data/google/paligemma-3b-pt-224"

echo "🚀 PaliGemma 权重下载脚本"
echo "=" | head -c 70; echo
echo "目标目录: $TARGET_DIR"
echo

# 检查目录是否已存在且有内容
if [ -d "$TARGET_DIR" ] && [ "$(ls -A $TARGET_DIR)" ]; then
    echo "⚠️  目标目录已存在且非空"
    echo "已有文件:"
    ls -lh "$TARGET_DIR"
    echo
    read -p "是否删除并重新下载？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TARGET_DIR"
        echo "✅ 已删除旧文件"
    else
        echo "❌ 取消下载"
        exit 0
    fi
fi

# 创建目标目录
mkdir -p "$TARGET_DIR"

echo
echo "请选择下载方式:"
echo "1) huggingface-cli（推荐，可断点续传）"
echo "2) huggingface-cli + 国内镜像（网络不好时使用）"
echo "3) git lfs（传统方式）"
echo "4) 手动下载核心文件（最小化安装，约 5GB）"
echo

read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo
        echo "使用 huggingface-cli 下载..."
        echo

        # 检查是否安装 huggingface-hub
        if ! python -c "import huggingface_hub" 2>/dev/null; then
            echo "安装 huggingface-hub..."
            pip install -q huggingface-hub
        fi

        huggingface-cli download google/paligemma-3b-pt-224 \
            --local-dir "$TARGET_DIR" \
            --local-dir-use-symlinks False \
            --resume-download

        ;;

    2)
        echo
        echo "使用 huggingface-cli + 国内镜像下载..."
        echo

        # 检查是否安装 huggingface-hub
        if ! python -c "import huggingface_hub" 2>/dev/null; then
            echo "安装 huggingface-hub..."
            pip install -q huggingface-hub
        fi

        export HF_ENDPOINT=https://hf-mirror.com

        huggingface-cli download google/paligemma-3b-pt-224 \
            --local-dir "$TARGET_DIR" \
            --local-dir-use-symlinks False \
            --resume-download

        ;;

    3)
        echo
        echo "使用 git lfs 下载..."
        echo

        # 检查 git lfs
        if ! git lfs version &>/dev/null; then
            echo "❌ git lfs 未安装"
            echo "请先安装: sudo apt install git-lfs"
            exit 1
        fi

        cd "$(dirname "$TARGET_DIR")"
        git lfs install
        git clone https://huggingface.co/google/paligemma-3b-pt-224

        ;;

    4)
        echo
        echo "手动下载核心文件（最小化安装）..."
        echo "这将下载以下文件:"
        echo "  - model.safetensors (~5GB)"
        echo "  - config.json"
        echo "  - tokenizer files"
        echo

        cd "$TARGET_DIR"

        # 定义需要下载的文件
        FILES=(
            "model.safetensors"
            "config.json"
            "tokenizer.json"
            "tokenizer_config.json"
            "special_tokens_map.json"
            "preprocessor_config.json"
        )

        BASE_URL="https://huggingface.co/google/paligemma-3b-pt-224/resolve/main"

        for file in "${FILES[@]}"; do
            echo "下载: $file"
            wget -c "$BASE_URL/$file" -O "$file" || {
                echo "❌ 下载失败: $file"
                echo "尝试使用镜像..."
                wget -c "https://hf-mirror.com/google/paligemma-3b-pt-224/resolve/main/$file" -O "$file"
            }
        done

        ;;

    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo
echo "=" | head -c 70; echo
echo "✅ 下载完成！"
echo "=" | head -c 70; echo
echo

# 验证文件
echo "验证下载的文件:"
ls -lh "$TARGET_DIR"

# 检查核心文件
echo
echo "检查核心文件:"
for file in "config.json" "tokenizer.json"; do
    if [ -f "$TARGET_DIR/$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (缺失)"
    fi
done

if [ -f "$TARGET_DIR/model.safetensors" ]; then
    size=$(du -h "$TARGET_DIR/model.safetensors" | cut -f1)
    echo "✅ model.safetensors ($size)"
else
    echo "❌ model.safetensors (缺失)"
fi

echo
echo "📝 下一步: 运行推理测试"
echo "  source .env"
echo "  python scripts/server_inference.py \\"
echo "    --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt"
