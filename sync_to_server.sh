#!/bin/bash
# 服务器文件同步脚本
# 用途：将 Mac 本地的部署文件同步到服务器
# 使用：bash sync_to_server.sh <server_user> <server_ip>

set -e

# 检查参数
if [ $# -ne 2 ]; then
    echo "用法: bash sync_to_server.sh <server_user> <server_ip>"
    echo "示例: bash sync_to_server.sh root 192.168.1.100"
    exit 1
fi

SERVER_USER=$1
SERVER_IP=$2
SERVER_PATH="/root/playground/GalaxeaVLA"

echo "🚀 同步部署文件到服务器"
echo "=" | head -c 70; echo
echo "目标服务器: $SERVER_USER@$SERVER_IP"
echo "目标路径: $SERVER_PATH"
echo

# 定义需要同步的文件
FILES=(
    "QUICKSTART.md"
    "SERVER_SETUP.md"
    "DEPLOYMENT_README.md"
    "scripts/setup_server.sh"
    "scripts/download_paligemma.sh"
    "scripts/server_inference.py"
)

# 创建临时目录
TEMP_DIR=$(mktemp -d)
echo "📦 准备文件..."

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        mkdir -p "$TEMP_DIR/$(dirname $file)"
        cp "$file" "$TEMP_DIR/$file"
        echo "  ✅ $file"
    else
        echo "  ⚠️  未找到: $file"
    fi
done

echo
echo "📤 上传到服务器..."

# 使用 rsync 同步
rsync -avz --progress "$TEMP_DIR/" "$SERVER_USER@$SERVER_IP:$SERVER_PATH/"

# 清理临时目录
rm -rf "$TEMP_DIR"

echo
echo "=" | head -c 70; echo
echo "✅ 同步完成！"
echo "=" | head -c 70; echo
echo
echo "📝 下一步（在服务器上执行）："
echo
echo "  ssh $SERVER_USER@$SERVER_IP"
echo "  cd $SERVER_PATH"
echo "  bash scripts/setup_server.sh"
echo "  bash scripts/download_paligemma.sh"
echo "  source .env"
echo "  python scripts/server_inference.py --ckpt_path checkpoints/G0Plus_3B_base/model_state_dict.pt"
echo
