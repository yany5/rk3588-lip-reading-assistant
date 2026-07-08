#!/usr/bin/env bash

PROJECT_DIR="/home/elf/lip_assistant_interfaces"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/autostart.log"

mkdir -p "$LOG_DIR"

exec >> "$LOG_FILE" 2>&1

echo
echo "========================================"
echo "[AUTOSTART] 启动时间：$(date)"
echo "[AUTOSTART] 用户：$(whoami)"
echo "[AUTOSTART] DISPLAY=${DISPLAY:-未设置}"
echo "[AUTOSTART] XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-未设置}"
echo "========================================"

# 等待桌面、HDMI 和摄像头设备初始化
sleep 5

source /home/elf/miniforge3/etc/profile.d/conda.sh
conda activate lipv4l

cd "$PROJECT_DIR" || exit 1

echo "[AUTOSTART] 当前目录：$(pwd)"
echo "[AUTOSTART] Python：$(which python3)"
echo "[AUTOSTART] 开始运行 run_assistant_online_offline.sh"

exec ./run_assistant_online_offline.sh
