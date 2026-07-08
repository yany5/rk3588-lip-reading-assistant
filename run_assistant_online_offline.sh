#!/usr/bin/env bash
set -e

cd ~/lip_assistant_interfaces

# =========================
# 0. Conda 环境
# =========================

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lipv4l


# =========================
# 1. 远程 DeepSeek 配置
# =========================
# 有网时优先使用官方 DeepSeek API。
# 如果远程不可达，再切换到本地 RKLLAMA + RK3588 NPU。

export REMOTE_LLM_ENABLED=1
export REMOTE_LLM_TYPE=openai
export REMOTE_LLM_BASE_URL=https://api.deepseek.com
export REMOTE_LLM_MODEL=deepseek-chat

# API Key 不直接写死在脚本里，而是从 .deepseek_api_key 读取
if [ -f ~/lip_assistant_interfaces/.deepseek_api_key ]; then
    export REMOTE_LLM_API_KEY="$(cat ~/lip_assistant_interfaces/.deepseek_api_key | tr -d '\n\r ')"
else
    export REMOTE_LLM_API_KEY=""
fi


# =========================
# 2. 本地 RKLLAMA / NPU 兜底配置
# =========================

export LOCAL_LLM_BASE_URL=http://127.0.0.1:8080
export LOCAL_LLM_MODEL=deepseek-r1-1.5b
export LOCAL_NPU_REQUIRED=1

export HF_HOME=~/RKLLAMA/hf_cache


# =========================
# 3. 生成参数
# =========================
# 桌面助手场景：回答短、稳定、不要发散

export LLM_NUM_PREDICT=48
export LLM_NUM_CTX=1024
export LLM_TEMPERATURE=0.0
export LLM_TOP_P=0.3
export LLM_HEALTH_TIMEOUT=2.0


# =========================
# 4. 先判断远程 DeepSeek 是否可用
# =========================

remote_ok=0

echo "[LLM] 检查远程 DeepSeek 是否可用..."

if [ "$REMOTE_LLM_ENABLED" = "1" ] && [ -n "$REMOTE_LLM_BASE_URL" ] && [ -n "$REMOTE_LLM_API_KEY" ]; then
    if curl --connect-timeout 3 -m 5 -fsS \
        -H "Authorization: Bearer $REMOTE_LLM_API_KEY" \
        "$REMOTE_LLM_BASE_URL/models" >/dev/null 2>&1; then
        remote_ok=1
    fi
fi

if [ "$remote_ok" = "1" ]; then
    echo "[LLM] 远程 DeepSeek 可用：本次优先联网调用，不启动本地 RKLLAMA。"
else
    echo "[LLM] 远程 DeepSeek 不可用，准备使用本地 RKLLAMA/NPU。"

    if ! curl --connect-timeout 2 -m 3 -fsS "$LOCAL_LLM_BASE_URL/api/tags" >/dev/null 2>&1; then
        echo "[RKLLAMA] 本地 NPU 服务没启动，准备后台启动..."

        mkdir -p ~/RKLLAMA/hf_cache ~/RKLLAMA/models

        nohup bash -lc '
            source ~/miniforge3/etc/profile.d/conda.sh
            conda activate lipv4l
            export HF_HOME=~/RKLLAMA/hf_cache
            rkllama_server --debug --models ~/RKLLAMA/models
        ' > ~/rkllama_server.log 2>&1 &

        echo "[RKLLAMA] 等待服务启动..."
        sleep 6

        if curl --connect-timeout 2 -m 3 -fsS "$LOCAL_LLM_BASE_URL/api/tags" >/dev/null 2>&1; then
            echo "[RKLLAMA] 本地 NPU 服务启动成功。"
        else
            echo "[ERROR] 本地 RKLLAMA 启动失败，请查看日志：~/rkllama_server.log"
            exit 1
        fi
    else
        echo "[RKLLAMA] 本地 NPU 服务已在运行。"
    fi
fi


# =========================
# 5. HDMI / OpenCV 显示环境
# =========================

export DISPLAY=:0

# 不再 unset LD_LIBRARY_PATH，避免影响 RKLLAMA/RKLLM runtime
unset QT_PLUGIN_PATH
unset QML2_IMPORT_PATH


# =========================
# 6. 启动主程序
# =========================

python3 main.py
