import os

# =========================
# Camera config
# =========================

DEVICE = os.getenv("CAMERA_DEVICE", "/dev/video21")
WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
FPS = int(os.getenv("CAMERA_FPS", "25"))

OK_HOLD_FRAMES = int(os.getenv("OK_HOLD_FRAMES", "2"))

GAIN = int(os.getenv("CAMERA_GAIN", "0"))
BRIGHTNESS = int(os.getenv("CAMERA_BRIGHTNESS", "-15"))
CONTRAST = int(os.getenv("CAMERA_CONTRAST", "40"))
SHARPNESS = int(os.getenv("CAMERA_SHARPNESS", "4"))

WINDOW_NAME = os.getenv("WINDOW_NAME", "ELF2 HDMI Assistant")

# OK 手势检测频率
# 1 = 每帧检测，2 = 每两帧检测一次，数值越大越省 CPU
OK_DETECT_EVERY_N_FRAMES = int(os.getenv("OK_DETECT_EVERY_N_FRAMES", "2"))


# =========================
# LLM router config
# =========================
# 策略：
# 1. 有网且服务器可达：优先用服务器 DeepSeek
# 2. 没网/服务器不可达：切换到本地 RKLLAMA + RK3588 NPU
#
# REMOTE_LLM_TYPE:
#   openai -> 服务器是 OpenAI-compatible，例如 vLLM/SGLang/FastAPI: http://服务器IP:8000/v1
#   ollama -> 服务器是 Ollama-compatible: http://服务器IP:11434

REMOTE_LLM_ENABLED = os.getenv("REMOTE_LLM_ENABLED", "1") == "1"
REMOTE_LLM_TYPE = os.getenv("REMOTE_LLM_TYPE", "openai").lower()
REMOTE_LLM_BASE_URL = os.getenv("REMOTE_LLM_BASE_URL", "")
REMOTE_LLM_MODEL = os.getenv("REMOTE_LLM_MODEL", "deepseek-r1")
REMOTE_LLM_API_KEY = os.getenv("REMOTE_LLM_API_KEY", "")


# =========================
# Local RKLLAMA / NPU config
# =========================

LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "deepseek-r1-1.5b")

# 本地兜底必须走 NPU，不允许误用本地 Ollama 11434
LOCAL_NPU_REQUIRED = os.getenv("LOCAL_NPU_REQUIRED", "1") == "1"


# =========================
# Generation config
# =========================
# Ollama/RKLLAMA 兼容接口支持 stream 流式输出；REST API 默认就是流式，stream=false 才会关闭流式。:contentReference[oaicite:1]{index=1}
# num_predict 控制最大生成长度；这里给桌面助手用，所以不要太长。

LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "512"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.8"))

# 后端健康检查超时，秒
LLM_HEALTH_TIMEOUT = float(os.getenv("LLM_HEALTH_TIMEOUT", "2.0"))
