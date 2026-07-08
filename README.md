# RK3588 唇语识别与 DeepSeek 交互助手

基于 ELFBoard/RK3588 开发板的离线/在线智能交互系统。系统通过 USB 摄像头采集视频，使用手势控制录制流程，调用板端唇语识别模型输出中文词语，并可进一步调用在线 DeepSeek 或本地 RKLLAMA/NPU 模型生成简短回复。系统同时支持 AHT20 温湿度传感器和 HDMI 全屏显示。

## 主要功能

- 摄像头常驻采集与最新帧共享
- OK、数字 1、数字 2 手势识别与防抖
- 唇语视频录制和 LRW-1000 模型推理
- 在线 DeepSeek 与本地 RKLLAMA/NPU 自动切换
- AHT20 温湿度读取与环境信息注入
- HDMI/OpenCV 全屏图形界面
- 开机启动脚本、NPU 诊断脚本和嘴部裁剪调试工具

## 交互流程

1. 初始界面比出 **OK**，进入主菜单。
2. 主菜单中：
   - 比出 **1**：进入唇语识别模式；
   - 比出 **2**：进入“唇语识别 + DeepSeek”模式；
   - 比出 **OK**：继续停留在主菜单。
3. 调整镜头后比出 **OK**。
4. 再次比出当前模式对应的 **1** 或 **2** 开始录制。
5. 说完后比出 **OK** 结束录制并开始识别。
6. 结果页中：
   - 比出 **2**：重复当前模式；
   - 比出 **OK**：返回主菜单。

键盘调试按键：`o` 模拟 OK，`1`/`2` 模拟对应手势，`q` 退出程序。

## 推荐仓库结构

为了不改动当前 Python 导入方式和启动脚本，运行所需文件继续放在仓库根目录；调试、旧版本和文档单独分类。

```text
rk3588-lip-reading-assistant/
├── main.py
├── display_service.py              # 必需，当前上传文件中缺少
├── camera_service.py
├── direct_hand_gesture.py
├── lip_model_bridge.py
├── llm_service.py
├── env_llm_context.py
├── env_sensor.py
├── config.py
├── ui_text_wrap.py
├── run_assistant_online_offline.sh
├── start_lip_gui.sh
│
├── tools/
│   ├── check_elf2_npu.sh
│   ├── fix_main_broken_strings.py
│   ├── mouth_crop_preview.py
│   └── mouth_crop_recorder.py
│
├── legacy/
│   ├── gesture_service.py
│   ├── ok_gesture_service.py
│   └── lip_reading_service.py
│
├── models/
│   ├── README.md
│   └── shape_predictor_68_face_landmarks.dat
│
├── docs/
│   └── 模块资料.pdf
│
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

> `gesture_service.py`、`ok_gesture_service.py` 和 `lip_reading_service.py` 是旧版或模拟服务，当前 `main.py` 不直接调用，因此放入 `legacy/`，避免与当前手势逻辑混淆。

## 硬件环境

- ELFBoard/RK3588 开发板
- Ubuntu/Linux 图形桌面环境
- USB 摄像头
- HDMI 显示器
- 可选：AHT20/AHT10 温湿度传感器
- 可选：RK3588 NPU 上运行的本地 RKLLAMA 服务

当前默认参数：

| 项目 | 默认值 |
|---|---|
| 摄像头 | `/dev/video21` |
| 分辨率 | `1280 × 720` |
| 帧率 | `25 FPS` |
| AHT20 I2C 总线 | `/dev/i2c-4` |
| AHT20 地址 | `0x38` |
| 本地 LLM 地址 | `http://127.0.0.1:8080` |
| 本地模型名 | `deepseek-r1-1.5b` |

摄像头设备可在启动前覆盖，例如：

```bash
export CAMERA_DEVICE=/dev/video22
```

## 软件环境

项目当前在以下环境中使用：

- Python 3.10
- Conda 环境：`lipv4l`
- OpenCV
- MediaPipe
- NumPy
- Requests
- Pygame/Pillow
- 可选：dlib
- 板端唇语推理所需 RKNN/ONNX 运行环境
- 本地大模型所需 RKLLAMA/RKLLM 环境

基础 Python 依赖可通过以下命令安装：

```bash
pip install -r requirements.txt
```

RKNN、RKLLAMA、MediaPipe ARM 包和 dlib 在 RK3588 上可能需要使用开发板适配版本，不建议只依赖普通 PyPI 自动安装。

## 唇语模型文件

`lip_model_bridge.py` 当前默认从下面的目录加载模型：

```text
/home/elf/桌面/lip-model/
```

该目录至少需要包含：

```text
lip-model/
├── recognize_latest_video_lrw1000.py
├── lrw1000_frontend_window.rknn
├── lrw1000_backend_gru.onnx
├── label_lrw1000_with_id.txt
├── runtime_videos/
├── runtime_infer_one/
├── lrw1000_pkl_output/
└── results/
```

这些模型文件通常体积较大，并且可能受到原项目许可证限制。建议不要直接提交到普通 Git 历史中，可选择：

1. 使用 Git LFS；
2. 放入 GitHub Release；
3. 在 README 中提供经过许可的下载说明；
4. 仅提交模型目录结构和校验值。

`shape_predictor_68_face_landmarks.dat` 约 95 MB，也建议使用 Git LFS，并在上传前确认其再分发许可证。

## 安装与运行

### 1. 克隆仓库

```bash
git clone <你的仓库地址>
cd rk3588-lip-reading-assistant
```

为了兼容当前脚本中的绝对路径，建议部署到：

```text
/home/elf/lip_assistant_interfaces
```

也可以修改 `run_assistant_online_offline.sh`、`start_lip_gui.sh` 和嘴部裁剪工具中的项目路径。

### 2. 激活 Conda 环境

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lipv4l
```

### 3. 添加缺少的显示模块

当前 `main.py` 会导入：

```python
from display_service import DisplayService
```

因此仓库根目录必须包含 `display_service.py`。没有该文件时，程序会在启动阶段直接报 `ModuleNotFoundError`。

### 4. 配置 DeepSeek API Key

不要把 API Key 写入代码，也不要提交到 GitHub。

```bash
printf '%s' '你的 DeepSeek API Key' > .deepseek_api_key
chmod 600 .deepseek_api_key
```

`.deepseek_api_key` 已在推荐的 `.gitignore` 中排除。

没有在线 API Key 时，启动脚本会尝试使用本地 `RKLLAMA + RK3588 NPU` 服务。

### 5. 添加执行权限

```bash
chmod +x run_assistant_online_offline.sh
chmod +x start_lip_gui.sh
chmod +x tools/check_elf2_npu.sh
```

### 6. 检查摄像头和 NPU

```bash
v4l2-ctl --list-devices
bash tools/check_elf2_npu.sh
```

### 7. 启动程序

```bash
./run_assistant_online_offline.sh
```

或使用 HDMI/桌面启动包装脚本：

```bash
DISPLAY=:0 XAUTHORITY=/home/elf/.Xauthority ./start_lip_gui.sh
```

## 配置项

主要配置通过环境变量覆盖：

| 变量 | 作用 |
|---|---|
| `CAMERA_DEVICE` | 摄像头设备节点 |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | 摄像头分辨率 |
| `CAMERA_FPS` | 摄像头帧率 |
| `CAMERA_GAIN` | 摄像头增益 |
| `CAMERA_BRIGHTNESS` | 摄像头亮度 |
| `CAMERA_CONTRAST` | 摄像头对比度 |
| `CAMERA_SHARPNESS` | 摄像头锐度 |
| `REMOTE_LLM_ENABLED` | 是否启用远程 LLM |
| `REMOTE_LLM_BASE_URL` | 远程 OpenAI-compatible 服务地址 |
| `REMOTE_LLM_MODEL` | 远程模型名 |
| `REMOTE_LLM_API_KEY` | 远程 API Key |
| `LOCAL_LLM_BASE_URL` | 本地 RKLLAMA 地址 |
| `LOCAL_LLM_MODEL` | 本地模型名 |
| `AHT20_I2C_BUS` | AHT20 I2C 总线编号 |
| `AHT20_I2C_ADDR` | AHT20 I2C 地址 |
| `DLIB_LANDMARK_MODEL` | dlib 68 点模型路径 |

## 开机自启动

`start_lip_gui.sh` 会：

1. 等待桌面、HDMI 和摄像头初始化；
2. 激活 `lipv4l` 环境；
3. 将日志写入 `logs/autostart.log`；
4. 执行 `run_assistant_online_offline.sh`。

可创建桌面自启动文件：

```ini
[Desktop Entry]
Type=Application
Name=Lip Reading Assistant
Exec=/home/elf/lip_assistant_interfaces/start_lip_gui.sh
Terminal=false
X-GNOME-Autostart-enabled=true
```

保存到：

```text
/home/elf/.config/autostart/lip_display.desktop
```

然后执行：

```bash
chmod 644 ~/.config/autostart/lip_display.desktop
chmod +x /home/elf/lip_assistant_interfaces/start_lip_gui.sh
```

## 调试工具

```bash
# 检查 RK3588、RKNPU 设备节点、驱动和 8080 服务
bash tools/check_elf2_npu.sh

# 检查/修复 main.py 中意外断裂的字符串
python3 tools/fix_main_broken_strings.py

# 独立读取 AHT20
python3 env_sensor.py

# 预览嘴部 ROI
python3 tools/mouth_crop_preview.py
```

## 上传前必须检查

- 不要提交 `.deepseek_api_key`、`.env` 或其他密钥文件。
- 不要提交 `logs/`、录制视频、推理中间文件、结果缓存和 `__pycache__/`。
- 确认 `display_service.py` 已加入仓库根目录。
- 确认模型、数据集和第三方 PDF/权重允许公开再分发。
- 大模型、RKNN、ONNX、DAT 等二进制文件优先使用 Git LFS 或 Release。
- `env_llm_context.py` 当前存在传感器/网络失败时使用演示兜底值的逻辑；正式公开或用于真实测量前，建议改为明确返回“数据不可用”，避免展示非真实温湿度。

## 许可证

请根据项目实际情况选择许可证，并分别遵守以下内容的原始许可证：

- 唇语识别模型和训练代码
- dlib 68 点关键点模型
- RKNN/RKLLAMA 运行时
- 第三方硬件资料和文档

在尚未确认全部第三方内容的再分发权限之前，建议先将 GitHub 仓库设为 Private。
