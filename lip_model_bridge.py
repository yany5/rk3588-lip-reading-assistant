# -*- coding: utf-8 -*-
import os
import re
import csv
import sys
import time
import subprocess
from pathlib import Path


MODEL_DIR = Path.home() / "桌面" / "lip-model"
RUNTIME_VIDEO_DIR = MODEL_DIR / "runtime_videos"
INFER_VIDEO_DIR = MODEL_DIR / "runtime_infer_one"
OUTPUT_PKL = MODEL_DIR / "lrw1000_pkl_output" / "latest_video.pkl"
RESULT_CSV = MODEL_DIR / "results" / "latest_result.csv"
FRONTEND_RKNN = MODEL_DIR / "lrw1000_frontend_window.rknn"
BACKEND_ONNX = MODEL_DIR / "lrw1000_backend_gru.onnx"
LABEL_INDEX = MODEL_DIR / "label_lrw1000_with_id.txt"
RECOGNIZE_SCRIPT = MODEL_DIR / "recognize_latest_video_lrw1000.py"

TARGET_FPS = 25.0
VIDEO_PATTERN = re.compile(r"^video(\d+)\.mp4$", re.IGNORECASE)


def ensure_model_files():
    missing = []
    for p in [MODEL_DIR, RECOGNIZE_SCRIPT, FRONTEND_RKNN, BACKEND_ONNX, LABEL_INDEX]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise RuntimeError("缺少唇语模型文件：" + " | ".join(missing))


def next_video_path():
    RUNTIME_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    max_idx = -1
    for p in RUNTIME_VIDEO_DIR.iterdir():
        m = VIDEO_PATTERN.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return RUNTIME_VIDEO_DIR / f"video{max_idx + 1:04d}.mp4"


def latest_video_path():
    if not RUNTIME_VIDEO_DIR.exists():
        raise RuntimeError(f"没有录制目录：{RUNTIME_VIDEO_DIR}")
    candidates = []
    for p in RUNTIME_VIDEO_DIR.iterdir():
        m = VIDEO_PATTERN.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise RuntimeError(f"没有找到 videoXXXX.mp4：{RUNTIME_VIDEO_DIR}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]



def prepare_exact_video_dir(video_path):
    """
    recognize_latest_video_lrw1000.py 只能接收 video-dir，并自动找最新 videoXXXX.mp4。
    为了确保它处理的就是本次刚录的视频，这里创建 runtime_infer_one/video0000.mp4
    指向本次视频。
    """
    import shutil

    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise RuntimeError(f"录制视频不存在：{video_path}")

    INFER_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    target = INFER_VIDEO_DIR / "video0000.mp4"

    try:
        if target.exists() or target.is_symlink():
            target.unlink()
    except Exception:
        pass

    try:
        target.symlink_to(video_path)
    except Exception:
        shutil.copy2(str(video_path), str(target))

    return INFER_VIDEO_DIR


def run_lip_model(video_path=None, topk=5, timeout=120):
    """
    返回：
    {
      ok, word, pinyin, class_id, prob, csv, stdout, stderr
    }
    """
    ensure_model_files()

    if video_path is None:
        video_path = latest_video_path()
    video_path = Path(video_path)

    if not video_path.exists():
        raise RuntimeError(f"录制视频不存在：{video_path}")

    infer_video_dir = prepare_exact_video_dir(video_path)

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RECOGNIZE_SCRIPT),
        "--video-dir", str(infer_video_dir),
        "--output-pkl", str(OUTPUT_PKL),
        "--result-csv", str(RESULT_CSV),
        "--frontend-rknn", str(FRONTEND_RKNN),
        "--backend-onnx", str(BACKEND_ONNX),
        "--label-index", str(LABEL_INDEX),
        "--topk", str(topk),
        "--threads", "1",
        "--frontend-data-format", "nchw",
        "--max-consecutive-fail", "120",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(MODEL_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "唇语识别脚本失败\n"
            + "CMD: " + " ".join(cmd) + "\n"
            + "STDOUT:\n" + proc.stdout + "\n"
            + "STDERR:\n" + proc.stderr
        )

    if not RESULT_CSV.exists():
        raise RuntimeError(f"没有生成结果文件：{RESULT_CSV}")

    with open(RESULT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"结果 CSV 为空：{RESULT_CSV}")

    best = rows[0]
    return {
        "ok": True,
        "word": best.get("word", "").strip(),
        "pinyin": best.get("pinyin", "").strip(),
        "class_id": best.get("class_id", "").strip(),
        "prob": best.get("prob", "").strip(),
        "csv": str(RESULT_CSV),
        "video": str(video_path),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def is_temperature_intent(word):
    word = "" if word is None else str(word)
    keys = ["温度", "湿度", "气温", "天气", "中央气象台", "多少度", "几度", "冷", "热"]
    return any(k in word for k in keys)


def is_time_intent(word):
    word = "" if word is None else str(word)
    keys = ["时间", "北京时间", "现在", "几点", "上午", "下午", "中午", "今天"]
    return any(k in word for k in keys)


if __name__ == "__main__":
    print(run_lip_model())


def close_lip_model():
    """
    兼容 main.py 的模型关闭接口。

    当前唇语模型资源由 Python 进程退出时自动释放，
    因此这里保留空实现，避免 main.py 导入失败。
    """
    return None
