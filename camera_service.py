import cv2
import time
import threading
import subprocess

from config import (
    DEVICE,
    WIDTH,
    HEIGHT,
    FPS,
    GAIN,
    BRIGHTNESS,
    CONTRAST,
    SHARPNESS,
)


def run_cmd(cmd):
    try:
        print("[CMD]", cmd)
        subprocess.run(cmd, shell=True, check=False)
    except Exception as e:
        print("[WARN] 命令执行失败:", cmd, e)


def set_camera_controls(set_format=True):
    """
    摄像头曝光部分保持你之前的版本，不做智能判断、不删除。
    你之前说这部分不用改，所以这里原样保留。
    """
    print("\n正在设置摄像头为自动曝光模式...")

    cmds = []

    if set_format:
        cmds += [
            f"v4l2-ctl -d {DEVICE} --set-fmt-video=width={WIDTH},height={HEIGHT},pixelformat=MJPG",
            f"v4l2-ctl -d {DEVICE} --set-parm={FPS}",
        ]

    cmds += [
        f"v4l2-ctl -d {DEVICE} --set-ctrl=exposure_auto=3",
        f"v4l2-ctl -d {DEVICE} --set-ctrl=exposure_auto_priority=0",
        f"v4l2-ctl -d {DEVICE} --set-ctrl=gain={GAIN}",
        f"v4l2-ctl -d {DEVICE} --set-ctrl=brightness={BRIGHTNESS}",
        f"v4l2-ctl -d {DEVICE} --set-ctrl=contrast={CONTRAST}",
        f"v4l2-ctl -d {DEVICE} --set-ctrl=sharpness={SHARPNESS}",
        f"v4l2-ctl -d {DEVICE} --set-ctrl=power_line_frequency=1",
        f"v4l2-ctl -d {DEVICE} --set-ctrl=white_balance_temperature_auto=1",
    ]

    for cmd in cmds:
        run_cmd(cmd)

    print("自动曝光设置完成。\n")


def dump_camera_controls():
    print("\n========== 当前摄像头关键参数 ==========")
    run_cmd(
        f"v4l2-ctl -d {DEVICE} "
        f"--get-ctrl=exposure_auto,exposure_auto_priority,gain,brightness,contrast,sharpness,power_line_frequency,white_balance_temperature_auto"
    )
    print("======================================\n")


class CameraService:
    """
    摄像头接口：
    1. 程序启动后打开一次摄像头
    2. 后台线程一直读取最新帧
    3. 其他模块只通过 get_frame() 拿最新帧
    """

    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.cap = None
        self.thread = None

        self.lock = threading.Lock()
        self.latest_frame = None

        self.real_width = WIDTH
        self.real_height = HEIGHT
        self.real_fps = FPS

    def start(self):
        set_camera_controls(set_format=True)

        self.cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头：{DEVICE}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

        # 保持你原来的逻辑：打开后再设置一次
        set_camera_controls(set_format=False)

        for _ in range(20):
            self.cap.read()
            time.sleep(0.02)

        self.real_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.real_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.real_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"摄像头已打开：{DEVICE}")
        print(f"实际参数：{self.real_width}x{self.real_height}, FPS={self.real_fps}")

        dump_camera_controls()

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        print("[CAM] CameraService 已启动：摄像头会一直保持打开。")

    def _loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()

            if not ret or frame is None:
                print("[WARN] 摄像头读取失败，继续尝试。")
                time.sleep(0.03)
                continue

            with self.lock:
                self.latest_frame = frame.copy()

        print("[CAM] CameraService 线程退出。")

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        if self.thread is not None:
            self.thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()

        print("[CAM] 摄像头已释放。")
