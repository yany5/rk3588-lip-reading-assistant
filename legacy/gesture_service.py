# =====================================================================
# MODIFICATION NOTICE:
# - This module replaces the legacy 'ok_gesture_service.py'.
# - Deleted: Single OK gesture detection and the corresponding single-event queue.
# - Added: Multi-gesture detection ('ok', '1', '2') using MediaPipe Hands,
#   multi-counter debounce holding logic, and a generic gesture event queue.
# - Added: Gesture simulation support (simulate_gesture) for keyboard testing.
# =====================================================================

import cv2
import time
import math
import queue
import threading

from config import OK_HOLD_FRAMES, OK_DETECT_EVERY_N_FRAMES

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception as e:
    print("[WARN] mediapipe 导入失败，手势识别不可用。")
    print("[WARN]", e)
    HAS_MEDIAPIPE = False
    mp = None


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class GestureService:
    """
    手势识别服务：
    1. 从 CameraService 获取最新帧
    2. 后台线程检测并分类手势："ok", "1", "2"
    3. 稳定保持指定帧数后，放入事件队列
    4. 支持模拟输入（键盘按键模拟手势）
    """

    def __init__(self, camera_service, stop_event):
        self.camera = camera_service
        self.stop_event = stop_event

        self.thread = None
        self.gesture_events = queue.Queue()

        # 当前实时检测到的手势
        self.current_gesture = None
        
        # 稳定帧数计数器
        self.ok_count = 0
        self.g1_count = 0
        self.g2_count = 0
        
        # 上一次触发时间，用于防抖冷却
        self.last_trigger_time = 0.0

        self.mp_hands = None
        self.mp_draw = None
        self.hands = None

        self.frame_index = 0

    def start(self):
        if HAS_MEDIAPIPE:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            print("[GESTURE] MediaPipe Hands 已启动。")
        else:
            print("[GESTURE] MediaPipe 不可用，只能用键盘模拟手势。")

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[GESTURE] GestureService 已启动。")

    def _loop(self):
        while not self.stop_event.is_set():
            frame = self.camera.get_frame()

            if frame is None:
                time.sleep(0.03)
                continue

            self.frame_index += 1

            if self.frame_index % OK_DETECT_EVERY_N_FRAMES != 0:
                time.sleep(0.005)
                continue

            detected_gesture = None

            if HAS_MEDIAPIPE and self.hands is not None:
                detected_gesture = self._detect_gesture(frame)

            self._push_event_if_needed(detected_gesture)

            time.sleep(0.005)

        print("[GESTURE] GestureService 线程退出。")

    def _detect_gesture(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        def pt(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        thumb_tip = pt(4)
        index_tip = pt(8)
        wrist = pt(0)
        middle_mcp = pt(9)

        palm_size = distance(wrist, middle_mcp)
        if palm_size < 1:
            palm_size = 1

        close_ratio = distance(thumb_tip, index_tip) / palm_size

        # 定义手指是否伸直（tip 的 y 坐标明显小于 pip 的 y 坐标）
        def finger_extended(tip_id, pip_id):
            return lm[tip_id].y < lm[pip_id].y - 0.02

        index_ext = finger_extended(8, 6)
        middle_ext = finger_extended(12, 10)
        ring_ext = finger_extended(16, 14)
        pinky_ext = finger_extended(20, 18)

        # 1. OK手势：大拇指和食指贴近，且其余手指（中指、无名指、小指）至少两根伸直
        extended_count = sum([middle_ext, ring_ext, pinky_ext])
        if close_ratio < 0.38 and extended_count >= 2:
            return "ok"

        # 2. 手势 1：食指伸直，中指、无名指、小指均折叠
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "1"

        # 3. 手势 2：食指和中指伸直，无名指、小指折叠
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "2"

        return None

    def _push_event_if_needed(self, detected_gesture):
        now = time.time()

        self.current_gesture = detected_gesture

        # 更新计数器
        if detected_gesture == "ok":
            self.ok_count += 1
            self.g1_count = 0
            self.g2_count = 0
        elif detected_gesture == "1":
            self.ok_count = 0
            self.g1_count += 1
            self.g2_count = 0
        elif detected_gesture == "2":
            self.ok_count = 0
            self.g1_count = 0
            self.g2_count += 1
        else:
            self.ok_count = 0
            self.g1_count = 0
            self.g2_count = 0

        # 判断是否达到稳定帧数
        target_gesture = None
        if self.ok_count >= OK_HOLD_FRAMES:
            target_gesture = "ok"
        elif self.g1_count >= OK_HOLD_FRAMES:
            target_gesture = "1"
        elif self.g2_count >= OK_HOLD_FRAMES:
            target_gesture = "2"

        if target_gesture is not None:
            # 防抖冷却：1.2秒内只触发一次
            if now - self.last_trigger_time > 1.2:
                self.last_trigger_time = now
                self.gesture_events.put((now, target_gesture))
                print(f"[GESTURE] 检测到手势: {target_gesture}")
                # 触发后重置计数器防止重复触发
                self.ok_count = 0
                self.g1_count = 0
                self.g2_count = 0

    def simulate_gesture(self, gesture_name):
        self.gesture_events.put((time.time(), gesture_name))
        print(f"[KEY] 模拟手势事件: {gesture_name}")

    def clear_events(self):
        while True:
            try:
                self.gesture_events.get_nowait()
            except queue.Empty:
                break

    def get_event(self, timeout=0.1):
        try:
            return self.gesture_events.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_release(self, stable_sec=0.45, max_wait=5.0):
        """
        防止同一个手势连续触发多个阶段。
        """
        start = time.time()
        release_start = None

        while time.time() - start < max_wait and not self.stop_event.is_set():
            if self.current_gesture is None:
                if release_start is None:
                    release_start = time.time()
                if time.time() - release_start >= stable_sec:
                    return True
            else:
                release_start = None

            time.sleep(0.05)

        return False

    def stop(self):
        if self.thread is not None:
            self.thread.join(timeout=2)

        if HAS_MEDIAPIPE and self.hands is not None:
            self.hands.close()

        print("[GESTURE] GestureService 已停止。")
