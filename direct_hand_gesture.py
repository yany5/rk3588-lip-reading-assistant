# -*- coding: utf-8 -*-
"""
严格版 OK / 1 / 2 手势识别。

核心目标：
1. 避免 1 被识别成 2；
2. 避免 OK 被识别成 2；
3. 主菜单中 OK 优先由 main.py 判断，这里 raw 只提供尽量可靠的 1/2/ok。
"""

import math
import time


class DirectHandGestureDetector:
    def __init__(self, min_detection_confidence=0.45, min_tracking_confidence=0.45):
        import mediapipe as mp

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.last_print_t = 0.0

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _finger_extended_y(lm, tip, pip, mcp):
        """
        严格伸直判断，适合你现在正对摄像头、手指向上的用法。

        图像坐标 y 越小越靠上：
        - tip 必须明显高于 pip；
        - pip 不能明显低于 mcp；
        这样能避免弯曲的中指被误判为伸直。
        """
        tip_y = lm[tip].y
        pip_y = lm[pip].y
        mcp_y = lm[mcp].y

        return (tip_y < pip_y - 0.025) and (pip_y < mcp_y + 0.035)

    @staticmethod
    def _finger_folded_y(lm, tip, pip, mcp):
        """
        收起判断。
        tip 没有明显高于 pip，就认为收起。
        """
        tip_y = lm[tip].y
        pip_y = lm[pip].y
        return tip_y > pip_y - 0.010

    def detect(self, frame_bgr):
        if frame_bgr is None:
            return "", {"reason": "no_frame"}

        import cv2

        h, w = frame_bgr.shape[:2]

        # 保持较高输入宽度，减少远距离手指关键点抖动
        scale_w = 800
        if w > scale_w:
            new_h = int(h * scale_w / w)
            small = cv2.resize(frame_bgr, (scale_w, new_h))
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return "", {"reason": "no_hand"}

        lm = result.multi_hand_landmarks[0].landmark

        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        box_size = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

        thumb_index_d = self._dist(lm[4], lm[8])
        pinch_ratio = thumb_index_d / box_size

        index_up = self._finger_extended_y(lm, 8, 6, 5)
        middle_up = self._finger_extended_y(lm, 12, 10, 9)
        ring_up = self._finger_extended_y(lm, 16, 14, 13)
        pinky_up = self._finger_extended_y(lm, 20, 18, 17)

        middle_folded = self._finger_folded_y(lm, 12, 10, 9)
        ring_folded = self._finger_folded_y(lm, 16, 14, 13)
        pinky_folded = self._finger_folded_y(lm, 20, 18, 17)

        # raw OK 只在非常明显捏合时输出。
        # 主菜单/结果页里 main.py 还会用更宽的 ok_like 兜底。
        raw_ok = pinch_ratio < 0.23

        # 先判最明确的 OK：捏合明显，并且不是清楚的 1/2 展指。
        if raw_ok and not (index_up and middle_up):
            gesture = "ok"

        # 2：食指和中指伸直；无名指和小指必须收起。
        elif index_up and middle_up and ring_folded and pinky_folded:
            gesture = "2"

        # 1：食指伸直；中指、无名指、小指都必须收起。
        elif index_up and middle_folded and ring_folded and pinky_folded:
            gesture = "1"

        else:
            gesture = ""

        debug = {
            "gesture": gesture,
            "index_up": index_up,
            "middle_up": middle_up,
            "ring_up": ring_up,
            "pinky_up": pinky_up,
            "middle_folded": middle_folded,
            "ring_folded": ring_folded,
            "pinky_folded": pinky_folded,
            "thumb_index_d": thumb_index_d,
            "box_size": box_size,
            "pinch_ratio": pinch_ratio,
            "index_tip_y": lm[8].y,
            "index_pip_y": lm[6].y,
            "middle_tip_y": lm[12].y,
            "middle_pip_y": lm[10].y,
        }

        now = time.time()
        if now - self.last_print_t > 0.25:
            print(
                f"[DIRECT_GESTURE] raw={gesture or '-'} "
                f"idx={index_up} mid={middle_up} ring={ring_up} pinky={pinky_up} "
                f"fold_mid={middle_folded} fold_ring={ring_folded} fold_pinky={pinky_folded} "
                f"pinch={pinch_ratio:.3f}"
            )
            self.last_print_t = now

        return gesture, debug
