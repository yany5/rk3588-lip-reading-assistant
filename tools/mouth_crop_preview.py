# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


class MouthCropPreviewer:
    """
    自动嘴唇裁剪预览器，不保存视频。

    优先级：
    1. dlib 68 点模型：取 48~67 嘴部关键点裁剪；
    2. MediaPipe FaceMesh：取嘴唇关键点裁剪；
    3. 如果都不可用，不再用固定框假装裁剪，直接提示未检测到嘴唇。
    """

    def __init__(self, crop_size=(260, 130)):
        self.crop_w, self.crop_h = crop_size
        self.last_box = None
        self.last_method = "none"
        self.last_error = ""

        self.dlib = None
        self.dlib_detector = None
        self.dlib_predictor = None

        self.has_mediapipe = False
        self.face_mesh = None

        self._init_dlib()
        self._init_mediapipe()

        self.mp_lip_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            415, 310, 311, 312, 13, 82, 81, 42, 183, 78
        ]

    def _init_dlib(self):
        model_path = os.getenv(
            "DLIB_LANDMARK_MODEL",
            "/home/elf/lip_assistant_interfaces/models/shape_predictor_68_face_landmarks.dat"
        )

        if not os.path.exists(model_path):
            self.last_error = f"dlib模型不存在：{model_path}"
            print("[MOUTH]", self.last_error)
            return

        try:
            import dlib
            self.dlib = dlib
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(model_path)
            print(f"[MOUTH] dlib 68点嘴唇关键点模型已加载：{model_path}")
        except Exception as e:
            self.last_error = f"dlib 初始化失败：{e}"
            print("[MOUTH]", self.last_error)

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.has_mediapipe = True
            print("[MOUTH] MediaPipe FaceMesh 可用：作为第二裁剪方案。")
        except Exception as e:
            print(f"[MOUTH] MediaPipe FaceMesh 不可用：{e}")
            self.has_mediapipe = False

    def _clip_box(self, x1, y1, x2, y2, w, h):
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(1, min(int(x2), w))
        y2 = max(1, min(int(y2), h))

        if x2 <= x1 + 8 or y2 <= y1 + 8:
            return None

        return x1, y1, x2, y2

    def _expand_box(self, x1, y1, x2, y2, img_w, img_h, sx=1.75, sy=2.20):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(32, (x2 - x1) * sx)
        bh = max(24, (y2 - y1) * sy)

        return self._clip_box(
            cx - bw / 2,
            cy - bh / 2,
            cx + bw / 2,
            cy + bh / 2,
            img_w,
            img_h,
        )

    def _detect_by_dlib(self, frame):
        if self.dlib is None or self.dlib_detector is None or self.dlib_predictor is None:
            return None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 上采样 0 更快；如果检测不到脸，可以改成 1，但会更慢
        faces = self.dlib_detector(rgb, 0)

        if len(faces) == 0:
            return None

        # 取面积最大的人脸
        face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
        shape = self.dlib_predictor(rgb, face)

        pts = []
        # dlib 68 点中，48~67 是嘴部区域
        for i in range(48, 68):
            pts.append((shape.part(i).x, shape.part(i).y))

        pts = np.array(pts, dtype=np.float32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)

        box = self._expand_box(x1, y1, x2, y2, w, h, sx=1.85, sy=2.35)
        if box is not None:
            self.last_method = "dlib-68-mouth"
        return box

    def _detect_by_mediapipe(self, frame):
        if not self.has_mediapipe or self.face_mesh is None:
            return None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark
        xs, ys = [], []

        for idx in self.mp_lip_indices:
            if idx < len(lm):
                xs.append(lm[idx].x * w)
                ys.append(lm[idx].y * h)

        if not xs or not ys:
            return None

        box = self._expand_box(
            min(xs), min(ys), max(xs), max(ys),
            w, h,
            sx=1.75,
            sy=2.20,
        )

        if box is not None:
            self.last_method = "mediapipe-facemesh-mouth"
        return box

    def detect_mouth_box(self, frame):
        self.last_box = None
        self.last_method = "not_found"

        if frame is None:
            return None

        box = self._detect_by_dlib(frame)

        if box is None:
            box = self._detect_by_mediapipe(frame)

        self.last_box = box
        return box

    def crop(self, frame):
        if frame is None:
            return None, None

        box = self.detect_mouth_box(frame)
        if box is None:
            return None, None

        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None, box

        roi = cv2.resize(roi, (self.crop_w, self.crop_h))
        return roi, box

    def draw_box_on_full_frame(self, frame, box=None):
        if frame is None:
            return None

        out = frame.copy()
        box = box or self.last_box

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 180), 3)
            cv2.putText(
                out,
                f"Mouth ROI: {self.last_method}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 180),
                2,
                cv2.LINE_AA,
            )
        else:
            msg = "No lip landmarks"
            cv2.putText(
                out,
                msg,
                (40, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 90, 255),
                2,
                cv2.LINE_AA,
            )

        return out

    def close(self):
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass
