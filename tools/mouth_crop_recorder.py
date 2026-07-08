# -*- coding: utf-8 -*-
"""
嘴部 ROI 裁剪与录制模块。

功能：
1. 录制阶段接收全帧 frame；
2. 自动裁剪嘴部区域；
3. 保存 mouth_roi.mp4；
4. 同时保存少量 jpg 帧，方便检查裁剪效果；
5. 不强依赖 mediapipe：
   - 有 mediapipe：用 FaceMesh 嘴唇关键点裁剪；
   - 没有 mediapipe：用 OpenCV Haar 人脸检测 + 下半脸估计嘴部；
   - 连人脸也检测不到：用画面中心偏下区域兜底。
"""

import os
import cv2
import time
from pathlib import Path


class MouthCropRecorder:
    def __init__(
        self,
        output_root="/home/elf/lip_assistant_interfaces/recordings",
        fps=25,
        mouth_size=(224, 112),
        save_debug_frames=True,
    ):
        self.output_root = Path(output_root)
        self.fps = int(fps)
        self.mouth_w, self.mouth_h = mouth_size
        self.save_debug_frames = save_debug_frames

        self.session_dir = None
        self.frames_dir = None
        self.video_path = None
        self.writer = None
        self.frame_index = 0
        self.last_box = None

        self.has_mediapipe = False
        self.face_mesh = None
        self.mp_face_mesh = None

        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.has_mediapipe = True
            print("[MOUTH] MediaPipe FaceMesh 可用：优先使用关键点裁剪嘴部。")
        except Exception as e:
            print(f"[MOUTH] MediaPipe FaceMesh 不可用，改用 OpenCV Haar/中心兜底裁剪：{e}")
            self.has_mediapipe = False

        self.face_cascade = None
        try:
            cascade_path = os.path.join(
                cv2.data.haarcascades,
                "haarcascade_frontalface_default.xml",
            )
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
                print("[MOUTH] Haar 人脸模型加载失败，将只使用中心兜底裁剪。")
            else:
                print("[MOUTH] Haar 人脸模型可用：用于无 MediaPipe 时估计嘴部。")
        except Exception as e:
            print(f"[MOUTH] Haar 初始化失败：{e}")
            self.face_cascade = None

        # MediaPipe FaceMesh 嘴唇区域常用关键点索引
        self.lip_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            415, 310, 311, 312, 13, 82, 81, 42, 183, 78
        ]

    def start(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_root / f"lip_record_{ts}"
        self.frames_dir = self.session_dir / "mouth_frames"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = str(self.session_dir / "mouth_roi.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.video_path,
            fourcc,
            self.fps,
            (self.mouth_w, self.mouth_h),
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建嘴部 ROI 视频文件：{self.video_path}")

        self.frame_index = 0
        self.last_box = None

        print(f"[MOUTH] 开始嘴部 ROI 录制：{self.video_path}")
        return self.video_path

    def _clip_box(self, x1, y1, x2, y2, w, h):
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(1, min(int(x2), w))
        y2 = max(1, min(int(y2), h))

        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return None

        return x1, y1, x2, y2

    def _expand_box(self, x1, y1, x2, y2, img_w, img_h, scale_x=1.45, scale_y=1.75):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * scale_x
        bh = (y2 - y1) * scale_y

        return self._clip_box(
            cx - bw / 2,
            cy - bh / 2,
            cx + bw / 2,
            cy + bh / 2,
            img_w,
            img_h,
        )

    def _detect_mouth_mediapipe(self, frame):
        if not self.has_mediapipe or self.face_mesh is None:
            return None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        xs, ys = [], []

        for idx in self.lip_indices:
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            xs.append(lm.x * w)
            ys.append(lm.y * h)

        if not xs or not ys:
            return None

        return self._expand_box(
            min(xs),
            min(ys),
            max(xs),
            max(ys),
            w,
            h,
            scale_x=1.55,
            scale_y=1.85,
        )

    def _detect_mouth_haar(self, frame):
        if self.face_cascade is None:
            return None

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(120, 120),
        )

        if len(faces) == 0:
            return None

        # 取面积最大的人脸
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        x, y, fw, fh = faces[0]

        # 嘴部大致位于人脸下半部
        mx1 = x + int(0.18 * fw)
        mx2 = x + int(0.82 * fw)
        my1 = y + int(0.55 * fh)
        my2 = y + int(0.88 * fh)

        return self._clip_box(mx1, my1, mx2, my2, w, h)

    def _detect_mouth_fallback(self, frame):
        h, w = frame.shape[:2]

        # 面向固定摄像头调镜头场景：默认嘴部在画面中间偏下
        crop_w = int(w * 0.46)
        crop_h = int(h * 0.26)
        cx = w // 2
        cy = int(h * 0.55)

        return self._clip_box(
            cx - crop_w // 2,
            cy - crop_h // 2,
            cx + crop_w // 2,
            cy + crop_h // 2,
            w,
            h,
        )

    def detect_mouth_box(self, frame):
        box = self._detect_mouth_mediapipe(frame)

        if box is None:
            box = self._detect_mouth_haar(frame)

        if box is None:
            box = self._detect_mouth_fallback(frame)

        self.last_box = box
        return box

    def crop_mouth(self, frame):
        if frame is None:
            return None, None

        box = self.detect_mouth_box(frame)
        if box is None:
            return None, None

        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None, box

        crop = cv2.resize(crop, (self.mouth_w, self.mouth_h))
        return crop, box

    def write(self, frame):
        if self.writer is None:
            raise RuntimeError("MouthCropRecorder 尚未 start()")

        crop, box = self.crop_mouth(frame)
        if crop is None:
            return None, box

        self.writer.write(crop)

        if self.save_debug_frames and self.frame_index % 5 == 0:
            img_path = self.frames_dir / f"mouth_{self.frame_index:05d}.jpg"
            cv2.imwrite(str(img_path), crop)

        self.frame_index += 1
        return crop, box

    def draw_debug_box(self, frame, box=None):
        if frame is None:
            return frame

        out = frame.copy()
        box = box or self.last_box

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 180), 3)
            cv2.putText(
                out,
                "Mouth ROI",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 180),
                2,
                cv2.LINE_AA,
            )

        return out

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass

        print(f"[MOUTH] 嘴部 ROI 录制结束：{self.video_path}")
        print(f"[MOUTH] 共写入帧数：{self.frame_index}")

        return self.video_path
