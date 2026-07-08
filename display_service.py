# =====================================================================
# MODIFICATION NOTICE (Plan B UI Version):
# - Replaced white background with a full-screen black OLED display wrapped
#   in a beautifully rendered 3D-styled white plastic device casing.
# - Replaced hollow line-art expressions with thick, solid-filled pixel-neon
#   eyes and mouths featuring cyan-blue glowing effects.
# - Added dynamic blinking animations (0.15s blink every 4 seconds).
# - Revamped all state expressions (idle, listening, thinking, speaking).
# - Changed all floating panels to neon-bordered dark cards with glow-stroke text.
# - FIXED: Corrected parameter from 'stroke_fill' to 'stroke_rgb' in _draw_neon_text.
# - FIXED: Standardized camera preview to normal color stream enclosed in frame.
# =====================================================================

import os
import cv2
import math
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import WIDTH, HEIGHT, WINDOW_NAME, OK_HOLD_FRAMES


class DisplayService:
    """
    HDMI/OpenCV Plan B: 3D 塑料外壳包裹的全屏 OLED 霓虹蓝交互界面
    """

    def __init__(self):
        self.window_created = False

        # 逻辑画布仍使用 config.py 中的 WIDTH×HEIGHT（当前界面按 1280×720 设计），
        # 最终显示时再等比例放大到 HDMI 的 1920×1080。两个尺寸都是 16:9，
        # 因此不会拉伸变形，也不用重写现有界面坐标。
        self.output_x = int(os.environ.get("LIP_HDMI_X", "0"))
        self.output_y = int(os.environ.get("LIP_HDMI_Y", "0"))
        self.output_width = int(os.environ.get("LIP_HDMI_WIDTH", "1920"))
        self.output_height = int(os.environ.get("LIP_HDMI_HEIGHT", "1080"))
        self._last_fullscreen_refresh = 0.0
        
        # 自动探测系统内置的中英文字体路径
        self.font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "C:/Windows/Fonts/simhei.ttf",  # 备用黑体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]

        # 预设 Plan B 青蓝色调 (BGR 格式)
        self.GLOW_BLUE = (255, 120, 0)       # 霓虹外发光蓝色
        self.BRIGHT_BLUE = (255, 220, 140)   # 核心高亮青白色
        self.DARK_BG = (28, 18, 10)          # 卡片面板暗色底背景
        self.SCREEN_BG = (16, 12, 8)         # OLED 屏幕底色
        
        # 预设警示红色调 (BGR 格式，用于退出)
        self.GLOW_RED = (50, 50, 255)
        self.BRIGHT_RED = (180, 180, 255)

    def _ensure_window(self):
        """创建并固定到 HDMI-1（1920×1080、坐标 0,0）的全屏窗口。"""
        if self.window_created:
            return

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        # 多显示器环境下，OpenCV 可能默认把窗口放到 DSI 屏。
        # HDMI-1 位于桌面坐标 (0, 0)，所以先移动、再设置物理尺寸、最后全屏。
        try:
            cv2.moveWindow(WINDOW_NAME, self.output_x, self.output_y)
            cv2.resizeWindow(WINDOW_NAME, self.output_width, self.output_height)

            # 先显示一帧，让窗口管理器真正创建窗口，再切换全屏。
            bootstrap = np.zeros(
                (self.output_height, self.output_width, 3),
                dtype=np.uint8,
            )
            cv2.imshow(WINDOW_NAME, bootstrap)
            cv2.waitKey(80)

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            cv2.moveWindow(WINDOW_NAME, self.output_x, self.output_y)
            cv2.waitKey(50)
        except Exception as exc:
            # 即使窗口管理器不接受 FULLSCREEN，也强制铺到 HDMI 的完整物理尺寸。
            print(f"[DISPLAY][WARN] OpenCV 全屏设置失败，改用无最大化尺寸：{exc}")
            try:
                cv2.moveWindow(WINDOW_NAME, self.output_x, self.output_y)
                cv2.resizeWindow(WINDOW_NAME, self.output_width, self.output_height)
            except Exception:
                pass

        self.window_created = True
        self._last_fullscreen_refresh = time.monotonic()
        print(
            "[DISPLAY] HDMI window ready: "
            f"logical={WIDTH}x{HEIGHT}, "
            f"output={self.output_width}x{self.output_height}+"
            f"{self.output_x}+{self.output_y}"
        )

    def _enforce_fullscreen(self):
        """周期性重申全屏状态，避免桌面窗口管理器把窗口恢复成普通大小。"""
        now = time.monotonic()
        if now - self._last_fullscreen_refresh < 1.5:
            return

        try:
            cv2.moveWindow(WINDOW_NAME, self.output_x, self.output_y)
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        except Exception:
            try:
                cv2.moveWindow(WINDOW_NAME, self.output_x, self.output_y)
                cv2.resizeWindow(WINDOW_NAME, self.output_width, self.output_height)
            except Exception:
                pass

        self._last_fullscreen_refresh = now

    def _show_canvas(self, canvas):
        """把 1280×720 逻辑画布放大到 HDMI 1920×1080 后显示。"""
        self._ensure_window()

        if canvas is None:
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        if (
            canvas.shape[1] != self.output_width
            or canvas.shape[0] != self.output_height
        ):
            output = cv2.resize(
                canvas,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            output = canvas

        cv2.imshow(WINDOW_NAME, output)
        self._enforce_fullscreen()

    # ==========================================
    # 🎨 核心绘图与 3D 外壳辅助函数
    # ==========================================

    def _draw_rounded_rect(self, img, pt1, pt2, color, thickness=-1, radius=20):
        """绘制圆角矩形"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        if thickness == -1:
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)

    def _draw_device_casing(self, canvas):
        """
        绘制 Plan B 的全屏 3D 白色质感外边框与内部大 OLED 屏幕
        """
        # 1. 填充屏幕外部区域底色 (偏深灰)
        canvas[:] = (50, 50, 48)

        # 2. 绘制最外圈 3D 投影
        self._draw_rounded_rect(canvas, (8, 8), (WIDTH - 8, HEIGHT - 8), (28, 28, 25), -1, 38)
        
        # 3. 绘制银白色塑料机壳主体 (暖白)
        self._draw_rounded_rect(canvas, (10, 10), (WIDTH - 10, HEIGHT - 10), (238, 240, 240), -1, 36)
        
        # 4. 绘制高光线条强化 3D 塑料感
        self._draw_rounded_rect(canvas, (12, 12), (WIDTH - 12, HEIGHT - 12), (255, 255, 255), 2, 34)
        self._draw_rounded_rect(canvas, (14, 14), (WIDTH - 14, HEIGHT - 14), (200, 202, 202), 1, 32)
        
        # 5. 绘制屏幕内嵌黑框
        self._draw_rounded_rect(canvas, (24, 24), (WIDTH - 24, HEIGHT - 24), (55, 55, 52), -1, 22)
        
        # 6. 填充 OLED 大屏幕显示底色
        self._draw_rounded_rect(canvas, (26, 26), (WIDTH - 26, HEIGHT - 26), self.SCREEN_BG, -1, 20)

    def _draw_screen_grid(self, canvas):
        """
        在 OLED 屏幕上绘制微弱的科技网格背景
        """
        grid_color = (26, 20, 14)  # 极其隐蔽的暗蓝线
        x_min, x_max = 26, WIDTH - 26
        y_min, y_max = 26, HEIGHT - 26
        
        for x in range(x_min + 40, x_max, 40):
            cv2.line(canvas, (x, y_min), (x, y_max), grid_color, 1)
        for y in range(y_min + 40, y_max, 40):
            cv2.line(canvas, (x_min, y), (x_max, y), grid_color, 1)


    def _get_pil_font(self, font_size):
        """获取可用字体。"""
        font = None
        for path in self.font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, font_size)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()
        return font

    def _text_size(self, draw, font, text, stroke_width=0):
        """兼容 Pillow 新旧版本的文本尺寸测量。"""
        text = "" if text is None else str(text)
        try:
            box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
            return box[2] - box[0], box[3] - box[1]
        except Exception:
            try:
                return draw.textsize(text, font=font)
            except Exception:
                return len(text) * 12, 20

    def _wrap_text_to_width(self, draw, font, text, max_width, stroke_width=0):
        """
        按像素宽度自动换行，兼容中文/英文混排。
        中文没有空格分词，所以这里按字符逐个测宽。
        """
        text = "" if text is None else str(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").strip()

        if not text:
            return [""]

        lines = []
        for para in text.split("\n"):
            para = para.strip()
            if not para:
                lines.append("")
                continue

            cur = ""
            for ch in para:
                test = cur + ch
                w, _ = self._text_size(draw, font, test, stroke_width=stroke_width)
                if w <= max_width:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = ch

            if cur:
                lines.append(cur)

        return lines

    def _ellipsize_to_width(self, draw, font, text, max_width, stroke_width=0):
        """让最后一行在宽度内显示，超出加省略号。"""
        text = "" if text is None else str(text)
        ell = "…"

        w, _ = self._text_size(draw, font, text, stroke_width=stroke_width)
        if w <= max_width:
            return text

        while text:
            w, _ = self._text_size(draw, font, text + ell, stroke_width=stroke_width)
            if w <= max_width:
                return text + ell
            text = text[:-1]

        return ell

    def _draw_text_box_pil(
        self,
        img,
        text,
        box,
        font_size=20,
        color_rgb=(230, 240, 245),
        stroke_width=0,
        stroke_rgb=(0, 0, 0),
        max_lines=None,
        line_gap=6,
        align="left",
        valign="top",
    ):
        """
        在固定矩形区域内绘制自动换行文字。

        关键修复：
        1. 先裁剪 ROI，再在 ROI 内绘制；
        2. 绘制完成后只把 ROI 贴回原图；
        3. 因此文字绝对不会画出 box 范围。
        """
        x, y, w, h = box
        x = int(max(0, x))
        y = int(max(0, y))
        w = int(max(1, w))
        h = int(max(1, h))

        # 防止 box 超出画布
        x2 = min(WIDTH, x + w)
        y2 = min(HEIGHT, y + h)
        if x >= WIDTH or y >= HEIGHT or x2 <= x or y2 <= y:
            return img

        roi_bgr = img[y:y2, x:x2].copy()
        roi_h, roi_w = roi_bgr.shape[:2]

        # 内边距，防止描边贴边后看起来越界
        pad_x = 10
        pad_y = 6
        usable_w = max(10, roi_w - pad_x * 2)
        usable_h = max(10, roi_h - pad_y * 2)

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_pil_font(font_size)

        lines = self._wrap_text_to_width(
            draw,
            font,
            text,
            max_width=usable_w,
            stroke_width=stroke_width,
        )

        _, sample_h = self._text_size(draw, font, "国", stroke_width=stroke_width)
        line_h = max(font_size + line_gap, sample_h + line_gap)

        height_limit_lines = max(1, usable_h // line_h)
        if max_lines is None:
            max_lines = height_limit_lines
        else:
            max_lines = max(1, min(max_lines, height_limit_lines))

        overflow = len(lines) > max_lines
        lines = lines[:max_lines]

        if overflow and lines:
            lines[-1] = self._ellipsize_to_width(
                draw,
                font,
                lines[-1],
                max_width=usable_w,
                stroke_width=stroke_width,
            )

        total_h = len(lines) * line_h - line_gap

        if valign == "center":
            yy = pad_y + max(0, (usable_h - total_h) // 2)
        elif valign == "bottom":
            yy = pad_y + max(0, usable_h - total_h)
        else:
            yy = pad_y

        for line in lines:
            tw, _ = self._text_size(draw, font, line, stroke_width=stroke_width)

            if align == "center":
                xx = pad_x + max(0, (usable_w - tw) // 2)
            elif align == "right":
                xx = pad_x + max(0, usable_w - tw)
            else:
                xx = pad_x

            draw.text(
                (xx, yy),
                line,
                font=font,
                fill=color_rgb,
                stroke_width=stroke_width,
                stroke_fill=stroke_rgb,
            )
            yy += line_h

        clipped_roi = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img[y:y2, x:x2] = clipped_roi
        return img

    def _draw_neon_text_box(
        self,
        img,
        text,
        pos,
        box_size,
        font_size=20,
        active=True,
        is_warning=False,
        max_lines=None,
        line_gap=8,
        align="left",
        valign="top",
    ):
        """
        霓虹风格多行文本框。
        用于 DeepSeek 回答、识别结果、错误提示等长文本。
        """
        x, y = pos
        w, h = box_size

        if is_warning:
            color_rgb = (255, 120, 120)
            stroke_rgb = (90, 0, 0)
        elif active:
            color_rgb = (175, 245, 255)
            stroke_rgb = (0, 105, 160)
        else:
            color_rgb = (185, 205, 215)
            stroke_rgb = (20, 60, 90)

        return self._draw_text_box_pil(
            img,
            text,
            (x, y, w, h),
            font_size=font_size,
            color_rgb=color_rgb,
            stroke_width=1,
            stroke_rgb=stroke_rgb,
            max_lines=max_lines,
            line_gap=line_gap,
            align=align,
            valign=valign,
        )


    def _draw_text(self, img, text, pos, font_size=20, color_rgb=(60, 60, 65), stroke_width=0, stroke_rgb=(0, 0, 0)):
        """使用 Pillow 渲染高抗锯齿中英文字体，支持描边"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        font = None
        for path in self.font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, font_size)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()
            
        if stroke_width > 0:
            draw.text(pos, text, font=font, fill=color_rgb, stroke_width=stroke_width, stroke_fill=stroke_rgb)
        else:
            draw.text(pos, text, font=font, fill=color_rgb)
            
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_neon_text(self, img, text, pos, font_size=20, active=True, is_warning=False):
        """
        绘制科技感边缘微发光的霓虹字体 (输入输出皆为 RGB 映射)
        """
        if is_warning:
            color_rgb = (255, 100, 100)      # 亮红
            stroke_rgb = (180, 20, 20)       # 暗红外描边
        elif active:
            color_rgb = (100, 220, 255)      # 亮青蓝
            stroke_rgb = (0, 120, 220)       # 科技蓝外描边
        else:
            color_rgb = (130, 150, 160)      # 灰蓝
            stroke_rgb = (40, 50, 60)        # 暗灰蓝描边
            
        return self._draw_text(img, text, pos, font_size, color_rgb, stroke_width=1, stroke_rgb=stroke_rgb)

    def _draw_neon_card(self, canvas, pt1, pt2, active=True, is_warning=False, radius=18):
        """
        绘制科技悬浮霓虹发光卡片底座
        """
        bg_col = self.DARK_BG
        if is_warning:
            glow_col = self.GLOW_RED
            bright_col = self.BRIGHT_RED
        elif active:
            glow_col = self.GLOW_BLUE
            bright_col = self.BRIGHT_BLUE
        else:
            glow_col = (100, 70, 40)
            bright_col = (130, 110, 95)
            
        # 1. 绘制暗色卡片背景
        self._draw_rounded_rect(canvas, pt1, pt2, bg_col, -1, radius)
        # 2. 双重叠描边描绘发光质感
        self._draw_rounded_rect(canvas, pt1, pt2, glow_col, 4, radius)
        self._draw_rounded_rect(canvas, pt1, pt2, bright_col, 1, radius)

    # ==========================================
    # 🌟 特效矢量图形绘制
    # ==========================================

    def _draw_star(self, img, cx, cy, size, color, thickness=-1):
        """绘制圆润五角星"""
        points = []
        for i in range(10):
            angle = i * math.pi / 5 - math.pi / 2
            r = size if i % 2 == 0 else size / 2.2
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            points.append((x, y))
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        if thickness == -1:
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)

    def _draw_heart(self, img, cx, cy, size, color, thickness=-1):
        """绘制圆润桃心"""
        points = []
        for t in np.linspace(0, 2*math.pi, 80):
            x = int(cx + (size * 1.3) * (16 * math.sin(t)**3) / 16)
            y = int(cy - (size * 1.3) * (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)) / 16)
            points.append((x, y))
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        if thickness == -1:
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)

    def _draw_neon_capsule(self, canvas, cx, cy, rx, ry, color_glow, color_bright, radius=18):
        """
        绘制粗线条、实心像素发光的胶囊结构 (核心眼睛逻辑)
        """
        x1, y1 = cx - rx, cy - ry
        x2, y2 = cx + rx, cy + ry
        
        # 1. 最外层扩散发光 (Glow)
        self._draw_rounded_rect(canvas, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), color_glow, -1, radius + 6)
        # 2. 中间层高饱和颜色 (Body)
        self._draw_rounded_rect(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color_bright, -1, radius + 2)
        # 3. 最内层核心发白光 (Core)
        inner_white = (
            min(color_bright[0] + 45, 255),
            min(color_bright[1] + 45, 255),
            min(color_bright[2] + 45, 255)
        )
        self._draw_rounded_rect(canvas, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), inner_white, -1, max(1, radius - 6))

    def _draw_neon_features(self, canvas, state):
        """
        渲染实心粗线条蓝色发光表情系统（附带眨眼和微动效）
        """
        # 计算面部中心坐标 (将位置上移，给底部文本框腾出空间)
        cx = WIDTH // 2
        cy = 210
        
        # 呼吸周期微调
        breath_offset = int(math.sin(time.time() * 2.5) * 4)
        cy_b = cy + breath_offset
        
        lex, rex = cx - 130, cx + 130
        ey = cy_b - 10
        my = cy_b + 65

        # 眨眼逻辑 (每隔 4 秒闭眼 0.15 秒)
        t_sec = time.time()
        is_blinking = (int(t_sec) % 4 == 0) and ((t_sec - int(t_sec)) < 0.15)

        # ==========================================
        # 🌸 绘制霓虹科技腮红 (低亮度蓝紫色，配合蓝色风格)
        # ==========================================
        blush_overlay = canvas.copy()
        if state in ["idle", "thinking", "speaking"]:
            cv2.ellipse(blush_overlay, (lex - 20, ey + 52), (38, 12), 0, 0, 360, (230, 100, 30), -1, cv2.LINE_AA)
            cv2.ellipse(blush_overlay, (rex + 20, ey + 52), (38, 12), 0, 0, 360, (230, 100, 30), -1, cv2.LINE_AA)
            cv2.addWeighted(blush_overlay, 0.35, canvas, 0.65, 0, canvas)
        elif state == "listening":
            # 科技感斜线腮红 (///)
            for offset in [-10, 0, 10]:
                cv2.line(canvas, (lex - 25 + offset, ey + 56), (lex - 18 + offset, ey + 44), self.GLOW_BLUE, 3, cv2.LINE_AA)
                cv2.line(canvas, (rex + 15 + offset, ey + 56), (rex + 22 + offset, ey + 44), self.GLOW_BLUE, 3, cv2.LINE_AA)

        # ==========================================
        # 👀 绘制眼睛部分
        # ==========================================
        if is_blinking and state != "thinking":
            # 眨眼动作：眼睛闭合呈一条粗横线
            self._draw_neon_capsule(canvas, lex, ey, 35, 6, self.GLOW_BLUE, self.BRIGHT_BLUE, 5)
            self._draw_neon_capsule(canvas, rex, ey, 35, 6, self.GLOW_BLUE, self.BRIGHT_BLUE, 5)
        else:
            if state == "idle":
                # 待机：像素感实心胶囊眼 + 侧边荧光副亮点 (参考图 3 萌感)
                self._draw_neon_capsule(canvas, lex, ey, 35, 28, self.GLOW_BLUE, self.BRIGHT_BLUE, 24)
                self._draw_neon_capsule(canvas, rex, ey, 35, 28, self.GLOW_BLUE, self.BRIGHT_BLUE, 24)
                
                # 侧下角发光亮点
                cv2.circle(canvas, (lex - 20, ey + 22), 8, self.GLOW_BLUE, -1, cv2.LINE_AA)
                cv2.circle(canvas, (lex - 20, ey + 22), 4, self.BRIGHT_BLUE, -1, cv2.LINE_AA)
                cv2.circle(canvas, (rex + 20, ey + 22), 8, self.GLOW_BLUE, -1, cv2.LINE_AA)
                cv2.circle(canvas, (rex + 20, ey + 22), 4, self.BRIGHT_BLUE, -1, cv2.LINE_AA)

            elif state == "listening":
                # 听取中：快乐弧形笑眼 (向上弯曲)
                cv2.ellipse(canvas, (lex, ey + 10), (35, 25), 0, 200, 340, self.GLOW_BLUE, 18, cv2.LINE_AA)
                cv2.ellipse(canvas, (lex, ey + 10), (35, 25), 0, 200, 340, self.BRIGHT_BLUE, 6, cv2.LINE_AA)
                cv2.ellipse(canvas, (rex, ey + 10), (35, 25), 0, 200, 340, self.GLOW_BLUE, 18, cv2.LINE_AA)
                cv2.ellipse(canvas, (rex, ey + 10), (35, 25), 0, 200, 340, self.BRIGHT_BLUE, 6, cv2.LINE_AA)

            elif state == "thinking":
                # 思考中：可爱斜线缩紧眼 (> <)
                # 左眼 >
                cv2.line(canvas, (lex - 20, ey - 15), (lex + 15, ey), self.GLOW_BLUE, 16, cv2.LINE_AA)
                cv2.line(canvas, (lex - 20, ey - 15), (lex + 15, ey), self.BRIGHT_BLUE, 5, cv2.LINE_AA)
                cv2.line(canvas, (lex - 20, ey + 15), (lex + 15, ey), self.GLOW_BLUE, 16, cv2.LINE_AA)
                cv2.line(canvas, (lex - 20, ey + 15), (lex + 15, ey), self.BRIGHT_BLUE, 5, cv2.LINE_AA)
                # 右眼 <
                cv2.line(canvas, (rex + 20, ey - 15), (rex - 15, ey), self.GLOW_BLUE, 16, cv2.LINE_AA)
                cv2.line(canvas, (rex + 20, ey - 15), (rex - 15, ey), self.BRIGHT_BLUE, 5, cv2.LINE_AA)
                cv2.line(canvas, (rex + 20, ey + 15), (rex - 15, ey), self.GLOW_BLUE, 16, cv2.LINE_AA)
                cv2.line(canvas, (rex + 20, ey + 15), (rex - 15, ey), self.BRIGHT_BLUE, 5, cv2.LINE_AA)

            elif state == "speaking":
                # 说话中：充满热情的桃心眼 (心跳收缩微动效)
                scale_size = int(22 + 3 * math.sin(time.time() * 12))
                self._draw_heart(canvas, lex, ey, scale_size + 4, self.GLOW_BLUE, -1)
                self._draw_heart(canvas, lex, ey, scale_size, self.BRIGHT_BLUE, -1)
                self._draw_heart(canvas, rex, ey, scale_size + 4, self.GLOW_BLUE, -1)
                self._draw_heart(canvas, rex, ey, scale_size, self.BRIGHT_BLUE, -1)

        # ==========================================
        # 👄 绘制嘴巴部分
        # ==========================================
        if state == "idle":
            # 待机：波浪猫咪嘴 (w 形状)
            cv2.ellipse(canvas, (cx - 13, my), (13, 10), 0, 0, 180, self.GLOW_BLUE, 12, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx - 13, my), (13, 10), 0, 0, 180, self.BRIGHT_BLUE, 4, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx + 13, my), (13, 10), 0, 0, 180, self.GLOW_BLUE, 12, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx + 13, my), (13, 10), 0, 0, 180, self.BRIGHT_BLUE, 4, cv2.LINE_AA)

        elif state == "listening":
            # 听取：律动的波浪示波线
            wave_width = 80
            points = []
            for x_off in range(-wave_width//2, wave_width//2, 4):
                wy = my + int(math.sin(x_off * 0.15 + time.time() * 18) * 8)
                points.append((cx + x_off, wy))
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i+1], self.GLOW_BLUE, 10, cv2.LINE_AA)
                cv2.line(canvas, points[i], points[i+1], self.BRIGHT_BLUE, 3, cv2.LINE_AA)

        elif state == "thinking":
            # 思考：加载进度感直嘴线 + 来回滑动的发光圆点
            cv2.line(canvas, (cx - 30, my), (cx + 30, my), self.GLOW_BLUE, 8, cv2.LINE_AA)
            cv2.line(canvas, (cx - 30, my), (cx + 30, my), self.BRIGHT_BLUE, 3, cv2.LINE_AA)
            
            dot_x = cx + int(24 * math.sin(time.time() * 6))
            cv2.circle(canvas, (dot_x, my), 8, self.GLOW_BLUE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (dot_x, my), 4, self.BRIGHT_BLUE, -1, cv2.LINE_AA)

        elif state == "speaking":
            # 说话：开口说话的椭圆大嘴巴 (高频收缩动效)
            talk_h = int(10 + 12 * abs(math.sin(time.time() * 16)))
            self._draw_neon_capsule(canvas, cx, my, 12, talk_h, self.GLOW_BLUE, self.BRIGHT_BLUE, 8)

        # 屏幕底部的微弱信号呼吸灯
        light_glow = int(120 + 30 * math.sin(time.time() * 3))
        cv2.circle(canvas, (cx, HEIGHT - 45), 4, (light_glow, int(light_glow*0.8), 0), -1, cv2.LINE_AA)

    # ==========================================
    # 🖼️ 各交互界面实现
    # ==========================================

    def show_prompt_frame(self, text, ok_detected=False):
        """首帧唤醒等待界面"""
        self._ensure_window()
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # 1. 绘制 3D 机器人机身外壳与大 OLED 屏幕
        self._draw_device_casing(canvas)
        self._draw_screen_grid(canvas)
        self._draw_neon_features(canvas, "idle")

        # 2. 绘制下置霓虹卡片
        self._draw_neon_card(canvas, (80, 400), (1200, 670), active=True)

        # 3. 渲染文字
        canvas = self._draw_neon_text(canvas, "系统已就绪，请确认相机正常", (130, 435), 28, active=True)
        canvas = self._draw_neon_text(canvas, f"当前步骤: {text}", (130, 500), 20, active=False)
        
        ok_status = "【已确认相机正常，正在进入...】" if ok_detected else "【等待手势中：请比 OK 确认相机正常】"
        canvas = self._draw_neon_text(canvas, ok_status, (130, 565), 22, active=ok_detected)

        self._show_canvas(canvas)
        return self.poll_key()

    def show_preview_frame(
        self,
        frame,
        ok_detected=False,
        ok_count=0,
        ok_need=None,
    ):
        """实时摄像头调焦预览界面"""
        self._ensure_window()
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # 1. 绘制 3D 机器人机身外壳与大 OLED 屏幕
        self._draw_device_casing(canvas)

        screen_x1, screen_y1 = 26, 26
        screen_x2, screen_y2 = WIDTH - 26, HEIGHT - 26
        sw = screen_x2 - screen_x1
        sh = screen_y2 - screen_y1

        if frame is None:
            self._draw_screen_grid(canvas)
            canvas = self._draw_neon_text(canvas, "正在加载摄像头画面...", (WIDTH//2 - 130, HEIGHT//2 - 20), 26, active=True)
        else:
            # 2. 将正常彩色摄像头帧缩放到 OLED 屏幕内部大小 (保留彩色画面)
            frame_resized = cv2.resize(frame, (sw, sh))
            
            # 将彩色画面嵌入屏幕显示区
            canvas[screen_y1:screen_y2, screen_x1:screen_x2] = frame_resized

            # 3. 顶部操作栏面板 (Plan B 科技风格)
            self._draw_neon_card(canvas, (80, 50), (1200, 190), active=True)
            canvas = self._draw_neon_text(canvas, " 画面调焦预览阶段 (Focusing Camera Preview)", (120, 75), 26, active=True)
            canvas = self._draw_neon_text(canvas, "请调整镜头使您的唇部清晰，随后再次对准镜头比 [OK] 进入主菜单", (120, 125), 18, active=False)

            # 4. 底部确认栏面板 (Plan B 科技风格)
            self._draw_neon_card(canvas, (80, 545), (1200, 665), active=ok_detected)
            preview_need = OK_HOLD_FRAMES if ok_need is None else max(1, int(ok_need))
            ok_text = f"检测状态: {'OK手势已触发' if ok_detected else '等待确认中'} | 稳定保持计数: {min(ok_count, preview_need)}/{preview_need}"
            canvas = self._draw_neon_text(canvas, ok_text, (120, 585), 22, active=ok_detected)

        self._show_canvas(canvas)
        return self.poll_key()

    def show_menu_frame(
        self,
        current_gesture=None,
        ok_count=0,
        g1_count=0,
        g2_count=0,
        ok_need=None,
        g1_need=None,
        g2_need=None,
    ):
        """HDMI 主选择菜单卡片界面"""
        self._ensure_window()

        ok_need = OK_HOLD_FRAMES if ok_need is None else max(1, int(ok_need))
        g1_need = OK_HOLD_FRAMES if g1_need is None else max(1, int(g1_need))
        g2_need = OK_HOLD_FRAMES if g2_need is None else max(1, int(g2_need))
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # 1. 3D 外壳与表情屏
        self._draw_device_casing(canvas)
        self._draw_screen_grid(canvas)
        self._draw_neon_features(canvas, "idle")

        # 2. 左右大功能卡片 (暗色底 + 霓虹亮蓝框)
        is_card1_act = (current_gesture == "1")
        is_card2_act = (current_gesture == "2")
        is_exit_act = (current_gesture == "ok")

        # 菜单卡片 1 (唇语翻译)
        self._draw_neon_card(canvas, (80, 400), (600, 590), active=is_card1_act)
        card1_txt_col = (100, 255, 180) if is_card1_act else (50, 200, 255)
        canvas = self._draw_text(canvas, "1. 唇语识别 (Translate)", (120, 440), 26, card1_txt_col, stroke_width=1, stroke_rgb=(0,120,220))
        canvas = self._draw_neon_text(canvas, "对摄像头比划手势 [1] 进入", (120, 495), 18, active=is_card1_act)
        if g1_count > 0:
            canvas = self._draw_text(canvas, f"正在识别: {min(g1_count, g1_need)}/{g1_need}", (120, 540), 16, (100, 255, 100))

        # 菜单卡片 2 (DeepSeek 聊天)
        self._draw_neon_card(canvas, (680, 400), (1200, 590), active=is_card2_act)
        card2_txt_col = (150, 180, 255) if is_card2_act else (50, 200, 255)
        canvas = self._draw_text(canvas, "2. 智能 AI 对话 (Chat)", (720, 440), 26, card2_txt_col, stroke_width=1, stroke_rgb=(0,120,220))
        canvas = self._draw_neon_text(canvas, "对摄像头比划手势 [2] 进入", (720, 495), 18, active=is_card2_act)
        if g2_count > 0:
            canvas = self._draw_text(canvas, f"正在识别: {min(g2_count, g2_need)}/{g2_need}", (720, 540), 16, (100, 200, 255))

        # 底部退出卡片
        self._draw_neon_card(canvas, (80, 610), (1200, 670), active=is_exit_act, is_warning=is_exit_act)
        exit_prompt = f"比划 OK 手势退出程序... 计数: {min(ok_count, ok_need)}/{ok_need}" if is_exit_act else "若要安全退出程序，请比划 [OK] 手势"
        canvas = self._draw_neon_text(canvas, exit_prompt, (110, 630), 17, active=is_exit_act, is_warning=is_exit_act)

        self._show_canvas(canvas)
        return self.poll_key()

    def show_lipreading_frame(self, text_list):
        """纯净唇语翻译展示界面"""
        self._ensure_window()
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # 1. 3D 外壳与 OLED 表情屏
        self._draw_device_casing(canvas)
        self._draw_screen_grid(canvas)
        self._draw_neon_features(canvas, "listening")

        # 2. 下置大卡片控制台
        self._draw_neon_card(canvas, (80, 400), (1200, 670), active=True)

        # 3. 渲染模式信息
        canvas = self._draw_neon_text(canvas, "● 当前模式：实时唇语翻译", (120, 430), 20, active=True)

        # 4. 渲染荧光发光翻译文本
        y_offset = 480
        for text in text_list[-3:]:
            canvas = self._draw_neon_text(canvas, f"翻译内容：{text}", (120, y_offset), 24, active=True)
            y_offset += 50

        # 5. 右下角退出提示词
        canvas = self._draw_neon_text(canvas, "【比 OK 手势返回主菜单】", (880, 648), 14, active=False)

        self._show_canvas(canvas)
        return self.poll_key()

    def show_deepseek_frame(self, messages, current_token="", is_thinking=False):
        """
        DeepSeek 流式对话界面：
        - 上方卡片固定显示“您说”的唇语识别结果；
        - 下方大卡片固定显示 DeepSeek 正在生成或已经生成的回答。
        """
        self._ensure_window()
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        self._draw_device_casing(canvas)
        self._draw_screen_grid(canvas)

        latest_user = ""
        latest_ai = ""
        for msg in reversed(messages or []):
            role = msg.get("role", "")
            if not latest_ai and role == "assistant":
                latest_ai = str(msg.get("content", "") or "")
            elif not latest_user and role == "user":
                latest_user = str(msg.get("content", "") or "")
            if latest_user and latest_ai:
                break

        display_ai = str(current_token or latest_ai or "")
        if is_thinking and not display_ai:
            dots = "." * (int(time.time() * 2) % 4)
            display_ai = "正在思考中" + dots

        canvas = self._draw_neon_text_box(
            canvas,
            "DeepSeek 智能对话",
            (220, 55),
            (840, 60),
            font_size=32,
            active=True,
            max_lines=1,
            align="center",
            valign="center",
        )

        # 上方：用户说了什么。
        self._draw_neon_card(canvas, (80, 130), (1200, 300), active=False)
        canvas = self._draw_neon_text_box(
            canvas,
            "您说",
            (115, 148),
            (180, 38),
            font_size=20,
            active=False,
            max_lines=1,
            align="left",
            valign="center",
        )
        canvas = self._draw_neon_text_box(
            canvas,
            latest_user or "未识别到有效词语",
            (115, 190),
            (1050, 82),
            font_size=27,
            active=True,
            max_lines=3,
            line_gap=6,
            align="left",
            valign="center",
        )

        # 下方：DeepSeek 回答。
        self._draw_neon_card(canvas, (80, 325), (1200, 635), active=True)
        canvas = self._draw_neon_text_box(
            canvas,
            "DeepSeek 回答",
            (115, 345),
            (300, 38),
            font_size=20,
            active=False,
            max_lines=1,
            align="left",
            valign="center",
        )
        canvas = self._draw_neon_text_box(
            canvas,
            display_ai or "等待回答……",
            (115, 390),
            (1050, 210),
            font_size=24,
            active=True,
            max_lines=7,
            line_gap=6,
            align="left",
            valign="top",
        )

        canvas = self._draw_neon_text_box(
            canvas,
            "正在生成回答，请稍候",
            (330, 650),
            (620, 34),
            font_size=16,
            active=is_thinking,
            max_lines=1,
            align="center",
            valign="center",
        )

        self._show_canvas(canvas)
        return self.poll_key()

    def show_deepseek_result_frame(
        self,
        user_text,
        answer_text,
        hint_text="再次识别比 2，退出识别模式比 OK",
    ):
        """DeepSeek 最终结果页：上方显示用户内容，下方显示回答。"""
        messages = [
            {"role": "user", "content": str(user_text or "未识别到有效词语")},
            {"role": "assistant", "content": str(answer_text or "DeepSeek 没有返回内容。")},
        ]

        self._ensure_window()
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self._draw_device_casing(canvas)
        self._draw_screen_grid(canvas)

        canvas = self._draw_neon_text_box(
            canvas,
            "DeepSeek 智能对话",
            (220, 45),
            (840, 58),
            font_size=31,
            active=True,
            max_lines=1,
            align="center",
            valign="center",
        )

        self._draw_neon_card(canvas, (80, 115), (1200, 275), active=False)
        canvas = self._draw_neon_text_box(
            canvas,
            "您说",
            (115, 132),
            (180, 36),
            font_size=19,
            active=False,
            max_lines=1,
            align="left",
            valign="center",
        )
        canvas = self._draw_neon_text_box(
            canvas,
            messages[0]["content"],
            (115, 174),
            (1050, 74),
            font_size=26,
            active=True,
            max_lines=3,
            line_gap=5,
            align="left",
            valign="center",
        )

        self._draw_neon_card(canvas, (80, 300), (1200, 615), active=True)
        canvas = self._draw_neon_text_box(
            canvas,
            "DeepSeek 回答",
            (115, 320),
            (300, 36),
            font_size=19,
            active=False,
            max_lines=1,
            align="left",
            valign="center",
        )
        canvas = self._draw_neon_text_box(
            canvas,
            messages[1]["content"],
            (115, 363),
            (1050, 220),
            font_size=24,
            active=True,
            max_lines=7,
            line_gap=6,
            align="left",
            valign="top",
        )

        canvas = self._draw_neon_text_box(
            canvas,
            hint_text,
            (170, 640),
            (940, 42),
            font_size=18,
            active=False,
            max_lines=2,
            line_gap=3,
            align="center",
            valign="center",
        )

        self._show_canvas(canvas)
        return self.poll_key()


    def show_camera_instruction_frame(
        self,
        frame,
        line1,
        line2,
        ok_detected=False,
        ok_count=0,
        ok_need=None,
        gesture_name="OK",
    ):
        """
        实时镜头提示界面：
        - 用于选择 1 / 2 后的镜头调整阶段；
        - 也用于唇语录制阶段；
        - 两行文字固定在下方聊天框内，避免挡住嘴部画面。
        """
        self._ensure_window()
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        self._draw_device_casing(canvas)

        screen_x1, screen_y1 = 26, 26
        screen_x2, screen_y2 = WIDTH - 26, HEIGHT - 26
        sw = screen_x2 - screen_x1
        sh = screen_y2 - screen_y1

        if frame is None:
            self._draw_screen_grid(canvas)
            canvas = self._draw_neon_text_box(
                canvas,
                "正在加载摄像头画面……",
                (180, 260),
                (900, 80),
                font_size=26,
                active=True,
                max_lines=1,
                align="center",
                valign="center",
            )
        else:
            frame_resized = cv2.resize(frame, (sw, sh))
            canvas[screen_y1:screen_y2, screen_x1:screen_x2] = frame_resized

            # 压一层暗色半透明遮罩，让底部提示更清楚
            overlay = canvas.copy()
            self._draw_neon_card(overlay, (80, 500), (1200, 670), active=ok_detected)
            cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

        status = f"{line1}\n{line2}"
        canvas = self._draw_neon_text_box(
            canvas,
            status,
            (130, 525),
            (1010, 88),
            font_size=26,
            active=True,
            max_lines=2,
            line_gap=8,
            align="center",
            valign="center",
        )

        gesture_need = OK_HOLD_FRAMES if ok_need is None else max(1, int(ok_need))
        gesture_count = min(ok_count, gesture_need)
        gesture_label = str(gesture_name or "OK")
        counter_text = f"{gesture_label} 稳定计数：{gesture_count}/{gesture_need}" if ok_detected else f"等待 {gesture_label} 手势确认"
        canvas = self._draw_neon_text_box(
            canvas,
            counter_text,
            (130, 620),
            (1010, 34),
            font_size=16,
            active=ok_detected,
            max_lines=1,
            align="center",
            valign="center",
        )

        self._show_canvas(canvas)
        return self.poll_key()


    def _draw_camera_background(self, frame):
        """
        把摄像头画面铺满 HDMI 屏幕，并叠一层暗色蒙版，方便文字可读。
        """
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        if frame is None:
            self._draw_device_casing(canvas)
            self._draw_screen_grid(canvas)
            return canvas

        try:
            frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
            canvas[:] = frame_resized
        except Exception:
            self._draw_device_casing(canvas)
            self._draw_screen_grid(canvas)
            return canvas

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        return canvas

    def show_camera_adjust_frame(self, frame, line1="开始调整镜头", line2="调整好比OK", ok_detected=False, ok_count=0):
        """
        选择 1/2 后进入的实时调整镜头界面。
        按 OK 退出该实时界面。
        """
        self._ensure_window()

        canvas = self._draw_camera_background(frame)

        self._draw_neon_card(canvas, (90, 430), (1190, 665), active=ok_detected)

        canvas = self._draw_neon_text_box(
            canvas,
            f"{line1}\n{line2}",
            (150, 455),
            (980, 120),
            font_size=34,
            active=True,
            max_lines=2,
            line_gap=12,
            align="center",
            valign="center",
        )

        status = f"OK 检测：{ok_count}/{OK_HOLD_FRAMES}" if ok_detected else "请调整唇部到画面中央，清晰后比 OK"
        canvas = self._draw_neon_text_box(
            canvas,
            status,
            (150, 590),
            (980, 42),
            font_size=18,
            active=ok_detected,
            max_lines=1,
            line_gap=4,
            align="center",
            valign="center",
        )

        self._show_canvas(canvas)
        return self.poll_key()

    def show_camera_record_frame(self, frame, is_recording=False, current_gesture=None, ok_count=0, g1_count=0, mouth_crop=None):
        """
        唇语识别模式里的实时录制界面。
        未录制：比 1 后开始录制
        录制中：结束录制按 OK
        """
        self._ensure_window()

        canvas = self._draw_camera_background(frame)

        # 左上角嘴部裁剪实时小窗：只显示，不保存视频
        if mouth_crop is not None:
            try:
                preview_w, preview_h = 260, 130
                mouth_show = cv2.resize(mouth_crop, (preview_w, preview_h))

                x1, y1 = 48, 48
                x2, y2 = x1 + preview_w, y1 + preview_h

                # 小窗背景和边框
                self._draw_neon_card(canvas, (x1 - 12, y1 - 34), (x2 + 12, y2 + 12), active=True, radius=14)

                # 贴入嘴部 ROI
                canvas[y1:y2, x1:x2] = mouth_show

                # 标题
                canvas = self._draw_neon_text_box(
                    canvas,
                    "嘴部实时裁剪",
                    (x1 - 2, y1 - 30),
                    (preview_w + 4, 24),
                    font_size=14,
                    active=True,
                    max_lines=1,
                    align="center",
                    valign="center",
                )

                # 细边框
                cv2.rectangle(canvas, (x1, y1), (x2, y2), self.BRIGHT_BLUE, 2, cv2.LINE_AA)
            except Exception as e:
                print(f"[DISPLAY][WARN] 嘴部裁剪小窗显示失败：{e}")

        self._draw_neon_card(canvas, (90, 430), (1190, 665), active=True)

        if is_recording:
            title = "正在录制唇部视频"
            subtitle = "结束录制按OK"
            state_line = f"录制中 | OK计数：{ok_count}/{OK_HOLD_FRAMES}"
        else:
            title = "比1后开始录制"
            subtitle = "结束录制按OK"
            state_line = f"等待手势1 | 计数：{g1_count}/{OK_HOLD_FRAMES}"

        canvas = self._draw_neon_text_box(
            canvas,
            f"{title}\n{subtitle}",
            (150, 455),
            (980, 120),
            font_size=34,
            active=True,
            max_lines=2,
            line_gap=12,
            align="center",
            valign="center",
        )

        canvas = self._draw_neon_text_box(
            canvas,
            state_line,
            (150, 590),
            (980, 42),
            font_size=18,
            active=True,
            max_lines=1,
            line_gap=4,
            align="center",
            valign="center",
        )

        self._show_canvas(canvas)
        return self.poll_key()


    def show_lip_result_frame(self, recognized_text, ok_detected=False, ok_count=0):
        """
        唇语识别完成后，在底部文字框中显示识别句子。
        """
        self._ensure_window()

        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self._draw_device_casing(canvas)
        self._draw_screen_grid(canvas)
        self._draw_neon_features(canvas, "speaking")

        self._draw_neon_card(canvas, (80, 470), (1200, 670), active=True)

        canvas = self._draw_neon_text_box(
            canvas,
            "识别结果",
            (120, 490),
            (1040, 36),
            font_size=20,
            active=False,
            max_lines=1,
            line_gap=4,
            align="left",
            valign="center",
        )

        canvas = self._draw_neon_text_box(
            canvas,
            recognized_text,
            (120, 532),
            (1040, 88),
            font_size=24,
            active=True,
            max_lines=3,
            line_gap=6,
            align="center",
            valign="center",
        )

        prompt = f"比 OK 返回主菜单  {ok_count}/{OK_HOLD_FRAMES}" if ok_detected else "比 OK 返回主菜单"
        canvas = self._draw_neon_text_box(
            canvas,
            prompt,
            (120, 630),
            (1040, 30),
            font_size=15,
            active=ok_detected,
            max_lines=1,
            line_gap=2,
            align="center",
            valign="center",
        )

        self._show_canvas(canvas)
        return self.poll_key()


    def show_action_result_frame(self, title, result_text, hint_text="再次识别比 2，退出比 OK"):
        """
        识别/DeepSeek结果页：
        - 中间显示 title；
        - 下方框显示 result_text；
        - result_text 上方提示：再次识别比2，退出比OK。
        """
        self._ensure_window()
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        self._draw_device_casing(canvas)
        self._draw_screen_grid(canvas)
        self._draw_neon_features(canvas, "speaking")

        canvas = self._draw_neon_text_box(
            canvas,
            title,
            (220, 285),
            (840, 70),
            font_size=34,
            active=True,
            max_lines=1,
            align="center",
            valign="center",
        )

        self._draw_neon_card(canvas, (80, 470), (1200, 680), active=True)

        canvas = self._draw_neon_text_box(
            canvas,
            hint_text,
            (120, 492),
            (1080, 42),
            font_size=20,
            active=False,
            max_lines=2,
            line_gap=3,
            align="center",
            valign="center",
        )

        canvas = self._draw_neon_text_box(
            canvas,
            result_text,
            (120, 540),
            (1080, 106),
            font_size=24,
            active=True,
            max_lines=4,
            line_gap=5,
            align="center",
            valign="center",
        )

        self._show_canvas(canvas)
        return self.poll_key()


    def show_recognizing_frame(self, status_text="正在识别", result_text=""):
        """
        最终版本：
        支持：
        - show_recognizing_frame()
        - show_recognizing_frame("正在识别", "")
        - show_recognizing_frame("DeepSeek 思考中", "识别到：xxx")
        """
        self._ensure_window()
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        self._draw_device_casing(canvas)
        self._draw_screen_grid(canvas)

        if result_text:
            self._draw_neon_features(canvas, "speaking")
            center_text = status_text
        else:
            self._draw_neon_features(canvas, "thinking")
            dots = "." * (int(time.time() * 2) % 4)
            center_text = f"{status_text}{dots}"

        canvas = self._draw_neon_text_box(
            canvas,
            center_text,
            (250, 318),
            (780, 70),
            font_size=34,
            active=True,
            max_lines=1,
            align="center",
            valign="center",
        )

        self._draw_neon_card(canvas, (80, 500), (1200, 670), active=True)

        if result_text:
            canvas = self._draw_neon_text_box(
                canvas,
                result_text,
                (120, 528),
                (1080, 92),
                font_size=24,
                active=True,
                max_lines=3,
                line_gap=5,
                align="center",
                valign="center",
            )
        else:
            canvas = self._draw_neon_text_box(
                canvas,
                "请稍候，系统正在分析刚才录制的唇部视频",
                (120, 560),
                (1080, 60),
                font_size=22,
                active=False,
                max_lines=2,
                align="center",
                valign="center",
            )

        self._show_canvas(canvas)
        return self.poll_key()

    def poll_key(self):
        if not self.window_created:
            return -1
        return cv2.waitKey(1) & 0xFF

    def hide(self):
        if self.window_created:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except Exception:
                pass
            self.window_created = False
            self._last_fullscreen_refresh = 0.0

    def close(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.window_created = False
        self._last_fullscreen_refresh = 0.0