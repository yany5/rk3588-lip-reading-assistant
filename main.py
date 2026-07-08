# -*- coding: utf-8 -*-
import os
import sys
import time
import queue
import datetime
import threading
import traceback

import cv2

from camera_service import CameraService
from display_service import DisplayService
from lip_model_bridge import next_video_path, run_lip_model, TARGET_FPS
from env_llm_context import append_environment_context
from llm_service import LLMService
from direct_hand_gesture import DirectHandGestureDetector


# =========================
# Direct gesture detector
# =========================

class GestureState:
    def __init__(self, need=3):
        self.need = need
        self.name = ""
        self.count = 0

    def update(self, name):
        name = "" if name is None else str(name)
        if name:
            if name == self.name:
                self.count += 1
            else:
                self.name = name
                self.count = 1
            if self.count >= self.need:
                return name
        else:
            self.name = ""
            self.count = 0
        return ""


class DirectGesture:
    def __init__(self):
        self.detector = DirectHandGestureDetector()
        self.states = {}

    def close(self):
        self.detector.close()

    def reset(self, key=None):
        if key is None:
            self.states.clear()
        else:
            self.states.pop(key, None)

    def raw(self, frame):
        try:
            name, debug = self.detector.detect(frame)
            return name or "", debug or {}
        except Exception as e:
            print("[GESTURE][ERROR]", repr(e))
            return "", {"error": str(e)}

    @staticmethod
    def ok_like(name, debug):
        """
        主菜单/结果页/结束录制阶段：OK 优先。
        只要拇指和食指明显捏合，就强制解释成 OK，
        防止 OK 被误判成 2 后进入 DeepSeek。
        """
        if name == "ok":
            return True
        try:
            d = float(debug.get("thumb_index_d", 999.0))
            box = float(debug.get("box_size", 1.0))
            threshold = max(0.075, 0.360 * box)
            return d < threshold
        except Exception:
            return False

    def context(self, frame, mode):
        """
        mode:
        - menu: 主菜单，OK 保持主菜单，1/2 进入对应模式
        - result: 结果页，OK 返回主菜单，2 重复
        - ok_only: 只接受 OK
        - start_1: 只接受 1，OK 不触发
        - start_2: 只接受 2，OK 不触发
        """
        name, debug = self.raw(frame)
        is_ok = self.ok_like(name, debug)

        if mode == "menu":
            if is_ok:
                return "ok"
            if name == "1":
                return "1"
            if name == "2":
                return "2"
            return ""

        if mode == "result":
            if is_ok:
                return "ok"
            if name == "2":
                return "2"
            return ""

        if mode == "ok_only":
            return "ok" if is_ok else ""

        if mode == "start_1":
            if is_ok:
                return ""
            return "1" if name == "1" else ""

        if mode == "start_2":
            if is_ok:
                return ""
            return "2" if name == "2" else ""

        return ""

    def stable(self, name, key, need=3):
        if key not in self.states:
            self.states[key] = GestureState(need=need)
        self.states[key].need = need
        return self.states[key].update(name)


# =========================
# LLM one-shot
# =========================

def build_deepseek_question_from_lip_word(word):
    word = "" if word is None else str(word)
    # 1. 移除了 "天气" 和 "气温"
    temperature_keys = [
        "温度", "湿度", "中央气象台", "多少度", "几度", "冷", "热", "环境", "舒适"
    ]
    time_keys = [
        "时间", "北京时间", "几点", "现在", "今天", "上午", "下午", "中午", "晚上"
    ]
    # 2. 单独建立天气问句分支，确保大模型识别到“天气”意图
    if "天气" in word or "气温" in word:
        return "请查询并回答当前天气情况以及温度和湿度。"
    if any(k in word for k in temperature_keys):
        return "请回答当前环境的温度和湿度。必须基于开发板温湿度传感器的实时数据回答，不允许编造。"

    if any(k in word for k in time_keys):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 强制命令直接说出当前时间，压缩其思考空间
        return "当前系统时间是 " + now + "。请直接用中文说出现在是几点几分，不要有任何多余的废话。"

    return "用户通过唇语识别得到的词语是：" + word + "。请根据这个词语给出一句简短中文回复。"


# 2. 优化 DeepSeek 思考阶段：传入 display 和 word，在循环中刷新动画
def call_deepseek_once(display, question, stop_event, word): # <-- 新增传入 display 和 word
    llm = LLMService()
    messages = [
        {"role": "system", "content": "你是板端语音/唇语助手。回答必须自然。"},
        {"role": "user", "content": append_environment_context(question)}
    ]
    
    q, worker = llm.start_stream_chat(messages, stop_event)
    out = ""
    while not stop_event.is_set():
        # 在等待网络回传的循环中，每 50ms 刷新一次“思考中”的表情动画
        display.show_recognizing_frame("DeepSeek 思考中", "识别到：" + word)
        try:
            item = q.get(timeout=0.05)
        except queue.Empty:
            continue
        typ = item[0]
        if typ == "token":
            out += item[1]
        elif typ == "error":
            return "DeepSeek 调用失败：" + str(item[1])
        elif typ == "done":
            break
    return out.strip() or "DeepSeek 没有返回内容。"

# =========================
# UI / flow helpers
# =========================

def wait_live_gesture(display, camera, dg, stop_event, line1, line2, mode, accept, key_name, need=3):
    """
    显示实时镜头，等待指定上下文手势。
    """
    dg.reset(key_name)
    time.sleep(0.15)

    while not stop_event.is_set():
        frame = camera.get_frame()
        name = dg.context(frame, mode)
        stable = dg.stable(name, key_name, need=need)

        gesture_name = "OK" if "ok" in accept else str(accept[0])

        key = display.show_camera_instruction_frame(
            frame,
            line1,
            line2,
            ok_detected=(name in accept),
            ok_count=dg.states.get(key_name, GestureState()).count,
            ok_need=need,
            gesture_name=gesture_name,
        )

        if key == ord("q"):
            stop_event.set()
            return None

        # 键盘调试
        if key == ord("o") and "ok" in accept:
            return "ok"
        if key == ord("1") and "1" in accept:
            return "1"
        if key == ord("2") and "2" in accept:
            return "2"

        if stable in accept:
            print(f"[FLOW] {key_name}: confirmed {stable}")
            return stable

        time.sleep(0.015)

    return None


def pre_record_delay(display, camera, stop_event, seconds=0.8):
    """
    比 1/2 后先让用户放下手，避免手挡嘴导致 FaceMesh 嘴部检测失败。
    """
    start = time.time()
    while not stop_event.is_set() and time.time() - start < seconds:
        frame = camera.get_frame()
        display.show_camera_instruction_frame(
            frame,
            "请放下手，嘴部对准镜头",
            "马上开始录制",
            ok_detected=True,
            ok_count=int((time.time() - start) * 10),
            ok_need=max(1, int(round(seconds * 10))),
            gesture_name="准备",
        )
        time.sleep(0.03)


def record_until_ok(display, camera, dg, stop_event, video_path):
    """
    真正开始录制 videoXXXX.mp4，只接受 OK 结束。
    """
    dg.reset("record_until_ok")
    time.sleep(0.15)

    writer = None
    record_start = None
    video_path = str(video_path)

    try:
        while not stop_event.is_set():
            frame = camera.get_frame()

            if frame is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(video_path, fourcc, float(TARGET_FPS), (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("无法创建录制视频：" + video_path)
                    record_start = time.time()
                    print(f"[REC] start {video_path}, fps={TARGET_FPS}, size={w}x{h}")

                writer.write(frame)

            name = dg.context(frame, "ok_only")
            stable = dg.stable(name, "record_until_ok", need=10)

            key = display.show_camera_instruction_frame(
                frame,
                "正在录制",
                "结束录制按 OK",
                ok_detected=(name == "ok"),
                ok_count=dg.states.get("record_until_ok", GestureState()).count,
                ok_need=10,
                gesture_name="OK",
            )

            if key == ord("q"):
                stop_event.set()
                return None

            if key == ord("o"):
                print("[REC] keyboard OK stop")
                return video_path

            if stable == "ok":
                elapsed = 0.0 if record_start is None else time.time() - record_start
                # 至少约 40 帧：25fps * 1.6s
                if elapsed < 1.6:
                    time.sleep(0.015)
                    continue
                print("[REC] gesture OK stop")
                return video_path

            time.sleep(0.015)

        return None

    finally:
        if writer is not None:
            writer.release()
            print("[REC] stop", video_path)


# 1. 优化正在识别阶段：采用多线程刷新 UI
def recognize_with_screen(display, stop_event, video_path):
    import threading
    result_container = {}
    err_container = []
    def worker():
        try:
            res = run_lip_model(video_path=video_path, topk=5)
            result_container["result"] = res
        except Exception as e:
            err_container.append(e)
    # 开辟后台线程执行推理
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    # 主线程负责持续渲染 UI，圆点和省略号动画将保持流畅
    while t.is_alive() and not stop_event.is_set():
        display.show_recognizing_frame("正在识别", "")
        time.sleep(0.03)
    if err_container:
        raise err_container[0]
    return result_container.get("result")


def record_and_recognize_once(display, camera, dg, stop_event, start_gesture):
    """
    start_gesture 是 "1" 或 "2"。
    """
    # 调整镜头 -> OK
    g = wait_live_gesture(
        display,
        camera,
        dg,
        stop_event,
        "开始调整镜头使得脸在屏幕中央",
        "调整好比 OK",
        mode="ok_only",
        accept=("ok",),
        key_name=f"adjust_{start_gesture}",
        need=10,
    )
    if g != "ok" or stop_event.is_set():
        return None

    # 等 1/2 开始录制
    g = wait_live_gesture(
        display,
        camera,
        dg,
        stop_event,
        f"比 {start_gesture} 后开始录制",
        "结束录制按 OK",
        mode=f"start_{start_gesture}",
        accept=(start_gesture,),
        key_name=f"start_{start_gesture}",
        need=10,
    )
    if g != start_gesture or stop_event.is_set():
        return None

    # 放下手再录
    pre_record_delay(display, camera, stop_event, seconds=0.8)

    # 录制 -> OK 结束
    video_path = next_video_path()
    saved = record_until_ok(display, camera, dg, stop_event, video_path)
    if not saved or stop_event.is_set():
        return None

    # 识别
    return recognize_with_screen(display, stop_event, saved)


def wait_repeat_or_menu(display, camera, dg, stop_event, title, result_text):
    """
    结果页：
    2：重复当前模式
    OK：返回主菜单
    """
    dg.reset("result")
    time.sleep(0.2)

    while not stop_event.is_set():
        frame = camera.get_frame()
        name = dg.context(frame, "result")
        stable = dg.stable(name, "result", need=10)

        key = display.show_action_result_frame(
            title,
            result_text,
            "如果要再次识别比 2，退出识别模式比 OK",
        )

        if key == ord("q"):
            stop_event.set()
            return "menu"
        if key == ord("2"):
            return "repeat"
        if key == ord("o"):
            return "menu"

        if stable == "2":
            print("[RESULT] repeat")
            return "repeat"

        if stable == "ok":
            print("[RESULT] menu")
            return "menu"

        time.sleep(0.015)

    return "menu"


def lipreading_mode(display, camera, dg, stop_event):
    """
    选择 1。
    """
    print("========== MODE 1: Lip Reading ==========")

    while not stop_event.is_set():
        try:
            result = record_and_recognize_once(display, camera, dg, stop_event, "1")
            if result is None:
                return

            word = result.get("word") or "未识别到有效词语"
            text = word
            

        except Exception as e:
            traceback.print_exc()
            text = "识别失败：" + str(e).split("\n")[-1]

        action = wait_repeat_or_menu(display, camera, dg, stop_event, "识别完成", text)
        if action != "repeat":
            return


def deepseek_mode(display, camera, dg, stop_event):
    """
    选择 2。
    """
    print("========== MODE 2: Lip + DeepSeek ==========")

    while not stop_event.is_set():
        try:
            result = record_and_recognize_once(display, camera, dg, stop_event, "2")
            if result is None:
                return

            word = result.get("word") or "未识别到有效词语"

            display.show_recognizing_frame("DeepSeek 思考中", "识别到：" + word)

            question = build_deepseek_question_from_lip_word(word)
            answer = call_deepseek_once(display, question, stop_event, word)

            text = "识别：" + word + "\n" + answer

        except Exception as e:
            traceback.print_exc()
            text = "处理失败：" + str(e).split("\n")[-1]

        action = wait_repeat_or_menu(display, camera, dg, stop_event, "DeepSeek 输出", text)
        if action != "repeat":
            return


def wait_initial_ok(display, camera, dg, stop_event):
    """
    初始界面：OK 进入主菜单。
    """
    return wait_live_gesture(
        display,
        camera,
        dg,
        stop_event,
        "比 OK 开始",
        "进入唇语识别助手",
        mode="ok_only",
        accept=("ok",),
        key_name="initial_ok",
        need=10,
    ) == "ok"


def main_menu_loop(display, camera, dg, stop_event):
    """
    主菜单最终逻辑：

    OK：保持主菜单，不进入任何模式；
    1 ：进入唇语识别；
    2 ：进入 DeepSeek 模式。

    注意：
    - 主菜单不读取旧 gesture_service 队列；
    - OK 优先级最高；
    - 2 要连续 10 帧，避免 OK/1 误触发成 2。
    """
    dg.reset("main_menu")
    menu_ok_ignore_until = 0.0

    while not stop_event.is_set():
        frame = camera.get_frame()
        name = dg.context(frame, "menu")

        # 主菜单稳定帧数：
        # OK：3帧，保持主菜单；
        # 1 ：4帧，进入唇语识别；
        # 2 ：10帧，进入 DeepSeek，防止误触发。
        if name == "2":
            stable = dg.stable(name, "main_menu", need=10)
        elif name == "1":
            stable = dg.stable(name, "main_menu", need=4)
        else:
            stable = dg.stable(name, "main_menu", need=3)

        state = dg.states.get("main_menu", GestureState())

        key = display.show_menu_frame(
            current_gesture=name,
            ok_count=state.count if name == "ok" else 0,
            g1_count=state.count if name == "1" else 0,
            g2_count=state.count if name == "2" else 0,
            ok_need=3,
            g1_need=4,
            g2_need=10,
        )

        if key == ord("q"):
            stop_event.set()
            return

        # 键盘调试
        if key == ord("1"):
            stable = "1"
        elif key == ord("2"):
            stable = "2"
        elif key == ord("o"):
            stable = "ok"

        if stable == "ok":
            now = time.monotonic()
            if now >= menu_ok_ignore_until:
                print("[MENU] OK: stay menu")
                menu_ok_ignore_until = now + 0.8

            dg.reset("main_menu")
            time.sleep(0.10)
            continue

        if stable == "1":
            print("[MENU] enter mode 1")
            dg.reset()
            lipreading_mode(display, camera, dg, stop_event)
            dg.reset("main_menu")
            continue

        if stable == "2":
            print("[MENU] enter mode 2")
            dg.reset()
            deepseek_mode(display, camera, dg, stop_event)
            dg.reset("main_menu")
            continue

        time.sleep(0.015)


def main():
    stop_event = threading.Event()
    camera = None
    display = None
    dg = None

    try:
        camera = CameraService(stop_event)
        camera.start()

        display = DisplayService()
        dg = DirectGesture()

        print("========== INITIAL OK ==========")
        if not wait_initial_ok(display, camera, dg, stop_event):
            return

        print("========== MAIN MENU ==========")
        main_menu_loop(display, camera, dg, stop_event)

    except KeyboardInterrupt:
        print("[CTRL-C] exit")
        stop_event.set()

    except Exception:
        traceback.print_exc()
        stop_event.set()

    finally:
        print("[EXIT] releasing resources")
        stop_event.set()

        if dg is not None:
            dg.close()

        if camera is not None:
            camera.stop()

        if display is not None:
            display.close()

        print("[EXIT] done")


if __name__ == "__main__":
    main()
