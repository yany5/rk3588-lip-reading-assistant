import os
import json
import glob
import queue
import threading
import subprocess
import requests

from config import (
    REMOTE_LLM_ENABLED,
    REMOTE_LLM_TYPE,
    REMOTE_LLM_BASE_URL,
    REMOTE_LLM_MODEL,
    REMOTE_LLM_API_KEY,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL,
    LOCAL_NPU_REQUIRED,
    LLM_NUM_PREDICT,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_HEALTH_TIMEOUT,
)


class StreamCleaner:
    """
    清理 DeepSeek-R1/RKLLAMA 的流式输出：
    1. 去掉 <think>...</think> 思考段
    2. 过滤 .setColumn 这类异常 token
    3. 遇到明显无限重复时截断
    """

    def __init__(self):
        self.in_think = False
        self.buffer = ""
        self.recent_tokens = []
        self.output_char_count = 0

    def clean_token(self, token):
        if token is None:
            return ""

        token = str(token)
        if token == "":
            return ""

        # 明显异常 token，直接丢掉
        if ".setColumn" in token:
            return "__STOP_BAD_REPEAT__"

        # 将新收到的 token 追加到内部缓冲区
        self.buffer += token
        out = ""

        # 循环解析缓冲区中的完整标签
        while self.buffer:
            if not self.in_think:
                # 检查是否匹配进入 think 标签
                if "<think>".startswith(self.buffer):
                    # 缓冲区是 "<think>" 的一部分，但还不完整，等待下一个 token
                    if self.buffer == "<think>":
                        self.in_think = True
                        self.buffer = ""
                    break
                elif "<think>" in self.buffer:
                    idx = self.buffer.find("<think>")
                    out += self.buffer[:idx]
                    self.in_think = True
                    self.buffer = self.buffer[idx + len("<think>"):]
                    continue
                else:
                    # 没有匹配到任何 <think> 前缀，安全输出全部缓冲区
                    out += self.buffer
                    self.buffer = ""
            else:
                # 处于 think 阶段，寻找闭合标签 </think>
                if "</think>".startswith(self.buffer):
                    if self.buffer == "</think>":
                        self.in_think = False
                        self.buffer = ""
                    break
                elif "</think>" in self.buffer:
                    idx = self.buffer.find("</think>")
                    self.in_think = False
                    self.buffer = self.buffer[idx + len("</think>"):]
                    continue
                else:
                    # 依然在思考中，吞掉缓冲区内容，不输出到屏幕
                    self.buffer = ""

        # 防止连续感叹号刷屏
        if out.strip() == "!":
            self.recent_tokens.append("!")
        elif out.strip():
            self.recent_tokens.append(out.strip())

        self.recent_tokens = self.recent_tokens[-20:]

        if len(self.recent_tokens) >= 12 and all(x == "!" for x in self.recent_tokens[-12:]):
            return "__STOP_BAD_REPEAT__"

        self.output_char_count += len(out)

        # 给桌面助手用，限制最大输出字数
        if self.output_char_count > 260:
            return "__STOP_TOO_LONG__"

        return out


class LLMService:
    """
    DeepSeek 接口路由器：
    1. 有网 / 服务器可达：优先调用服务器 DeepSeek
    2. 没网 / 服务器不可达：自动调用本地 RKLLAMA + RK3588 NPU DeepSeek
    3. 输出统一为流式 token，main.py 不需要知道后端是谁
    """

    def __init__(self):
        self.backend = None

    def _run_text(self, cmd):
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return r.stdout.strip()
        except Exception:
            return ""

    def _headers_openai(self):
        headers = {"Content-Type": "application/json"}

        if REMOTE_LLM_API_KEY and REMOTE_LLM_API_KEY.upper() not in ["EMPTY", "NONE", "NULL"]:
            headers["Authorization"] = f"Bearer {REMOTE_LLM_API_KEY}"

        return headers

    def _normalize_base(self, url):
        return url.rstrip("/")

    def _detect_rknpu_driver(self):
        evidence = []

        dev_nodes = glob.glob("/dev/rknpu*") + glob.glob("/dev/rknn*")
        if dev_nodes:
            evidence.append("device_nodes=" + ",".join(dev_nodes))

        dri_nodes = glob.glob("/dev/dri/renderD*")
        if dri_nodes:
            evidence.append("dri_render_nodes=" + ",".join(dri_nodes))

        if os.path.exists("/sys/devices/platform/fdab0000.npu"):
            evidence.append("sysfs=/sys/devices/platform/fdab0000.npu")

        if os.path.exists("/sys/class/devfreq/fdab0000.npu"):
            evidence.append("devfreq=/sys/class/devfreq/fdab0000.npu")

        dmesg_tail = self._run_text(
            "dmesg | grep -Ei 'Initialized rknpu|RKNPU fdab0000.npu|rknpu 0.9' | tail -20"
        )

        if "Initialized rknpu" in dmesg_tail or "RKNPU fdab0000.npu" in dmesg_tail:
            evidence.append("dmesg_has_rknpu")

        return evidence

    def _check_remote_openai(self):
        if not REMOTE_LLM_BASE_URL:
            return False, "REMOTE_LLM_BASE_URL 为空"

        base = self._normalize_base(REMOTE_LLM_BASE_URL)
        url = f"{base}/models"

        try:
            resp = requests.get(
                url,
                headers=self._headers_openai(),
                timeout=LLM_HEALTH_TIMEOUT,
            )

            if resp.status_code < 400:
                return True, f"OpenAI-compatible server ok: {url}"

            return False, f"{url} status={resp.status_code}"

        except Exception as e:
            return False, str(e)

    def _check_remote_ollama(self):
        if not REMOTE_LLM_BASE_URL:
            return False, "REMOTE_LLM_BASE_URL 为空"

        base = self._normalize_base(REMOTE_LLM_BASE_URL)
        url = f"{base}/api/tags"

        try:
            resp = requests.get(url, timeout=LLM_HEALTH_TIMEOUT)

            if resp.status_code < 400:
                return True, f"Ollama-compatible server ok: {url}"

            return False, f"{url} status={resp.status_code}"

        except Exception as e:
            return False, str(e)

    def _check_local_rkllama(self):
        base = self._normalize_base(LOCAL_LLM_BASE_URL)

        if LOCAL_NPU_REQUIRED:
            if ":11434" in base:
                return False, "LOCAL_LLM_BASE_URL 是 11434，疑似 Ollama CPU，不允许作为 NPU 兜底"

            evidence = self._detect_rknpu_driver()

            if not evidence:
                return False, "没有检测到 RKNPU 驱动证据"

        url = f"{base}/api/tags"

        try:
            resp = requests.get(url, timeout=LLM_HEALTH_TIMEOUT)

            if resp.status_code < 400:
                return True, f"Local RKLLAMA ok: {url}"

            return False, f"{url} status={resp.status_code}"

        except Exception as e:
            return False, str(e)

    def _select_backend(self):
        print("\n========== LLM 后端选择 ==========")

        if REMOTE_LLM_ENABLED and REMOTE_LLM_BASE_URL:
            if REMOTE_LLM_TYPE == "openai":
                ok, msg = self._check_remote_openai()
                print("[REMOTE/openai]", msg)

                if ok:
                    backend = {
                        "kind": "remote_openai",
                        "base": self._normalize_base(REMOTE_LLM_BASE_URL),
                        "model": REMOTE_LLM_MODEL,
                    }
                    print(f"[LLM] 使用在线服务器 DeepSeek：{backend}")
                    print("=================================\n")
                    return backend

            elif REMOTE_LLM_TYPE == "ollama":
                ok, msg = self._check_remote_ollama()
                print("[REMOTE/ollama]", msg)

                if ok:
                    backend = {
                        "kind": "remote_ollama",
                        "base": self._normalize_base(REMOTE_LLM_BASE_URL),
                        "model": REMOTE_LLM_MODEL,
                    }
                    print(f"[LLM] 使用在线 Ollama-compatible DeepSeek：{backend}")
                    print("=================================\n")
                    return backend
            else:
                print(f"[REMOTE] 未知 REMOTE_LLM_TYPE={REMOTE_LLM_TYPE}，跳过远程")
        else:
            print("[REMOTE] 未配置远程服务器，跳过在线模式")

        ok, msg = self._check_local_rkllama()
        print("[LOCAL/rkllama]", msg)

        if ok:
            backend = {
                "kind": "local_rkllama",
                "base": self._normalize_base(LOCAL_LLM_BASE_URL),
                "model": LOCAL_LLM_MODEL,
            }
            print(f"[LLM] 使用本地离线 RKLLAMA/NPU DeepSeek：{backend}")
            print("=================================\n")
            return backend

        print("=================================\n")

        raise RuntimeError(
            "没有可用的 DeepSeek 后端：远程服务器不可达，本地 RKLLAMA/NPU 服务也不可达。"
        )

    def require_npu_ready(self):
        self.backend = self._select_backend()

    def start_stream_chat(self, messages, stop_event):
        self.backend = self._select_backend()

        out_queue = queue.Queue()

        worker = threading.Thread(
            target=self._stream_worker,
            args=(self.backend, messages, out_queue, stop_event),
            daemon=True,
        )
        worker.start()

        return out_queue, worker

    def _stream_worker(self, backend, messages, out_queue, stop_event):
        kind = backend["kind"]

        if kind == "remote_openai":
            self._stream_openai_worker(backend, messages, out_queue, stop_event)
        elif kind in ["remote_ollama", "local_rkllama"]:
            self._stream_ollama_style_worker(backend, messages, out_queue, stop_event)
        else:
            out_queue.put(("error", f"未知 LLM backend: {kind}"))

    def _stream_openai_worker(self, backend, messages, out_queue, stop_event):
        url = f'{backend["base"]}/chat/completions'

        payload = {
            "model": backend["model"],
            "messages": messages,
            "stream": True,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "max_tokens": LLM_NUM_PREDICT,
        }

        cleaner = StreamCleaner()

        try:
            with requests.post(
                url,
                headers=self._headers_openai(),
                json=payload,
                stream=True,
                timeout=(5, 300),
            ) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if stop_event.is_set():
                        out_queue.put(("stopped", ""))
                        return

                    if not raw_line:
                        continue

                    line = raw_line.strip()

                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()

                    if line == "[DONE]":
                        out_queue.put(("done", ""))
                        return

                    try:
                        data = json.loads(line)
                    except Exception:
                        continue

                    choices = data.get("choices", [])

                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "") or ""

                    cleaned = cleaner.clean_token(token)

                    if cleaned.startswith("__STOP_"):
                        out_queue.put(("done", ""))
                        return

                    if cleaned:
                        out_queue.put(("token", cleaned))

                out_queue.put(("done", ""))

        except Exception as e:
            out_queue.put(("error", f"OpenAI-compatible 远程服务器调用失败：{e}"))

    def _stream_ollama_style_worker(self, backend, messages, out_queue, stop_event):
        """
        注意：
        本地 RKLLAMA / Ollama-compatible 统一使用 /api/chat。
        不使用 /api/generate，因为你实测 generate 容易重复感叹号。
        """
        url = f'{backend["base"]}/api/chat'

        payload = {
            "model": backend["model"],
            "messages": messages,
            "stream": True,
            "think": True,
            "keep_alive": "30m",
            "options": {
                "num_predict": max(256, LLM_NUM_PREDICT),
                "num_ctx": LLM_NUM_CTX,
                "temperature": LLM_TEMPERATURE,
                "top_p": LLM_TOP_P,
                "repeat_penalty": 1.2,
                "stop": [".setColumn"],
            },
        }

        cleaner = StreamCleaner()

        try:
            with requests.post(
                url,
                json=payload,
                stream=True,
                timeout=(5, 300),
            ) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines(decode_unicode=True):
                    if stop_event.is_set():
                        out_queue.put(("stopped", ""))
                        return

                    if not raw_line:
                        continue

                    try:
                        data = json.loads(raw_line)
                    except Exception:
                        continue

                    token = ""

                    if isinstance(data.get("message"), dict):
                        token = data["message"].get("content", "") or ""

                    if not token:
                        token = data.get("response", "") or ""

                    cleaned = cleaner.clean_token(token)

                    if cleaned.startswith("__STOP_"):
                        out_queue.put(("done", ""))
                        return

                    if cleaned:
                        out_queue.put(("token", cleaned))

                    if data.get("done") is True:
                        out_queue.put(("done", ""))
                        return

                out_queue.put(("done", ""))

        except Exception as e:
            out_queue.put(("error", f"Ollama/RKLLAMA-compatible 调用失败：{e}"))
