# =====================================================================
# MODIFICATION NOTICE:
# - This is a NEW module added to the project.
# - Added: LipReadingService class to simulate a lip-reading classifier.
# - Added: A rotating buffer of English mock sentences emitted every 5 seconds
#   when active, making the UI and LLM chat testable without a real model.
# =====================================================================

import time

class LipReadingService:
    """
    唇语识别服务模拟器：
    1. 模拟实时视频流唇部翻译
    2. 每隔一定时间或手动触发返回一段英文语句
    """

    def __init__(self):
        self.is_active = False
        self.start_time = 0.0
        self.mock_sentences = [
            "Hello, welcome to use the lip reading assistant!",
            "How is the weather today in Shenzhen?",
            "The RK3588 board has powerful NPU computing capacity.",
            "Artificial intelligence is changing our daily life.",
            "Can you tell me a short story?",
            "Thank you for your question. See you next time!"
        ]
        self.mock_index = 0

    def start(self):
        self.is_active = True
        self.start_time = time.time()

    def get_text(self):
        """
        模拟轮询：在激活状态下，模拟说话 5 秒后，返回一句识别出的文本
        返回：(has_new_text, text)
        """
        if not self.is_active:
            return False, ""
        
        elapsed = time.time() - self.start_time
        if elapsed >= 5.0:
            text = self.mock_sentences[self.mock_index]
            self.mock_index = (self.mock_index + 1) % len(self.mock_sentences)
            self.start_time = time.time()  # 重置计时
            return True, text
            
        return False, ""


    def recognize_once(self, video_path=None, timeout=6.0):
        """
        单次唇语识别接口。
        当前仍然使用模拟结果；后续真实唇语模型接入时，只需要在这里读取 video_path 并返回中文句子。
        """
        self.start()
        deadline = time.time() + timeout

        while time.time() < deadline:
            has_new, text = self.get_text()
            if has_new:
                self.stop()
                return text
            time.sleep(0.05)

        # 超时兜底：返回一条模拟结果，避免界面一直卡住
        text = self.mock_sentences[self.mock_index]
        self.mock_index = (self.mock_index + 1) % len(self.mock_sentences)
        self.stop()
        return text


    def stop(self):
        self.is_active = False

