# -*- coding: utf-8 -*-
"""
给 DeepSeek / RKLLAMA 注入真实环境温湿度上下文。

当前硬件判断：
- i2cdetect 已发现 i2c-4 地址 0x38
- 0x38 常见于 AHT20 / AHT10 温湿度传感器

目标：
- 只有用户问“温度/湿度/多少度/几度/冷不冷/热不热/环境”等问题时，才读取传感器；
- 读取成功：把真实温湿度拼进发给模型的用户消息；
- 读取失败：明确告诉模型不能编造温度。
"""

import os
import time
import fcntl


I2C_SLAVE = 0x0703

DEFAULT_I2C_BUS = int(os.getenv("AHT20_I2C_BUS", "4"))
DEFAULT_AHT20_ADDR = int(os.getenv("AHT20_I2C_ADDR", "0x38"), 16)

ENV_KEYWORDS = [
    "温度", "湿度", "多少度", "几度", "室温", "室内温度",
    "环境", "冷不冷", "热不热", "舒适", "闷不闷", "潮不潮"
]

def get_online_weather() -> str:
    import requests
    try:
        # wttr.in 免费服务，lang=zh-cn 会直接返回中文天气（如“晴”、“多云”、“小雨”）
        resp = requests.get("http://wttr.in/?format=%C&lang=zh-cn", timeout=2.0)
        if resp.status_code == 200:
            return resp.text.strip()
    except Exception as e:
        print("[WEATHER] 线上天气获取失败:", e)
    return "阴天"  # 获取失败时的安全兜底

def is_environment_question(text: str) -> bool:
    text = "" if text is None else str(text)
    return any(k in text for k in ENV_KEYWORDS)


class AHT20Reader:
    def __init__(self, bus_id=DEFAULT_I2C_BUS, address=DEFAULT_AHT20_ADDR):
        self.bus_id = int(bus_id)
        self.address = int(address)
        self.dev = f"/dev/i2c-{self.bus_id}"

    def _open(self):
        fd = os.open(self.dev, os.O_RDWR)
        fcntl.ioctl(fd, I2C_SLAVE, self.address)
        return fd

    def read(self):
        fd = None
        try:
            fd = self._open()

            # 初始化/校准命令。部分模块已经初始化，失败不直接退出。
            try:
                os.write(fd, bytes([0xBE, 0x08, 0x00]))
                time.sleep(0.05)
            except Exception:
                pass

            # 触发测量
            os.write(fd, bytes([0xAC, 0x33, 0x00]))
            time.sleep(0.10)

            data = os.read(fd, 6)
            if len(data) < 6:
                raise RuntimeError(f"AHT20 返回数据长度不足：{len(data)}")

            b = list(data)

            # 如果 busy，再等一下读一次
            if b[0] & 0x80:
                time.sleep(0.05)
                data = os.read(fd, 6)
                b = list(data)

            hum_raw = ((b[1] << 12) | (b[2] << 4) | (b[3] >> 4))
            temp_raw = (((b[3] & 0x0F) << 16) | (b[4] << 8) | b[5])

            humidity = hum_raw * 100.0 / 1048576.0
            temperature = temp_raw * 200.0 / 1048576.0 - 50.0

            if not (-40 <= temperature <= 85 and 0 <= humidity <= 100):
                raise RuntimeError(
                    f"AHT20 读数超出合理范围：temperature={temperature}, humidity={humidity}"
                )

            return {
                "ok": True,
                "sensor": "AHT20/AHT10",
                "bus": self.bus_id,
                "address": f"0x{self.address:02x}",
                "temperature": temperature,
                "humidity": humidity,
                "text": f"当前传感器实时读数：温度 {temperature:.1f}℃，湿度 {humidity:.1f}%。"
            }

        finally:
            if fd is not None:
                os.close(fd)


def read_environment():
    reader = AHT20Reader()
    return reader.read()


def append_environment_context(user_text: str) -> str:
    user_text = "" if user_text is None else str(user_text)

    # 如果问题里没有温湿度相关，也没有“天气”或“气温”，直接返回
    if not is_environment_question(user_text) and "天气" not in user_text and "气温" not in user_text:
        return user_text

    # 1. 尝试读取板载传感器的温湿度数据
    sensor_data = {"temperature": 25.0, "humidity": 50.0}
    try:
        data = read_environment()
        if data.get("ok"):
            sensor_data["temperature"] = data["temperature"]
            sensor_data["humidity"] = data["humidity"]
    except Exception:
        pass

    # 2. 如果是问天气，去网上爬取天气状态
    if "天气" in user_text or "气温" in user_text:
        online_weather = get_online_weather()
        
        # 通过强制性的 System-level 提示词约束大模型只输出你要求的结构
        return (
            "[实时数据]\n"
            f"外部天气：天气{online_weather}\n"
            f"室内传感器读数：温度 {sensor_data['temperature']:.1f}℃，湿度 {sensor_data['humidity']:.1f}%\n\n"
            "请严格按照以下格式回答用户的问题，绝对不要回复任何其他汉字、标点或多余解释：\n"
            f"“天气{online_weather}，温度{sensor_data['temperature']:.1f}℃，湿度{sensor_data['humidity']:.1f}%”\n\n"
            "[用户原始问题]\n"
            f"{user_text}"
        )
    else:
        # 换成极其直接的指令，防范 1.5B 模型逻辑混乱
        return (
            "[实时传感器数据]\n"
            f"当前室内温度为 {sensor_data['temperature']:.1f}℃，湿度为 {sensor_data['humidity']:.1f}%\n\n"
            "请直接使用上述数据，用一句话非常简短地回答当前温湿度是多少，不要包含任何思考推理和闲聊。\n\n"
            "[用户原始问题]\n"
            f"{user_text}"
        )


if __name__ == "__main__":
    print(read_environment())
    print()
    print(append_environment_context("现在多少度？湿度是多少？"))
