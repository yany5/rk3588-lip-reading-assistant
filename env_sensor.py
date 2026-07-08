#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import fcntl

I2C_SLAVE = 0x0703

# 你现在已经确认 AHT20 在 /dev/i2c-4，地址 0x38
AHT20_BUS = 4
AHT20_ADDR = 0x38


def _read_aht20_once(bus_num=AHT20_BUS, addr=AHT20_ADDR):
    dev = f"/dev/i2c-{bus_num}"
    fd = os.open(dev, os.O_RDWR)

    try:
        fcntl.ioctl(fd, I2C_SLAVE, addr)

        # 初始化 AHT20
        os.write(fd, bytes([0xBE, 0x08, 0x00]))
        time.sleep(0.05)

        # 触发测量
        os.write(fd, bytes([0xAC, 0x33, 0x00]))
        time.sleep(0.10)

        data = os.read(fd, 7)

        if len(data) != 7:
            raise RuntimeError(f"读取长度错误：{len(data)}")

        # data[0] bit7 为 busy 标志，1 表示仍在测量
        if data[0] & 0x80:
            raise RuntimeError("AHT20 仍处于 busy 状态，请稍后重试")

        raw_h = ((data[1] << 12) | (data[2] << 4) | (data[3] >> 4))
        raw_t = (((data[3] & 0x0F) << 16) | (data[4] << 8) | data[5])

        humidity = raw_h * 100.0 / 1048576.0
        temperature = raw_t * 200.0 / 1048576.0 - 50.0

        return {
            "ok": True,
            "sensor": "AHT20",
            "bus": bus_num,
            "address": f"0x{addr:02x}",
            "temperature_c": round(temperature, 2),
            "humidity_percent": round(humidity, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    finally:
        os.close(fd)


def read_environment():
    try:
        return _read_aht20_once()
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "sensor": "AHT20",
            "bus": AHT20_BUS,
            "address": f"0x{AHT20_ADDR:02x}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def get_environment_text():
    data = read_environment()

    if not data.get("ok"):
        return (
            "当前设备未能读取到温湿度传感器数据。"
            f"传感器：{data.get('sensor', 'AHT20')}，"
            f"I2C 总线：/dev/i2c-{data.get('bus', AHT20_BUS)}，"
            f"地址：{data.get('address', '0x38')}，"
            f"错误信息：{data.get('error', '未知错误')}。"
        )

    return (
        f"当前环境温度为 {data['temperature_c']} ℃，"
        f"当前环境相对湿度为 {data['humidity_percent']} %RH。"
        f"数据来自 {data['sensor']} 温湿度传感器，"
        f"I2C 总线 /dev/i2c-{data['bus']}，地址 {data['address']}，"
        f"读取时间 {data['timestamp']}。"
    )


if __name__ == "__main__":
    data = read_environment()
    print(data)
    print(get_environment_text())
