#!/usr/bin/env bash
set +e

echo "========== 1. 基本系统信息 =========="
uname -a
echo
cat /etc/os-release 2>/dev/null | head -20
echo

echo "========== 2. 确认是否 RK3588 / ELF2 =========="
echo "[compatible]"
tr '\0' '\n' < /proc/device-tree/compatible 2>/dev/null
echo
echo "[model]"
tr '\0' '\n' < /proc/device-tree/model 2>/dev/null
echo

echo "========== 3. 查 NPU 设备节点 =========="
ls -l /dev/rknpu* /dev/rknn* 2>/dev/null || echo "[NO] 没有 /dev/rknpu* 或 /dev/rknn*"
echo

echo "========== 4. 挂载 debugfs 并查 RKNPU 版本 =========="
sudo mount -t debugfs none /sys/kernel/debug 2>/dev/null
if [ -e /sys/kernel/debug/rknpu/version ]; then
    cat /sys/kernel/debug/rknpu/version
else
    echo "[NO] 没有 /sys/kernel/debug/rknpu/version"
fi
echo

echo "========== 5. dmesg 搜索 rknpu/npu/rknn =========="
dmesg | grep -Ei "rknpu|rknn|npu|fdab0000" | tail -80
echo

echo "========== 6. 查内核模块 =========="
lsmod | grep -Ei "rknpu|rknn|npu" || echo "[NO] lsmod 未看到 rknpu/rknn/npu 模块"
echo

echo "========== 7. 查 sysfs 里的 NPU 节点 =========="
find /sys -iname "*rknpu*" -o -iname "*rknn*" -o -iname "*npu*" 2>/dev/null | head -100
echo

echo "========== 8. 查内核配置是否包含 RKNPU =========="
if [ -e /proc/config.gz ]; then
    zcat /proc/config.gz | grep -Ei "RKNPU|ROCKCHIP.*NPU|RKNN"
else
    grep -Ei "RKNPU|ROCKCHIP.*NPU|RKNN" /boot/config-$(uname -r) 2>/dev/null || echo "[NO] 没找到 /proc/config.gz 或 /boot/config"
fi
echo

echo "========== 9. 查 8080 服务 =========="
ss -lntp | grep ':8080' || echo "[NO] 8080 没有服务监听"
curl -sS http://127.0.0.1:8080/api/tags || true
echo

echo "========== 10. 查 rkllama 命令 =========="
which rkllama_server || echo "[NO] 没有 rkllama_server"
which rkllama_client || echo "[NO] 没有 rkllama_client"
echo

echo "========== 诊断完成 =========="
