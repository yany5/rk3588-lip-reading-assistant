# -*- coding: utf-8 -*-
import re
import subprocess
from pathlib import Path

MAIN = Path("main.py")


def run_compile():
    p = subprocess.run(
        ["python3", "-m", "py_compile", str(MAIN)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.returncode, p.stdout, p.stderr


def extract_error_line(stderr):
    m = re.search(r'File "main\.py", line (\d+)', stderr)
    if not m:
        return None
    return int(m.group(1))


def find_unclosed_quote(line):
    """
    找一行里没有闭合的普通字符串/f-string/r-string。
    忽略三引号，专门修这种：
        x = f"第一行
        第二行"
    """
    i = 0
    quote = None
    open_pos = None
    escaped = False

    while i < len(line):
        ch = line[i]

        if quote is None:
            # 注释后面的内容不管
            if ch == "#":
                break

            # 跳过三引号，避免误伤 docstring
            if line.startswith('"""', i) or line.startswith("'''", i):
                i += 3
                continue

            if ch in ("'", '"'):
                quote = ch
                open_pos = i
                escaped = False
                i += 1
                continue

        else:
            if escaped:
                escaped = False
                i += 1
                continue

            if ch == "\\":
                escaped = True
                i += 1
                continue

            if ch == quote:
                quote = None
                open_pos = None
                i += 1
                continue

        i += 1

    if quote is not None and open_pos is not None:
        return open_pos, quote

    return None, None


def find_closing_quote(line, quote):
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue

        if ch == "\\":
            escaped = True
            continue

        if ch == quote:
            return i

    return None


def fix_at_line(lines, line_no):
    """
    从报错行开始，或者从上一行开始，寻找被断开的字符串。
    找到后把多行合成一行，并把中间换行转成 \\n。
    """
    candidates = [line_no - 1, line_no - 2]

    for start in candidates:
        if start < 0 or start >= len(lines):
            continue

        line = lines[start]
        open_pos, quote = find_unclosed_quote(line)

        if quote is None:
            continue

        prefix = line[:open_pos + 1]
        content = line[open_pos + 1:]

        j = start + 1
        while j < len(lines):
            cont_raw = lines[j]
            cont = cont_raw.strip()

            close_pos = find_closing_quote(cont, quote)

            if close_pos is not None:
                content += "\\n" + cont[:close_pos]
                suffix = cont[close_pos:]
                new_line = prefix + content + suffix

                lines[start:j + 1] = [new_line]
                return True, start + 1, j + 1, new_line

            content += "\\n" + cont
            j += 1

    return False, None, None, None


def main():
    fixed = 0

    for round_id in range(1, 80):
        code, out, err = run_compile()

        if code == 0:
            print(f"[OK] main.py 编译通过，共修复 {fixed} 处断裂字符串。")
            return 0

        if "unterminated string literal" not in err:
            print("[STOP] 不是 unterminated string literal，停止自动修复。")
            print(err)
            return 1

        line_no = extract_error_line(err)
        if line_no is None:
            print("[STOP] 无法解析错误行号。")
            print(err)
            return 1

        lines = MAIN.read_text(encoding="utf-8").splitlines()
        ok, start, end, new_line = fix_at_line(lines, line_no)

        if not ok:
            print(f"[STOP] 第 {line_no} 行附近没有找到可自动修复的断裂字符串。")
            a = max(1, line_no - 8)
            b = min(len(lines), line_no + 8)
            for idx in range(a, b + 1):
                print(f"{idx:04d}: {lines[idx-1]}")
            print(err)
            return 1

        MAIN.write_text("\n".join(lines) + "\n", encoding="utf-8")
        fixed += 1
        print(f"[FIX {fixed}] 修复 main.py 第 {start}-{end} 行 -> {new_line[:120]}")

    print("[STOP] 修复次数超过上限，可能文件被严重破坏。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
