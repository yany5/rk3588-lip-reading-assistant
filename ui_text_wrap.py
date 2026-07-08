# -*- coding: utf-8 -*-
import pygame


def normalize_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", "    ")
    return text.strip()


def wrap_text_by_pixel(font, text, max_width):
    """
    按像素宽度换行，适合中文/英文混排。
    中文没有天然空格，所以这里按字符逐个试宽度。
    """
    text = normalize_text(text)
    if not text:
        return [""]

    lines = []

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue

        current = ""

        for ch in paragraph:
            test = current + ch
            if font.size(test)[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = ch

        if current:
            lines.append(current)

    return lines


def add_ellipsis_to_fit(font, text, max_width):
    """
    最后一行超长时加省略号，保证不会出框。
    """
    ellipsis = "…"
    if font.size(text)[0] <= max_width:
        return text

    while text and font.size(text + ellipsis)[0] > max_width:
        text = text[:-1]

    return text + ellipsis if text else ellipsis


def draw_text_box(
    surface,
    text,
    font,
    color,
    rect,
    max_lines=None,
    line_gap=8,
    align="left",
    valign="top",
    antialias=True,
):
    """
    在固定矩形区域内绘制多行文字。
    - 自动按像素宽度换行
    - 自动按高度限制行数
    - 最后一行自动省略
    - 支持 left / center / right 对齐
    """
    if not isinstance(rect, pygame.Rect):
        rect = pygame.Rect(rect)

    text = normalize_text(text)
    max_width = rect.width

    lines = wrap_text_by_pixel(font, text, max_width)

    line_height = font.get_linesize() + line_gap
    height_limit_lines = max(1, rect.height // line_height)

    if max_lines is None:
        max_lines = height_limit_lines
    else:
        max_lines = max(1, min(max_lines, height_limit_lines))

    overflow = len(lines) > max_lines
    lines = lines[:max_lines]

    if overflow and lines:
        lines[-1] = add_ellipsis_to_fit(font, lines[-1], max_width)

    total_h = len(lines) * line_height - line_gap

    if valign == "center":
        y = rect.y + max(0, (rect.height - total_h) // 2)
    elif valign == "bottom":
        y = rect.bottom - total_h
    else:
        y = rect.y

    for line in lines:
        surf = font.render(line, antialias, color)

        if align == "center":
            x = rect.x + (rect.width - surf.get_width()) // 2
        elif align == "right":
            x = rect.right - surf.get_width()
        else:
            x = rect.x

        surface.blit(surf, (x, y))
        y += line_height

    return lines
