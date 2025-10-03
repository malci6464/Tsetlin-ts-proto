"""Utility helpers for drawing simple grayscale graphics without external dependencies."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

Color = int

FONT_5X7 = {
    " ": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ],
    "0": [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110",
    ],
    "1": [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "2": [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ],
    "3": [
        "11110",
        "00001",
        "00001",
        "00110",
        "00001",
        "00001",
        "11110",
    ],
    "4": [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    "5": [
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "10001",
        "01110",
    ],
    "6": [
        "00110",
        "01000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110",
    ],
    "7": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000",
    ],
    "8": [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ],
    "9": [
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00010",
        "01100",
    ],
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "B": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10001",
        "10001",
        "11110",
    ],
    "C": [
        "01110",
        "10001",
        "10000",
        "10000",
        "10000",
        "10001",
        "01110",
    ],
    "D": [
        "11100",
        "10010",
        "10001",
        "10001",
        "10001",
        "10010",
        "11100",
    ],
    "E": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],
    "F": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "10000",
    ],
    "G": [
        "01110",
        "10001",
        "10000",
        "10111",
        "10001",
        "10001",
        "01111",
    ],
    "H": [
        "10001",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "I": [
        "01110",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "J": [
        "00001",
        "00001",
        "00001",
        "00001",
        "10001",
        "10001",
        "01110",
    ],
    "K": [
        "10001",
        "10010",
        "10100",
        "11000",
        "10100",
        "10010",
        "10001",
    ],
    "L": [
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "11111",
    ],
    "M": [
        "10001",
        "11011",
        "10101",
        "10101",
        "10001",
        "10001",
        "10001",
    ],
    "N": [
        "10001",
        "11001",
        "10101",
        "10011",
        "10001",
        "10001",
        "10001",
    ],
    "O": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "P": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10000",
        "10000",
        "10000",
    ],
    "Q": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10101",
        "10010",
        "01101",
    ],
    "R": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10100",
        "10010",
        "10001",
    ],
    "S": [
        "01111",
        "10000",
        "10000",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "T": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "U": [
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "V": [
        "10001",
        "10001",
        "10001",
        "01010",
        "01010",
        "00100",
        "00100",
    ],
    "W": [
        "10001",
        "10001",
        "10001",
        "10101",
        "10101",
        "10101",
        "01010",
    ],
    "X": [
        "10001",
        "01010",
        "00100",
        "00100",
        "00100",
        "01010",
        "10001",
    ],
    "Y": [
        "10001",
        "01010",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "Z": [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "10000",
        "11111",
    ],
    "-": [
        "00000",
        "00000",
        "00000",
        "11111",
        "00000",
        "00000",
        "00000",
    ],
    ",": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00110",
        "00100",
    ],
    ".": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00110",
        "00110",
    ],
    ":": [
        "00000",
        "00110",
        "00110",
        "00000",
        "00110",
        "00110",
        "00000",
    ],
    "/": [
        "00001",
        "00010",
        "00100",
        "01000",
        "10000",
        "00000",
        "00000",
    ],
    "%": [
        "11001",
        "11010",
        "00010",
        "00100",
        "01000",
        "01011",
        "10011",
    ],
}


class SimpleCanvas:
    def __init__(self, width: int, height: int, background: Color = 255) -> None:
        self.width = width
        self.height = height
        self.pixels: List[List[Color]] = [[background for _ in range(width)] for _ in range(height)]

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = max(0, min(255, color))

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def draw_horizontal_line(self, x0: int, x1: int, y: int, color: Color) -> None:
        if x0 > x1:
            x0, x1 = x1, x0
        for x in range(x0, x1 + 1):
            self.set_pixel(x, y, color)

    def draw_vertical_line(self, x: int, y0: int, y1: int, color: Color) -> None:
        if y0 > y1:
            y0, y1 = y1, y0
        for y in range(y0, y1 + 1):
            self.set_pixel(x, y, color)

    def draw_rect(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        self.draw_horizontal_line(x0, x1, y0, color)
        self.draw_horizontal_line(x0, x1, y1, color)
        self.draw_vertical_line(x0, y0, y1, color)
        self.draw_vertical_line(x1, y0, y1, color)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        for y in range(max(0, y0), min(self.height, y1 + 1)):
            for x in range(max(0, x0), min(self.width, x1 + 1)):
                self.set_pixel(x, y, color)

    def draw_polyline(self, points: Sequence[Tuple[int, int]], color: Color) -> None:
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            self.draw_line(x0, y0, x1, y1, color)

    def draw_text(self, x: int, y: int, text: str, color: Color, scale: int = 1) -> None:
        cursor_x = x
        for char in text.upper():
            glyph = FONT_5X7.get(char, FONT_5X7[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        for sy in range(scale):
                            for sx in range(scale):
                                self.set_pixel(cursor_x + gx * scale + sx, y + gy * scale + sy, color)
            cursor_x += (len(glyph[0]) + 1) * scale

    def measure_text(self, text: str, scale: int = 1) -> int:
        width = 0
        for char in text.upper():
            glyph = FONT_5X7.get(char, FONT_5X7[" "])
            width += (len(glyph[0]) + 1) * scale
        if width:
            width -= scale  # remove trailing spacing
        return width

    def to_pixels(self) -> List[List[Color]]:
        return [row[:] for row in self.pixels]
