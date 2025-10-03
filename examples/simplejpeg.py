"""Minimal baseline JPEG encoder for grayscale images using only the standard library."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple

__all__ = ["save_jpeg"]

ZIGZAG_ORDER: Sequence[Tuple[int, int]] = (
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (4, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 4),
    (0, 5),
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (5, 0),
    (6, 0),
    (5, 1),
    (4, 2),
    (3, 3),
    (2, 4),
    (1, 5),
    (0, 6),
    (0, 7),
    (1, 6),
    (2, 5),
    (3, 4),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (7, 1),
    (6, 2),
    (5, 3),
    (4, 4),
    (3, 5),
    (2, 6),
    (1, 7),
    (2, 7),
    (3, 6),
    (4, 5),
    (5, 4),
    (6, 3),
    (7, 2),
    (7, 3),
    (6, 4),
    (5, 5),
    (4, 6),
    (3, 7),
    (4, 7),
    (5, 6),
    (6, 5),
    (7, 4),
    (7, 5),
    (6, 6),
    (5, 7),
    (6, 7),
    (7, 6),
    (7, 7),
)

LUMA_QTABLE: Sequence[int] = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)

LUMA_DC_CODE_LENGTHS: Sequence[int] = (
    0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
)
LUMA_DC_VALUES: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

LUMA_AC_CODE_LENGTHS: Sequence[int] = (
    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03,
    0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
)
LUMA_AC_VALUES: Sequence[int] = (
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
)

INV_SQRT2 = 1.0 / math.sqrt(2.0)
COS_TABLE = [
    [math.cos(((2 * x + 1) * u * math.pi) / 16.0) for x in range(8)]
    for u in range(8)
]


def _build_huffman_table(code_lengths: Sequence[int], values: Sequence[int]) -> dict[int, Tuple[int, int]]:
    table: dict[int, Tuple[int, int]] = {}
    code = 0
    idx = 0
    for bit_length in range(1, 17):
        for _ in range(code_lengths[bit_length - 1]):
            table[values[idx]] = (code, bit_length)
            code += 1
            idx += 1
        code <<= 1
    return table


dc_huffman = _build_huffman_table(LUMA_DC_CODE_LENGTHS, LUMA_DC_VALUES)
ac_huffman = _build_huffman_table(LUMA_AC_CODE_LENGTHS, LUMA_AC_VALUES)


class BitWriter:
    def __init__(self) -> None:
        self._buffer = 0
        self._bits = 0
        self.bytes = bytearray()

    def write(self, value: int, length: int) -> None:
        for i in range(length - 1, -1, -1):
            bit = (value >> i) & 1
            self._buffer = (self._buffer << 1) | bit
            self._bits += 1
            if self._bits == 8:
                byte = self._buffer & 0xFF
                self.bytes.append(byte)
                if byte == 0xFF:
                    self.bytes.append(0x00)
                self._buffer = 0
                self._bits = 0

    def flush(self) -> None:
        if self._bits:
            byte = (self._buffer << (8 - self._bits)) & 0xFF
            self.bytes.append(byte)
            if byte == 0xFF:
                self.bytes.append(0x00)
            self._buffer = 0
            self._bits = 0


def _dct_2d(block: Sequence[Sequence[int]]) -> list[list[float]]:
    result = [[0.0 for _ in range(8)] for _ in range(8)]
    for u in range(8):
        for v in range(8):
            acc = 0.0
            for y in range(8):
                for x in range(8):
                    acc += block[y][x] * COS_TABLE[u][x] * COS_TABLE[v][y]
            cu = INV_SQRT2 if u == 0 else 1.0
            cv = INV_SQRT2 if v == 0 else 1.0
            result[u][v] = 0.25 * cu * cv * acc
    return result


def _quantize(block: Sequence[Sequence[float]]) -> list[list[int]]:
    quantized = [[0 for _ in range(8)] for _ in range(8)]
    for y in range(8):
        for x in range(8):
            idx = y * 8 + x
            quantized[y][x] = int(round(block[y][x] / LUMA_QTABLE[idx]))
    return quantized


def _zigzag(block: Sequence[Sequence[int]]) -> list[int]:
    return [block[y][x] for (y, x) in ZIGZAG_ORDER]


def _value_to_bits(value: int) -> Tuple[int, int]:
    if value == 0:
        return 0, 0
    abs_value = abs(value)
    size = abs_value.bit_length()
    if value > 0:
        return size, value
    max_value = (1 << size) - 1
    return size, max_value + value


def _encode_blocks(pixels: Sequence[Sequence[int]], width: int, height: int) -> bytes:
    padded_width = (width + 7) & ~7
    padded_height = (height + 7) & ~7

    writer = BitWriter()
    prev_dc = 0

    for by in range(0, padded_height, 8):
        for bx in range(0, padded_width, 8):
            block = [[0 for _ in range(8)] for _ in range(8)]
            for y in range(8):
                src_y = min(height - 1, by + y)
                row = pixels[src_y]
                for x in range(8):
                    src_x = min(width - 1, bx + x)
                    block[y][x] = row[src_x] - 128

            dct_block = _dct_2d(block)
            quant_block = _quantize(dct_block)
            zz = _zigzag(quant_block)

            dc = zz[0]
            diff = dc - prev_dc
            prev_dc = dc

            size, bits = _value_to_bits(diff)
            dc_code, dc_len = dc_huffman[size]
            writer.write(dc_code, dc_len)
            if size:
                writer.write(bits, size)

            zero_run = 0
            for coef in zz[1:]:
                if coef == 0:
                    zero_run += 1
                    if zero_run == 16:
                        zrl_code, zrl_len = ac_huffman[0xF0]
                        writer.write(zrl_code, zrl_len)
                        zero_run = 0
                    continue

                while zero_run > 15:
                    zrl_code, zrl_len = ac_huffman[0xF0]
                    writer.write(zrl_code, zrl_len)
                    zero_run -= 16

                size, bits = _value_to_bits(coef)
                symbol = (zero_run << 4) | size
                ac_code, ac_len = ac_huffman[symbol]
                writer.write(ac_code, ac_len)
                if size:
                    writer.write(bits, size)
                zero_run = 0

            if zero_run:
                eob_code, eob_len = ac_huffman[0x00]
                writer.write(eob_code, eob_len)

    writer.flush()
    return bytes(writer.bytes)


def _write_segment(marker: int, payload: Iterable[int]) -> bytearray:
    data = bytearray(payload)
    length = len(data) + 2
    segment = bytearray([0xFF, marker, (length >> 8) & 0xFF, length & 0xFF])
    segment.extend(data)
    return segment


def save_jpeg(pixels: Sequence[Sequence[int]], path: str | Path | None = None) -> bytes:
    if not pixels:
        raise ValueError("Pixel data cannot be empty")
    height = len(pixels)
    width = len(pixels[0])
    for row in pixels:
        if len(row) != width:
            raise ValueError("All rows must share the same width")

    output = bytearray()
    output.extend(b"\xFF\xD8")  # SOI

    app0 = bytearray(b"JFIF\x00")
    app0.extend(b"\x01\x01")  # version 1.1
    app0.append(0x00)  # units
    app0.extend(b"\x00\x01\x00\x01")  # density
    app0.extend(b"\x00\x00")
    output.extend(_write_segment(0xE0, app0))

    dqt = bytearray([0x00])
    dqt.extend(LUMA_QTABLE)
    output.extend(_write_segment(0xDB, dqt))

    sof = bytearray()
    sof.append(8)
    sof.extend(height.to_bytes(2, "big"))
    sof.extend(width.to_bytes(2, "big"))
    sof.append(1)
    sof.extend([1, 0x11, 0])
    output.extend(_write_segment(0xC0, sof))

    dht_dc = bytearray([0x00])
    dht_dc.extend(LUMA_DC_CODE_LENGTHS)
    dht_dc.extend(LUMA_DC_VALUES)
    output.extend(_write_segment(0xC4, dht_dc))

    dht_ac = bytearray([0x10])
    dht_ac.extend(LUMA_AC_CODE_LENGTHS)
    dht_ac.extend(LUMA_AC_VALUES)
    output.extend(_write_segment(0xC4, dht_ac))

    sos = bytearray()
    sos.append(1)
    sos.extend([1, 0x00])
    sos.extend([0x00, 0x3F, 0x00])
    output.extend(_write_segment(0xDA, sos))

    output.extend(_encode_blocks(pixels, width, height))
    output.extend(b"\xFF\xD9")

    data = bytes(output)
    if path is not None:
        Path(path).write_bytes(data)
    return data
