# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import lzw

from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from dataclasses import dataclass, replace
from functools import partial, wraps
from itertools import batched, chain, repeat
from struct import pack, unpack
from enum import Enum

from typing import ClassVar, Optional, Self

type Parsed[T] = tuple[bytes, T]

type SubBlock = bytes
type Block = list[SubBlock]


class SectionType(Enum):
    IMAGE     = 0x2C
    EXTENSION = 0x21
    TRAILER   = 0x3B


class ExtensionType(Enum):
    PLAIN_TEXT      = 0x01
    GRAPHIC_CONTROL = 0xF9
    COMMENT         = 0xFE
    APPLICATION     = 0xFF


class ParseError(ValueError): ...


class Serializable(ABC):
    @classmethod
    @abstractmethod
    def decode(cls, stream: bytes, *args, **kwargs) -> Parsed[Self]:
        ...

    @abstractmethod
    def encode(self) -> bytes:
        ...

    @staticmethod
    def stream_length_at_least(length):
        def wrapper(f):
            @wraps(f)
            def inner(cls, stream, *args, **kwargs):
                if len(stream) < length:
                    raise ParseError("unexpected end of stream")

                return f(cls, stream, *args, **kwargs)

            return inner

        return wrapper


@dataclass
class Color(Serializable):
    DEFAULT: ClassVar[Color]

    red: int
    green: int
    blue: int

    is_transparent: bool=False

    @classmethod
    @Serializable.stream_length_at_least(3)
    def decode(cls, stream):
        return stream[3:], cls(*stream[:3])

    def encode(self):
        return bytes([self.red, self.green, self.blue])


Color.DEFAULT = Color(0, 0, 0)


@dataclass
class ColorTable(MutableSequence[Color], Serializable):
    MAX_SIZE: ClassVar[int] = 256

    data: list[Color]
    is_sorted: bool

    @classmethod
    def decode(cls, stream: bytes, size: int, is_sorted: bool):
        if size not in range(1, cls.MAX_SIZE + 1):
            raise ParseError("invalid color table size")

        data = []
        for _ in range(size):
            stream, color = Color.decode(stream)
            data.append(color)

        return stream, cls(data, is_sorted)

    def encode(self):
        if len(self) == 0:
            return b""

        length_bit_count = (len(self) - 1).bit_length()
        padding_length = 2**length_bit_count - len(self)

        return b"".join(map(Color.encode, chain(self.data, repeat(Color.DEFAULT, padding_length))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, color):
        self.data[i] = color

    def __delitem__(self, i):
        del self.data[i]

    def insert(self, i, color):
        if len(self.data) == ColorTable.MAX_SIZE:
            raise ValueError("color table too big")

        self.data.insert(i, color)


@dataclass
class Screen(Serializable):
    width: int
    height: int
    color_table: Optional[ColorTable]
    color_resolution: int
    background_color_index: int
    pixel_aspect_ratio: int

    @classmethod
    @Serializable.stream_length_at_least(7)
    def decode(cls, stream: bytes):
        width, height, packed_fields, background_color_index, pixel_aspect_ratio = unpack("<HHBBB", stream[:7])
        stream = stream[7:]

        has_color_table =    bool((packed_fields >> 7) & 1)
        color_resolution =        (packed_fields >> 4) & 7
        color_table_is_sorted =   (packed_fields >> 3) & 1
        color_table_size = 2 ** (((packed_fields >> 0) & 7) + 1)

        color_table = None
        if has_color_table:
            stream, color_table = ColorTable.decode(stream, color_table_size, color_table_is_sorted)

        return stream, Screen(width, height, color_table, color_resolution, background_color_index, pixel_aspect_ratio)

    def encode(self):
        packed_fields = self.color_resolution << 4
        if self.color_table is not None:
            packed_fields |=                                              1 << 7  # has color table
            packed_fields |=                     self.color_table.is_sorted << 3
            packed_fields |= ((len(self.color_table) - 1).bit_length() - 1) << 0  # color table size

        result = pack("<HHBBB", self.width, self.height, packed_fields, self.background_color_index, self.pixel_aspect_ratio)
        if self.color_table is not None:
            result += self.color_table.encode()

        return result


def decode_subblock(stream: bytes) -> Parsed[SubBlock]:
    if len(stream) < 1:
        raise ParseError("unexpected end of stream")

    length = stream[0]
    stream = stream[1:]

    if len(stream) < length:
        raise ParseError("unexpected end of stream")

    return stream[length:], stream[:length]


def encode_subblock(subblock: SubBlock) -> bytes:
    return bytes([len(subblock)]) + subblock


TERMINATOR_SUBBLOCK = b""


def decode_block(stream: bytes) -> Parsed[Block]:
    block = []
    while True:
        stream, subblock = decode_subblock(stream)
        if subblock == TERMINATOR_SUBBLOCK:
            break

        block.append(subblock)

    return stream, block


def encode_block(block: Block) -> bytes:
    return b"".join(map(encode_subblock, chain(block, [TERMINATOR_SUBBLOCK])))


@dataclass
class GraphicControlExtension(Serializable):
    disposal_method: int
    waits_for_user_input: int
    delay_time: int
    transparent_color_index: Optional[int]

    @classmethod
    def decode(cls, stream: bytes):
        stream, block = decode_block(stream)
        if len(block) != 1:
            raise ParseError("too many blocks in graphic control extension")

        if len(block[0]) != 4:
            raise ParseError("invalid graphic control extension initial block size")

        packed_fields, delay_time, transparent_color_index = unpack("<BHB", block[0])

        has_transparent_color = bool((packed_fields >> 0) & 1)
        waits_for_user_input  = bool((packed_fields >> 1) & 1)
        disposal_method =            (packed_fields >> 2) & 7
        if not has_transparent_color:
            transparent_color_index = None

        return stream, cls(disposal_method, waits_for_user_input, delay_time, transparent_color_index)


    def encode(self):
        packed_fields = (self.disposal_method << 2) | (self.waits_for_user_input << 1)
        if self.transparent_color_index is not None:
            packed_fields |= 1 << 0  # has transparent color

        return bytes([SectionType.EXTENSION.value, ExtensionType.GRAPHIC_CONTROL.value]) \
             + encode_block([pack("<BHB", packed_fields, self.delay_time, self.transparent_color_index or 0)])


class Section(Serializable): ...


@dataclass
class ApplicationExtension(Section):
    identifier: bytes
    authentication_code: bytes
    data: Block

    @classmethod
    def decode(cls, stream: bytes):
        stream, block = decode_block(stream)
        if len(block) < 1:
            raise ParseError("empty application extension")

        if len(block[0]) != 11:
            raise ParseError("invalid initial application extension block size")

        identifier = block[0][:8]
        authentication_code = block[0][8:]

        return stream, cls(identifier, authentication_code, block[1:])

    def encode(self):
        return bytes([SectionType.EXTENSION.value, ExtensionType.APPLICATION.value]) \
             + encode_block([self.identifier + self.authentication_code] + self.data)


@dataclass
class CommentExtension(Section):
    data: list[str]

    @classmethod
    def decode(cls, stream: bytes):
        stream, block = decode_block(stream)

        return stream, cls(list(map(partial(bytes.decode, encoding="ASCII"), block)))

    def encode(self):
        return bytes([SectionType.EXTENSION.value, ExtensionType.COMMENT.value]) \
             + encode_block(map(partial(str.encode, encoding="ASCII"), self.data))


@dataclass
class PlainTextExtension(Section):
    top: int
    left: int
    width: int
    height: int
    cell_width: int
    cell_height: int
    foreground_color_index: int
    background_color_index: int
    data: Block

    @classmethod
    def decode(cls, stream: bytes):
        stream, block = decode_block(stream)
        top, left, width, height, cell_width, cell_height, foreground_color_index, background_color_index = unpack("<HHHHBBBB", block[0])

        return stream, cls(top, left, width, height, cell_width, cell_height, foreground_color_index, background_color_index, block[1:])

    def encode(self):
        return bytes([SectionType.EXTENSION.value, ExtensionType.PLAIN_TEXT.value]) \
             + encode_block([pack("<HHHHBBBB", self.top, self.left, self.width, self.height, self.cell_width, self.cell_height, self.foreground_color_index, self.background_color_index)] + self.data)


@dataclass
class Image(Section):
    graphic_control_extension: Optional[GraphicControlExtension]
    left: int
    top: int
    width: int
    height: int
    color_table: Optional[ColorTable]
    is_interlaced: bool
    minimum_code_size: int
    data: Block

    def get_pixels(self, global_color_table: Optional[ColorTable]=None) -> list[list[Color]]:
        color_table = global_color_table if self.color_table is None else self.color_table
        data = lzw.decode(self.minimum_code_size, chain.from_iterable(self.data))

        return [[replace(
                     color_table[i],
                     is_transparent
                         =   self.graphic_control_extension is not None
                         and i == self.graphic_control_extension.transparent_color_index)
                 for i in row_indices]
                 for row_indices in batched(data, self.width)]

    @classmethod
    @Serializable.stream_length_at_least(9)
    def decode(cls, stream: bytes, graphic_control_extension: GraphicControlExtension):
        left, top, width, height, packed_fields = unpack("<HHHHB", stream[:9])
        stream = stream[9:]

        has_color_table = bool(packed_fields >> 7)

        has_color_table =    bool((packed_fields >> 7) & 1)
        is_interlaced =      bool((packed_fields >> 6) & 1)
        color_table_is_sorted =   (packed_fields >> 5) & 1
        color_table_size = 2 ** (((packed_fields >> 0) & 7) + 1)

        color_table = None
        if has_color_table:
            stream, color_table = ColorTable.decode(stream, color_table_size, color_table_is_sorted)

        minimum_code_size = stream[0]
        stream = stream[1:]

        stream, data = decode_block(stream)

        return stream, cls(graphic_control_extension, left, top, width, height, color_table, is_interlaced, minimum_code_size, data)

    def encode(self):
        packed_fields = self.is_interlaced << 6
        if self.color_table is not None:
            packed_fields |=                                              1 << 7  # has color table
            packed_fields |=                     self.color_table.is_sorted << 5
            packed_fields |= ((len(self.color_table) - 1).bit_length() - 1) << 0  # color table size

        result = b""

        if self.graphic_control_extension is not None:
            result += self.graphic_control_extension.encode()

        result += bytes([SectionType.IMAGE.value]) + pack("<HHHHB", self.left, self.top, self.width, self.height, packed_fields)

        if self.color_table is not None:
            result += self.color_table.encode()

        result += bytes([self.minimum_code_size]) + encode_block(self.data)

        return result


@dataclass
class GIF(Serializable):
    signature: bytes
    version: bytes
    screen: Screen
    sections: list[Section]

    @classmethod
    @Serializable.stream_length_at_least(6)
    def decode(cls, stream: bytes):
        signature, version = stream[:3], stream[3:6]
        stream = stream[6:]

        if signature != b"GIF":
            raise ParseError(f"invalid signature: {signature!r}")

        if version not in (b"87a", b"89a"):
            raise ParseError(f"invalid version: {version!r}")

        stream, screen = Screen.decode(stream)

        graphic_control_extension: Optional[GraphicControlExtension] = None
        sections: list[Section] = []
        while True:
            try:
                section_type = SectionType(stream[0])
            except IndexError:
                raise ParseError("unexpected end of stream") from None
            except ValueError:
                raise ParseError(f"unknown section type: 0x{stream[0]:02x}") from None
            else:
                stream = stream[1:]

            match section_type:
                case SectionType.EXTENSION:
                    try:
                        extension_type = ExtensionType(stream[0])
                    except IndexError:
                        raise ParseError("unexpected end of stream") from None
                    except ValueError:
                        raise ParseError(f"unknown extension type: 0x{stream[0]:02x}") from None
                    else:
                        stream = stream[1:]

                    match extension_type:
                        case ExtensionType.APPLICATION:
                            stream, application_extension = ApplicationExtension.decode(stream)
                            sections.append(application_extension)
                        case ExtensionType.COMMENT:
                            stream, comment_extension = CommentExtension.decode(stream)
                            sections.append(comment_extension)
                        case ExtensionType.PLAIN_TEXT:
                            stream, plain_text_extension = PlainTextExtension.decode(stream)
                            sections.append(plain_text_extension)
                        case ExtensionType.GRAPHIC_CONTROL:
                            stream, graphic_control_extension = GraphicControlExtension.decode(stream)
                case SectionType.IMAGE:
                    stream, image = Image.decode(stream, graphic_control_extension)
                    sections.append(image)
                    graphic_control_extension = None
                case SectionType.TRAILER:
                    break

        return stream, cls(signature, version, screen, sections)

    def encode(self):
        result = self.signature + self.version
        result += self.screen.encode()
        result += b"".join(map(lambda x: x.encode(), self.sections))
        result += bytes([SectionType.TRAILER.value])

        return result
