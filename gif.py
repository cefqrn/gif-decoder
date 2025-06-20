# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from serializable import ParseError, Parsed, Serializable, Stream, stream_length_at_least
import lzw

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from itertools import batched, chain, repeat
from struct import pack, unpack
from enum import IntEnum

from typing import Any, ClassVar, Optional, overload
from abc import abstractmethod

type SubBlock = bytes
type DataBlock = tuple[SubBlock, ...]


class BlockType(IntEnum):
    EXTENSION = 0x21
    IMAGE     = 0x2C
    TRAILER   = 0x3B


class ExtensionType(IntEnum):
    PLAIN_TEXT      = 0x01
    GRAPHIC_CONTROL = 0xF9
    COMMENT         = 0xFE
    APPLICATION     = 0xFF


@dataclass(frozen=True)
class Color(Serializable):
    red: int
    green: int
    blue: int

    is_transparent: bool=False

    @staticmethod
    @stream_length_at_least(3)
    def decode(stream: Stream) -> Parsed[Color]:
        return stream[3:], Color(stream[0], stream[1], stream[2])

    def encode(self) -> bytes:
        return bytes([self.red, self.green, self.blue])


@dataclass(frozen=True)
class ColorTable(Sequence[Color], Serializable):
    MAX_SIZE: ClassVar[int] = 256
    PADDING_COLOR: ClassVar[Color] = Color(0, 0, 0)

    data: tuple[Color, ...]
    is_sorted: bool

    @staticmethod
    def decode(stream: Stream, size: int, is_sorted: bool) -> Parsed[ColorTable]:
        if size not in range(1, ColorTable.MAX_SIZE + 1):
            raise ParseError("invalid color table size")

        data = []
        for _ in range(size):
            stream, color = Color.decode(stream)
            data.append(color)

        return stream, ColorTable(tuple(data), is_sorted)

    def encode(self) -> bytes:
        if len(self) == 0:
            return b""

        length_bit_count = (len(self) - 1).bit_length()
        padding_length = 2**length_bit_count - len(self)

        return b"".join(map(Color.encode, chain(self.data, repeat(self.PADDING_COLOR, padding_length))))

    def __len__(self) -> int:
        return len(self.data)

    @overload
    def __getitem__(self, i: int, /) -> Color: ...
    @overload
    def __getitem__(self, i: slice[Any, Any, Any], /) -> Sequence[Color]: ...

    def __getitem__(self, i: int | slice[Any, Any, Any], /) -> Color | Sequence[Color]:
        return self.data[i]


@dataclass(frozen=True)
class Screen(Serializable):
    width: int
    height: int
    color_table: Optional[ColorTable]
    color_resolution: int
    background_color_index: int
    pixel_aspect_ratio: int

    @staticmethod
    @stream_length_at_least(7)
    def decode(stream: Stream) -> Parsed[Screen]:
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

    def encode(self) -> bytes:
        packed_fields = self.color_resolution << 4
        if self.color_table is not None:
            packed_fields |=                                              1 << 7  # has color table
            packed_fields |=                     self.color_table.is_sorted << 3
            packed_fields |= ((len(self.color_table) - 1).bit_length() - 1) << 0  # color table size

        result = pack("<HHBBB", self.width, self.height, packed_fields, self.background_color_index, self.pixel_aspect_ratio)
        if self.color_table is not None:
            result += self.color_table.encode()

        return result


def decode_subblock(stream: Stream) -> Parsed[SubBlock]:
    if len(stream) < 1:
        raise ParseError("unexpected end of stream")

    length = stream[0]
    stream = stream[1:]

    if len(stream) < length:
        raise ParseError("unexpected end of stream")

    return stream[length:], stream[:length].tobytes()


def encode_subblock(subblock: SubBlock) -> bytes:
    return bytes([len(subblock)]) + subblock


TERMINATOR_SUBBLOCK = b""


def decode_data_block(stream: Stream) -> Parsed[DataBlock]:
    block = []
    while True:
        stream, subblock = decode_subblock(stream)
        if subblock == TERMINATOR_SUBBLOCK:
            break

        block.append(subblock)

    return stream, tuple(block)


def encode_data_block(block: DataBlock) -> bytes:
    return b"".join(map(encode_subblock, chain(block, [TERMINATOR_SUBBLOCK])))


class LabeledBlock(Serializable):
    @staticmethod
    @abstractmethod
    def decode(stream: Stream, *args: Any, **kwargs: Any) -> Parsed[LabeledBlock]:
        ...


class Extension(LabeledBlock):
    @staticmethod
    @abstractmethod
    def decode(stream: Stream, *args: Any, **kwargs: Any) -> Parsed[Extension]:
        ...


@stream_length_at_least(1)
def decode_labeled_block(stream: Stream, graphic_control_extension: Optional[GraphicControlExtension]) -> Parsed[LabeledBlock]:
    block_type = stream[0]
    stream = stream[1:]

    if block_type == BlockType.IMAGE:
        return Image.decode(stream, graphic_control_extension)
    if block_type == BlockType.EXTENSION:
        return decode_extension(stream)
    if block_type == BlockType.TRAILER:
        return Trailer.decode(stream)

    raise ParseError(f"unknown section type: 0x{block_type:02X}")


@stream_length_at_least(1)
def decode_extension(stream: Stream) -> Parsed[Extension]:
    extension_type = stream[0]
    stream = stream[1:]

    try:
        extension_class: Extension = {
            ExtensionType.PLAIN_TEXT:      PlainTextExtension,
            ExtensionType.GRAPHIC_CONTROL: GraphicControlExtension,
            ExtensionType.COMMENT:         CommentExtension,
            ExtensionType.APPLICATION:     ApplicationExtension
        }[extension_type]  # type: ignore[index, assignment]
    except KeyError:
        return UnknownExtension.decode(stream, extension_type)
    else:
        return extension_class.decode(stream)


@dataclass(frozen=True)
class Trailer(LabeledBlock):
    @staticmethod
    def decode(stream: Stream) -> Parsed[Trailer]:
        return stream, Trailer()

    def encode(self) -> bytes:
        return b""


@dataclass(frozen=True)
class UnknownExtension(Extension):
    extension_type: int
    data: DataBlock

    @staticmethod
    def decode(stream: Stream, extension_type: int) -> Parsed[UnknownExtension]:
        stream, data = decode_data_block(stream)
        return stream, UnknownExtension(extension_type, data)

    def encode(self) -> bytes:
        return bytes([BlockType.EXTENSION, self.extension_type]) + encode_data_block(self.data)


@dataclass(frozen=True)
class GraphicControlExtension(Extension):
    disposal_method: int
    waits_for_user_input: int
    delay_time: int
    transparent_color_index: Optional[int]

    @staticmethod
    def decode(stream: Stream) -> Parsed[GraphicControlExtension]:
        stream, block = decode_data_block(stream)
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

        return stream, GraphicControlExtension(disposal_method, waits_for_user_input, delay_time, transparent_color_index)

    def encode(self) -> bytes:
        packed_fields = (self.disposal_method << 2) | (self.waits_for_user_input << 1)
        if self.transparent_color_index is not None:
            packed_fields |= 1 << 0  # has transparent color

        return bytes([BlockType.EXTENSION, ExtensionType.GRAPHIC_CONTROL]) \
             + encode_data_block((pack("<BHB", packed_fields, self.delay_time, self.transparent_color_index or 0),))


@dataclass(frozen=True)
class ApplicationExtension(Extension):
    identifier: bytes
    authentication_code: bytes
    data: DataBlock

    @staticmethod
    def decode(stream: Stream) -> Parsed[ApplicationExtension]:
        stream, block = decode_data_block(stream)
        if len(block) < 1:
            raise ParseError("empty application extension")

        if len(block[0]) != 11:
            raise ParseError("invalid initial application extension block size")

        identifier = block[0][:8]
        authentication_code = block[0][8:]

        return stream, ApplicationExtension(identifier, authentication_code, block[1:])

    def encode(self) -> bytes:
        return bytes([BlockType.EXTENSION, ExtensionType.APPLICATION]) \
             + encode_data_block((self.identifier + self.authentication_code, *self.data))


@dataclass(frozen=True)
class CommentExtension(Extension):
    data: tuple[str, ...]

    @staticmethod
    def decode(stream: Stream) -> Parsed[CommentExtension]:
        stream, block = decode_data_block(stream)

        return stream, CommentExtension(tuple(map(partial(bytes.decode, encoding="ASCII"), block)))

    def encode(self) -> bytes:
        return bytes([BlockType.EXTENSION, ExtensionType.COMMENT]) \
             + encode_data_block(tuple(map(partial(str.encode, encoding="ASCII"), self.data)))


@dataclass(frozen=True)
class PlainTextExtension(Extension):
    top: int
    left: int
    width: int
    height: int
    cell_width: int
    cell_height: int
    foreground_color_index: int
    background_color_index: int
    data: DataBlock

    @staticmethod
    def decode(stream: Stream) -> Parsed[PlainTextExtension]:
        stream, block = decode_data_block(stream)
        top, left, width, height, cell_width, cell_height, foreground_color_index, background_color_index = unpack("<HHHHBBBB", block[0])

        return stream, PlainTextExtension(top, left, width, height, cell_width, cell_height, foreground_color_index, background_color_index, block[1:])

    def encode(self) -> bytes:
        return bytes([BlockType.EXTENSION, ExtensionType.PLAIN_TEXT]) \
             + encode_data_block((pack("<HHHHBBBB", self.top, self.left, self.width, self.height, self.cell_width, self.cell_height, self.foreground_color_index, self.background_color_index), *self.data))


@dataclass(frozen=True)
class Image(LabeledBlock):
    graphic_control_extension: Optional[GraphicControlExtension]
    left: int
    top: int
    width: int
    height: int
    color_table: Optional[ColorTable]
    is_interlaced: bool
    minimum_code_size: int
    data: DataBlock

    def get_pixels(self, global_color_table: Optional[ColorTable]=None) -> list[list[Color]]:
        color_table = global_color_table if self.color_table is None else self.color_table
        data = lzw.decode(self.minimum_code_size, bytes(chain.from_iterable(self.data)))

        if color_table is None:
            raise ValueError("missing active color table")

        colors = [replace(
                     color,
                     is_transparent
                         =   self.graphic_control_extension is not None
                         and i == self.graphic_control_extension.transparent_color_index)
                  for i, color in enumerate(color_table)]

        rows = [[colors[i] for i in row_indices] for row_indices in batched(data, self.width)]

        if not self.is_interlaced:
            return rows

        result: list[list[Color]] = [[]] * len(rows)

        end = 0
        for initial, stride in (0, 8), (4, 8), (2, 4), (1, 2):
            start, end = end, end + len(rows[initial::stride])
            result[initial::stride] = rows[start:end]

        return result


    @staticmethod
    @stream_length_at_least(9)
    def decode(stream: Stream, graphic_control_extension: Optional[GraphicControlExtension]) -> Parsed[Image]:
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

        stream, data = decode_data_block(stream)

        return stream, Image(graphic_control_extension, left, top, width, height, color_table, is_interlaced, minimum_code_size, data)

    def encode(self) -> bytes:
        packed_fields = self.is_interlaced << 6
        if self.color_table is not None:
            packed_fields |=                                              1 << 7  # has color table
            packed_fields |=                     self.color_table.is_sorted << 5
            packed_fields |= ((len(self.color_table) - 1).bit_length() - 1) << 0  # color table size

        result = b""

        if self.graphic_control_extension is not None:
            result += self.graphic_control_extension.encode()

        result += bytes([BlockType.IMAGE]) + pack("<HHHHB", self.left, self.top, self.width, self.height, packed_fields)

        if self.color_table is not None:
            result += self.color_table.encode()

        result += bytes([self.minimum_code_size]) + encode_data_block(self.data)

        return result


@dataclass(frozen=True)
class GIF(Serializable):
    signature: bytes
    version: bytes
    screen: Screen
    blocks: tuple[LabeledBlock, ...]  # blocks after the screen descriptor, not including the trailer

    @staticmethod
    @stream_length_at_least(6)
    def decode(stream: Stream) -> Parsed[GIF]:
        signature, version = stream[:3].tobytes(), stream[3:6].tobytes()
        stream = stream[6:]

        if signature != b"GIF":
            raise ParseError(f"invalid signature: {signature!r}")

        if version not in (b"87a", b"89a"):
            raise ParseError(f"invalid version: {version!r}")

        stream, screen = Screen.decode(stream)

        graphic_control_extension: Optional[GraphicControlExtension] = None
        blocks: list[LabeledBlock] = []
        while True:
            stream, block = decode_labeled_block(stream, graphic_control_extension)
            match block:
                case GraphicControlExtension():
                    graphic_control_extension = block
                case Image():
                    graphic_control_extension = None
                    blocks.append(block)
                case Trailer():
                    break
                case _:
                    blocks.append(block)

        return stream, GIF(signature, version, screen, tuple(blocks))

    def encode(self) -> bytes:
        result = self.signature + self.version
        result += self.screen.encode()
        result += b"".join(map(lambda x: x.encode(), self.blocks))
        result += bytes([BlockType.TRAILER])

        return result
