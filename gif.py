from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from struct import unpack
from typing import Optional
from enum import Enum

type Parsed[T] = Optional[tuple[bytes, T]]


class ParseError(Exception): ...


@dataclass
class Screen:
    width: int
    height: int
    global_color_table_size: int


@dataclass
class Image:
    local_color_table_size: int


class SectionType(Enum):
    IMAGE     = 0x2C
    EXTENSION = 0x21
    TRAILER   = 0x3B


class ExtensionType(Enum):
    PLAIN_TEXT      = 0x01
    GRAPHIC_CONTROL = 0xF9
    COMMENT         = 0xFE
    APPLICATION     = 0xFF


def stream_length_at_least(length):
    def wrapper(f):
        @wraps(f)
        def inner(stream, *args, **kwargs):
            if len(stream) < length:
                raise ParseError("unexpected end of stream")

            return f(stream, *args, **kwargs)

        return inner

    return wrapper


@stream_length_at_least(6)
def parse_header(stream: bytes) -> Parsed[None]:
    # doesn't handle GIF87a
    if stream[:6] != b"GIF89a":
        raise ParseError("cannot handle stream (incorrect header)")

    return stream[6:], None


@stream_length_at_least(7)
def parse_logical_screen_descriptor(stream: bytes) -> Parsed[Screen]:
    image_width, image_height, packed_fields, *_ = unpack("<HHBBB", stream[:7])

    has_global_color_table = bool(packed_fields >> 7)
    global_color_table_size = 3 * 2 ** ((packed_fields & 7) + 1)

    return stream[7:], Screen(
        image_width,
        image_height,
        global_color_table_size if has_global_color_table else 0
    )


def skip_bytes(stream: bytes, offset: int) -> Parsed[None]:
    if len(stream) < offset:
        raise ParseError("unexpected end of stream")

    return stream[offset:], None


def parse_section_type(stream: bytes) -> Parsed[SectionType]:
    try:
        return stream[1:], SectionType(stream[0])
    except ValueError:
        raise ParseError(f"unknown extension type, got 0x{stream[0]:02X}") from None



@stream_length_at_least(1)
def parse_extension_type(stream: bytes) -> Parsed[ExtensionType]:
    try:
        return stream[1:], ExtensionType(stream[0])
    except ValueError:
        raise ParseError(f"unknown extension type, got 0x{stream[0]:02X}") from None


@stream_length_at_least(1)
def parse_sub_block(stream: bytes) -> Parsed[bytes]:
    length = stream[0]
    if len(stream[1:]) < length:
        raise ParseError("unexpected end of stream")

    return stream[1+length:], stream[1:1+length]


def parse_block(stream: bytes) -> Parsed[bytes]:
    data = []
    while True:
        stream, block_data = parse_sub_block(stream)
        if block_data:
            data.append(block_data)
        else:
            break

    return stream, b"".join(data)


def parse_comment_extension(stream: bytes) -> Parsed[str]:
    stream, data = parse_block(stream)
    try:
        return stream, data.decode("ascii")
    except UnicodeError as e:
        raise ParseError("failed to decode comment") from e


@stream_length_at_least(13)
def parse_plain_text_extension(stream: bytes) -> Parsed[None]:
    return stream[13:], None

@stream_length_at_least(12)
def parse_application_extension(stream: bytes) -> Parsed[None]:
    if stream[0] != 11:
        raise ParseError("invalid application extension initial block length")

    identifier, authentication_code = stream[1:9], stream[9:12]
    print("application extension:", identifier, authentication_code)

    stream, _ = parse_block(stream[12:])

    return stream, None

@stream_length_at_least(6)
def parse_graphic_control_extension(stream: bytes) -> Parsed[None]:
    if stream[0] != 4:
        raise ParseError("invalid graphic control extension initial block length")

    if stream[5] != 0:
        raise ParseError("invalid graphic control extension block terminator")

    return stream[6:], None


@stream_length_at_least(9)
def parse_image_descriptor(stream: bytes) -> Parsed[None]:
    *_, packed_fields = unpack("<HHHHB", stream[:9])

    has_local_color_table = bool(packed_fields >> 7)
    local_color_table_size = 3 * 2 ** ((packed_fields & 7) + 1)

    return stream[9:], Image(
        local_color_table_size if has_local_color_table else 0
    )


INPUT_DIRECTORY = Path()
INPUT_FILENAME = "test.gif"

input_path = INPUT_DIRECTORY / INPUT_FILENAME

stream = input_path.read_bytes()
try:
    stream, _      = parse_header(stream)
    stream, screen = parse_logical_screen_descriptor(stream)
    stream, _      = skip_bytes(stream, screen.global_color_table_size)
    while True:
        # print(stream[:20].hex())
        stream, section_type = parse_section_type(stream)
        print(section_type)
        match section_type:
            case SectionType.EXTENSION:
                stream, extension_type = parse_extension_type(stream)
                match extension_type:
                    case ExtensionType.PLAIN_TEXT:
                        stream, _       = parse_plain_text_extension(stream)
                    case ExtensionType.COMMENT:
                        stream, comment = parse_comment_extension(stream)
                        print("comment:", comment)
                    case ExtensionType.APPLICATION:
                        stream, _       = parse_application_extension(stream)
                    case ExtensionType.GRAPHIC_CONTROL:
                        stream, _       = parse_graphic_control_extension(stream)
            case SectionType.IMAGE:
                stream, image = parse_image_descriptor(stream)
                stream, _     = skip_bytes(stream, image.local_color_table_size)
                stream, _     = skip_bytes(stream, 1)
                stream, _     = parse_block(stream)
            case SectionType.TRAILER:
                break
except ParseError as e:
    print("failed to parse:", e)


"""
<GIF Data Stream> ::=     Header <Logical Screen> <Data>* Trailer
<Logical Screen> ::=      Logical Screen Descriptor [Global Color Table]
<Data> ::=                <Graphic Block>  |
                          <Special-Purpose Block>
<Graphic Block> ::=       [Graphic Control Extension] <Graphic-Rendering Block>
<Graphic-Rendering Block> ::=  <Table-Based Image>  |
                               Plain Text Extension
<Table-Based Image> ::=   Image Descriptor [Local Color Table] Image Data
<Special-Purpose Block> ::=    Application Extension  |
                               Comment Extension
"""