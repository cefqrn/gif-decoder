# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from collections.abc import Iterable, Sequence, Generator
from itertools import chain, islice
from enum import Enum, auto

MAX_CODE_SIZE = 12


class Bitstream(Sequence[bool]):
    def __init__(self, data: bytes):
        self.data = data

    def __getitem__(self, i):
        byte, offset = divmod(i, 8)
        return (self.data[byte] >> offset) & 1

    def __len__(self):
        return len(self.data) * 8


def bits_to_int(bits: Iterable[bool]) -> int:
    result = 0
    for bit in reversed(list(bits)):
        result = (result << 1) | bit

    return result


class SpecialEntry(Enum):
    CLEAR = auto()
    END = auto()


type Entry = list[int] | SpecialEntry


def create_dictionary(minimum_code_size: int) -> list[Entry]:
    return [[x] for x in range(2**minimum_code_size)] + [SpecialEntry.CLEAR, SpecialEntry.END]


def decode(minimum_code_size: int, data: bytes) -> Generator[int, None, None]:
    stream = iter(Bitstream(data))

    dictionary = create_dictionary(minimum_code_size)

    prev = None
    while True:
        code_size = len(dictionary).bit_length()
        code_index = bits_to_int(islice(stream, code_size))

        if code_index < len(dictionary):
            code = dictionary[code_index]
        elif code_index == len(dictionary) and prev is not None:
            code = prev + prev[:1]
        else:
            raise ValueError(f"code index out of range: {code_index:0{code_size}b}")

        if code is SpecialEntry.END:
            return

        if code is SpecialEntry.CLEAR:
            dictionary = create_dictionary(minimum_code_size)
            prev = None
            continue

        if prev is not None and (len(dictionary) + 1).bit_length() <= MAX_CODE_SIZE:
            dictionary.append(list(chain(prev, code[:1])))

        yield from code
        prev = code
