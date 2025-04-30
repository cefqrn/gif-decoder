# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from collections.abc import Iterable, Generator
from itertools import chain, islice
from enum import Enum, auto

MAX_CODE_SIZE = 12


class SpecialEntry(Enum):
    CLEAR = auto()
    END = auto()


type Entry = list[int] | SpecialEntry


def bits_to_int(bits: Iterable[bool]) -> int:
    result = 0
    for bit in reversed(list(bits)):
        result = 2*result + bit

    return result


def create_dictionary(minimum_code_size: int) -> list[Entry]:
    return [[x] for x in range(2**minimum_code_size)] + [SpecialEntry.CLEAR, SpecialEntry.END]


def decode(minimum_code_size: int, data: Iterable[bool]) -> Generator[int, None, None]:
    dictionary = create_dictionary(minimum_code_size)

    prev = None
    while True:
        code_size = len(dictionary).bit_length()
        code_index = bits_to_int(islice(data, code_size))

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
