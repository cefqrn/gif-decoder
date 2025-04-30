# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from collections.abc import Iterable, Generator, Sequence
from dataclasses import dataclass, field
from itertools import batched, chain, islice
from enum import Enum, auto

MAX_CODE_SIZE = 12


class SpecialEntry(Enum):
    CLEAR = auto()
    END = auto()


type Entry = list[int] | SpecialEntry


def bits_to_int(bits: Iterable[bool]) -> int:
    result = 0
    for bit in bits:
        result = 2*result + bit

    return result


def create_dictionary(minimum_code_size: int) -> list[Entry]:
    return [[x] for x in range(2**minimum_code_size)] + [SpecialEntry.CLEAR, SpecialEntry.END]


def byte_to_bits(byte: int) -> list[bool]:
    return [(byte >> i) & 1 for i in range(8)]


def decode(minimum_code_size: int, data: bytes) -> Generator[int, None, None]:
    bitstream = chain.from_iterable(map(byte_to_bits, data))
    dictionary = create_dictionary(minimum_code_size)

    prev = None
    while True:
        code_size = len(dictionary).bit_length()
        code_index = bits_to_int(reversed(list(islice(bitstream, code_size))))

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


@dataclass
class TrieNode:
    code: int
    children: dict[TrieNode] = field(default_factory=dict)


def encode_to_codes(dictionary: Sequence[int], data: Iterable[int]) -> Generator[int, None, None]:
    code_count = len(dictionary)
    if not code_count or code_count & (code_count - 1):
        raise ValueError("dictionary length must be power of 2")

    clear_code = code_count
    end_code = code_count + 1

    code_count += 2

    yield (code_count - 1).bit_length(), clear_code

    root = TrieNode(0)
    for code, entry in enumerate(dictionary):
        root.children[entry] = TrieNode(code)

    curr = root
    for entry in data:
        if entry in curr.children:
            curr = curr.children[entry]
        else:
            yield (code_count - 1).bit_length(), curr.code

            if (code_count + 1).bit_length() <= MAX_CODE_SIZE:
                curr.children[entry] = TrieNode(code_count)
                code_count += 1

            curr = root.children[entry]

    yield (code_count - 1).bit_length(), curr.code
    yield (code_count - 1).bit_length(), end_code


def code_to_bits(code_size: int, x: int) -> list[bool]:
    return [(x >> i) & 1 for i in range(code_size)][::-1]


def encode(dictionary: Sequence[int], data: Iterable[int]) -> Generator(int, None, None):
    bitstream = chain.from_iterable(reversed(code_to_bits(code_size, code)) for code_size, code in encode_to_codes(dictionary, data))
    for batch in batched(chain(bitstream, 7*[False]), 8):
        batch = batch[::-1]
        if len(batch) < 8:
            break

        yield bits_to_int(batch)
