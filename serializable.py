from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

from collections.abc import Callable
from typing import Any, Protocol


type Stream = memoryview
type Parsed[T] = tuple[Stream, T]


class ParseError(ValueError): ...


class Serializable(ABC):
    @staticmethod
    @abstractmethod
    def decode(stream: Stream, *args: Any, **kwargs: Any) -> Parsed[Serializable]:
        ...

    @abstractmethod
    def encode(self) -> bytes:
        ...


class Parser[**P, R](Protocol):
    def __call__(self, stream: Stream, *args: P.args, **kwargs: P.kwargs) -> R:
        ...


def stream_length_at_least[**P, R](length: int) -> Callable[[Parser[P, R]], Parser[P, R]]:
    def wrapper(f: Parser[P, R]) -> Parser[P, R]:
        @wraps(f)
        def inner(stream: Stream, *args: P.args, **kwargs: P.kwargs) -> R:
            if len(stream) < length:
                raise ParseError("unexpected end of stream")

            return f(stream, *args, **kwargs)

        return inner

    return wrapper
