from abc import ABC, abstractmethod
from functools import wraps
from typing import Self

type Parsed[T] = tuple[bytes, T]


class ParseError(ValueError): ...


class Serializable(ABC):
    @classmethod
    @abstractmethod
    def decode(cls, stream: bytes, *args, **kwargs) -> Parsed[Self]:
        ...

    @abstractmethod
    def encode(self) -> bytes:
        ...


def stream_length_at_least(length):
    def wrapper(f):
        @wraps(f)
        def inner(cls, stream, *args, **kwargs):
            if len(stream) < length:
                raise ParseError("unexpected end of stream")

            return f(cls, stream, *args, **kwargs)

        return inner

    return wrapper
