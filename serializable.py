from abc import ABC, abstractmethod
from functools import wraps

type Parsed[T] = tuple[memoryview, T]


class ParseError(ValueError): ...


class Serializable(ABC):
    @staticmethod
    @abstractmethod
    def decode(stream: memoryview, *args, **kwargs):
        ...

    @abstractmethod
    def encode(self) -> bytes:
        ...


def stream_length_at_least(length):
    def wrapper(f):
        @wraps(f)
        def inner(stream, *args, **kwargs):
            if len(stream) < length:
                raise ParseError("unexpected end of stream")

            return f(stream, *args, **kwargs)

        return inner

    return wrapper
