import queue
from typing import Generator, Any

SENTINEL = object()


def queue_to_generator(q: queue.Queue, sentinel: Any = None) -> Generator:
    """Wrap a thread-safe queue as a lazy generator. Stops when sentinel is received."""
    while True:
        item = q.get()
        if item is sentinel:
            return
        yield item
