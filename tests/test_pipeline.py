import queue
from pipeline import queue_to_generator, SENTINEL


def test_yields_items_until_sentinel():
    q = queue.Queue()
    q.put('a')
    q.put('b')
    q.put(SENTINEL)
    assert list(queue_to_generator(q, sentinel=SENTINEL)) == ['a', 'b']


def test_empty_queue_with_immediate_sentinel():
    q = queue.Queue()
    q.put(SENTINEL)
    assert list(queue_to_generator(q, sentinel=SENTINEL)) == []


def test_sentinel_is_not_yielded():
    q = queue.Queue()
    q.put(42)
    q.put(SENTINEL)
    result = list(queue_to_generator(q, sentinel=SENTINEL))
    assert SENTINEL not in result
