from app.ingest.chunkers import chunk_file
from app.ingest.chunkers.python_ast import chunk_python
from tests.conftest import shape

SAMPLE = '''"""Module doc."""
import os

X = 1


def top(a):
    """Add one."""
    return a + 1


@decorator
async def fetch():
    pass


class Greeter:
    """Says hello."""

    name = "x"

    def hello(self):
        return "hi"

    def bye(self):
        return "bye"


if __name__ == "__main__":
    top(1)
'''


def test_python_file_is_cut_along_its_structure():
    chunks = chunk_python("demo", "app.py", SAMPLE)
    assert shape(chunks) == [
        ("top-level code", "module", 1, 4),
        ("top", "function", 7, 9),
        ("fetch", "function", 12, 14),  # starts at the decorator, not at "async def"
        ("Greeter", "class", 17, 20),  # signature, docstring, attributes
        ("Greeter.hello", "method", 22, 23),
        ("Greeter.bye", "method", 25, 26),
        ("top-level code", "module", 29, 30),
    ]


def test_chunk_text_matches_the_line_numbers():
    chunks = {c.name: c for c in chunk_python("demo", "app.py", SAMPLE)}
    assert chunks["top"].text.startswith("def top(a):")
    assert chunks["top"].text.endswith("return a + 1")
    assert chunks["fetch"].text.startswith("@decorator")


def test_ids_are_unique():
    chunks = chunk_python("demo", "app.py", SAMPLE)
    assert len({c.id for c in chunks}) == len(chunks)


def test_broken_python_is_kept_as_one_block():
    chunks = chunk_python("demo", "bad.py", "def (:\n    nope\n")
    assert len(chunks) == 1 and chunks[0].kind == "file"


def test_class_without_methods_is_one_chunk():
    chunks = chunk_python("demo", "m.py", "class Row:\n    a = 1\n    b = 2\n")
    assert shape(chunks) == [("Row", "class", 1, 3)]


def test_long_function_is_split_without_gaps():
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    chunks = chunk_file("demo", "big.py", "def big():\n" + body + "\n", max_chars=500)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert chunks[0].start_line == 1
    for before, after in zip(chunks, chunks[1:]):
        assert after.start_line == before.end_line + 1
