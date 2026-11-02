"""Cut Java, JavaScript, JSX, TypeScript and TSX files with tree-sitter.

tree-sitter builds a syntax tree for each file. We walk the top level of that
tree and turn each class, method, function and route into one chunk. Comments
that sit right above a definition (Javadoc, JSDoc) are kept with it.
"""
from __future__ import annotations

from functools import lru_cache

from app.ingest.chunkers.common import gap_chunks, split_lines, trim_end
from app.ingest.models import Chunk

# file extension -> (grammar name, language label stored on the chunk)
EXTENSIONS = {
    ".java": ("java", "java"),
    ".js": ("javascript", "javascript"),
    ".jsx": ("javascript", "javascript"),
    ".mjs": ("javascript", "javascript"),
    ".cjs": ("javascript", "javascript"),
    ".ts": ("typescript", "typescript"),
    ".tsx": ("tsx", "typescript"),
}

COMMENT_TYPES = {"comment", "line_comment", "block_comment"}
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "use", "all"}
JS_FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
JS_CLASS_TYPES = {"class_declaration", "abstract_class_declaration"}
JS_OTHER_TYPES = {
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
}
JAVA_TYPE_NODES = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "class",
}
JAVA_MEMBER_NODES = {"method_declaration", "constructor_declaration"}


@lru_cache(maxsize=None)
def _parser(grammar: str):
    # Imported here so the rest of the app works even if a grammar is missing.
    from tree_sitter import Language, Parser

    if grammar == "java":
        import tree_sitter_java as module

        language = module.language()
    elif grammar == "javascript":
        import tree_sitter_javascript as module

        language = module.language()
    elif grammar == "typescript":
        import tree_sitter_typescript as module

        language = module.language_typescript()
    elif grammar == "tsx":
        import tree_sitter_typescript as module

        language = module.language_tsx()
    else:
        raise ValueError(f"unknown grammar: {grammar}")
    return Parser(Language(language))


def _text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _name(node, source: bytes, fallback: str = "default") -> str:
    name_node = node.child_by_field_name("name")
    return _text(name_node, source) if name_node is not None else fallback


def _first_row(node) -> int:
    """First row of a node, widened to include comments right above it."""
    row = node.start_point[0]
    previous = node.prev_sibling
    while previous is not None and previous.type in COMMENT_TYPES and previous.end_point[0] >= row - 1:
        before = previous.prev_sibling
        if before is not None and before.end_point[0] == previous.start_point[0]:
            break  # a comment at the end of the line above belongs to that line, not to us
        row = previous.start_point[0]
        previous = before
    return row


class _Collector:
    """Collects chunks and remembers which rows they cover."""

    def __init__(self, repo: str, path: str, language: str, lines: list[str]):
        self.repo, self.path, self.language, self.lines = repo, path, language, lines
        self.chunks: list[Chunk] = []
        self.covered: list[tuple[int, int]] = []

    def add(self, name: str, kind: str, first: int, last: int) -> None:
        last = trim_end(self.lines, first, last)
        self.covered.append((first, last))
        self.chunks.append(
            Chunk(
                repo=self.repo,
                path=self.path,
                name=name,
                kind=kind,
                language=self.language,
                text="\n".join(self.lines[first : last + 1]),
                start_line=first + 1,
                end_line=last + 1,
            )
        )

    def add_container(self, name: str, kind: str, wrapper, members: list[tuple[str, object]]) -> None:
        """A class with members: one chunk for the header, one per member.

        The header is everything above the first member (annotations, signature,
        fields). Whatever follows the last member, such as the closing brace,
        is added to the last member so it does not become a chunk of its own.
        """
        class_first = _first_row(wrapper)
        class_last = wrapper.end_point[0]
        if not members:
            self.add(name, kind, class_first, class_last)
            return
        header_last = _first_row(members[0][1]) - 1
        if header_last >= class_first:
            self.add(name, kind, class_first, header_last)
        for index, (member_name, member) in enumerate(members):
            last = class_last if index == len(members) - 1 else member.end_point[0]
            self.add(f"{name}.{member_name}", "method", _first_row(member), last)

    def finish(self) -> list[Chunk]:
        self.chunks.extend(gap_chunks(self.lines, self.covered, self.repo, self.path, self.language))
        self.chunks.sort(key=lambda c: c.start_line or 0)
        return self.chunks


# ----------------------------------------------------------------- Java

def _java_members(body, source: bytes) -> list[tuple[str, object]]:
    if body is None:
        return []
    members = []
    for child in body.named_children:
        if child.type in JAVA_MEMBER_NODES:
            members.append((_name(child, source), child))
        elif child.type == "enum_body_declarations":  # methods inside an enum
            members.extend(_java_members(child, source))
    return members


def _chunk_java(root, source: bytes, out: _Collector) -> None:
    for node in root.named_children:
        kind = JAVA_TYPE_NODES.get(node.type)
        if kind is None:
            continue
        members = _java_members(node.child_by_field_name("body"), source)
        out.add_container(_name(node, source), kind, node, members)


# ------------------------------------------------------ JavaScript / TypeScript

def _route_name(node, source: bytes) -> str | None:
    """Name for an Express-style route such as router.get('/users', handler)."""
    if node.type != "expression_statement" or not node.named_children:
        return None
    call = node.named_children[0]
    if call.type != "call_expression":
        return None
    function = call.child_by_field_name("function")
    arguments = call.child_by_field_name("arguments")
    if function is None or arguments is None or function.type != "member_expression":
        return None
    prop = function.child_by_field_name("property")
    method = _text(prop, source) if prop is not None else ""
    if method not in HTTP_METHODS or not arguments.named_children:
        return None
    first_arg = arguments.named_children[0]
    if first_arg.type not in ("string", "template_string"):
        return None
    path = _text(first_arg, source).strip("'\"`")
    return f"{method.upper()} {path}"


def _js_members(class_node, source: bytes) -> list[tuple[str, object]]:
    body = class_node.child_by_field_name("body")
    if body is None:
        return []
    return [
        (_name(child, source), child)
        for child in body.named_children
        if child.type == "method_definition"
    ]


def _chunk_js(root, source: bytes, out: _Collector) -> None:
    for node in root.named_children:
        route = _route_name(node, source)
        if route is not None:
            out.add(route, "route", _first_row(node), node.end_point[0])
            continue

        inner = node
        if node.type == "export_statement":
            inner = node.child_by_field_name("declaration") or node.child_by_field_name("value")
            if inner is None:
                continue

        if inner.type in ("function_declaration", "generator_function_declaration"):
            out.add(_name(inner, source), "function", _first_row(node), node.end_point[0])
        elif inner.type in JS_CLASS_TYPES:
            out.add_container(_name(inner, source), "class", node, _js_members(inner, source))
        elif inner.type in JS_OTHER_TYPES:
            out.add(_name(inner, source), JS_OTHER_TYPES[inner.type], _first_row(node), node.end_point[0])
        elif inner.type in ("lexical_declaration", "variable_declaration"):
            for declarator in inner.named_children:
                value = declarator.child_by_field_name("value")
                if declarator.type == "variable_declarator" and value is not None and value.type in JS_FUNCTION_VALUES:
                    out.add(_name(declarator, source), "function", _first_row(node), node.end_point[0])
                    break
        elif inner.type in JS_FUNCTION_VALUES:  # export default () => ...
            out.add("default", "function", _first_row(node), node.end_point[0])
        # everything else is handled as a gap


def chunk_with_tree_sitter(repo: str, path: str, text: str, extension: str) -> list[Chunk]:
    grammar, language = EXTENSIONS[extension]
    source = text.encode("utf-8")
    root = _parser(grammar).parse(source).root_node
    lines = split_lines(text)
    out = _Collector(repo, path, language, lines)
    if grammar == "java":
        _chunk_java(root, source, out)
    else:
        _chunk_js(root, source, out)
    return out.finish()
