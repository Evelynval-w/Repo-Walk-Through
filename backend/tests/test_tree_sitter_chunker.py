from app.ingest.chunkers.tree_sitter_chunker import chunk_with_tree_sitter
from tests.conftest import shape

JAVA = """package demo;

import java.util.List;

/** A booking. */
@Entity
public class Booking {
    private int id;

    /** Returns the id. */
    public int getId() {
        return id;
    }

    public Booking() {
    }

    @Override
    public String toString() {
        return "b";
    }
}
"""

JS = """import express from "express";

const router = express.Router();

/** List users. */
router.get("/users", async (req, res) => {
  res.json([]);
});

router.post("/users/:id", handler);

export function helper(a) {
  return a;
}

const arrow = (x) => x * 2;

export default class Service {
  start() {
    return 1;
  }
  stop() {
    return 2;
  }
}
"""

TSX = """interface Props {
  name: string;
}

export type Id = string;

export const Hello = ({ name }: Props) => {
  return <div>{name}</div>;
};
"""


def test_java_class_is_cut_into_header_and_methods():
    chunks = chunk_with_tree_sitter("demo", "Booking.java", JAVA, ".java")
    assert shape(chunks) == [
        ("top-level code", "module", 1, 3),  # package and import
        ("Booking", "class", 5, 8),  # Javadoc, annotation, signature, field
        ("Booking.getId", "method", 10, 13),  # includes its Javadoc
        ("Booking.Booking", "method", 15, 16),  # constructor
        ("Booking.toString", "method", 18, 22),  # includes @Override and the closing brace
    ]
    assert all(c.language == "java" for c in chunks)


def test_javascript_functions_classes_and_routes():
    chunks = chunk_with_tree_sitter("demo", "routes.js", JS, ".js")
    assert shape(chunks) == [
        ("top-level code", "module", 1, 3),
        ("GET /users", "route", 5, 8),  # includes the comment above it
        ("POST /users/:id", "route", 10, 10),
        ("helper", "function", 12, 14),
        ("arrow", "function", 16, 16),
        ("Service", "class", 18, 18),
        ("Service.start", "method", 19, 21),
        ("Service.stop", "method", 22, 25),
    ]


def test_comment_at_the_end_of_a_line_is_not_taken_by_the_next_statement():
    code = 'app.use("/a", one); // mounts a\napp.use("/b", two);\n'
    chunks = chunk_with_tree_sitter("demo", "index.js", code, ".js")
    assert shape(chunks) == [("USE /a", "route", 1, 1), ("USE /b", "route", 2, 2)]


def test_typescript_interface_type_and_component():
    chunks = chunk_with_tree_sitter("demo", "Hello.tsx", TSX, ".tsx")
    assert shape(chunks) == [
        ("Props", "interface", 1, 3),
        ("Id", "type", 5, 5),
        ("Hello", "function", 7, 9),
    ]
    assert all(c.language == "typescript" for c in chunks)


def test_file_with_only_top_level_code_is_not_lost():
    chunks = chunk_with_tree_sitter("demo", "a.js", "const a = 1;\nconsole.log(a);\n", ".js")
    assert shape(chunks) == [("top-level code", "module", 1, 2)]
