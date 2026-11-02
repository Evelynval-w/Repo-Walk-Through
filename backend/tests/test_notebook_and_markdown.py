import json

from app.ingest.chunkers.markdown import chunk_markdown
from app.ingest.chunkers.notebook import chunk_notebook

BIG_IMAGE = "iVBORw0KGgo" + "A" * 5000


def make_notebook():
    return json.dumps(
        {
            "metadata": {"language_info": {"name": "python"}},
            "cells": [
                {"cell_type": "markdown", "source": ["# Load data\n", "Read the loans."]},
                {
                    "cell_type": "code",
                    "source": ["import pandas as pd\n", "df = pd.read_csv('loans.csv')"],
                    "outputs": [{"data": {"image/png": BIG_IMAGE}}],
                },
                {"cell_type": "code", "source": [], "outputs": []},
                {"cell_type": "code", "source": "df.describe()", "outputs": []},
                {"cell_type": "markdown", "source": "Final thoughts."},
            ],
        }
    )


def test_notebook_gives_one_chunk_per_code_cell_and_drops_outputs():
    chunks = chunk_notebook("demo", "nb.ipynb", make_notebook())
    assert [(c.name, c.kind, c.cell) for c in chunks] == [
        ("Load data", "notebook_cell", 2),
        ("cell 4", "notebook_cell", 4),
        ("notes before cell 5", "markdown_section", 5),
    ]
    assert "import pandas as pd" in chunks[0].text
    assert "Read the loans." in chunks[0].text  # the markdown above is kept as context
    assert all("iVBORw0KGgo" not in c.text for c in chunks)  # the image output is gone
    assert chunks[0].start_line is None


def test_broken_notebook_gives_no_chunks():
    assert chunk_notebook("demo", "nb.ipynb", "not json") == []


MARKDOWN = """Intro text

# Title
Some text

## Setup
Run this:

```bash
# not a heading
pip install x
```

## Empty

### Child
Child text
"""


def test_markdown_is_cut_by_heading_and_ignores_code_fences():
    chunks = chunk_markdown("demo", "README.md", MARKDOWN)
    assert [c.name for c in chunks] == [
        "(intro)",
        "Title",
        "Title > Setup",
        "Title > Empty > Child",  # "Empty" has no text of its own, so it is skipped
    ]
    setup = chunks[2]
    assert "# not a heading" in setup.text  # stayed inside the Setup section
