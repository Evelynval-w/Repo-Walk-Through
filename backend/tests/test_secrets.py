from app.ingest.pipeline import chunk_repo
from app.ingest.secrets import REDACTED, find_secret, redact_secret_values
from app.ingest.walker import RepoConfig

# Built from pieces so this file holds no real-looking secrets.
PROPS = "spring.datasource.url=jdbc:mysql://localhost/db\nspring.datasource.password=" + "hunter" + "2x\n"


def test_password_in_a_properties_file_is_hidden_and_the_rest_is_kept():
    text, count = redact_secret_values(PROPS, config_like=True)
    assert count == 1
    assert "hunter" not in text
    assert "spring.datasource.password=<redacted>" in text
    assert "jdbc:mysql://localhost/db" in text


def test_yaml_and_env_lines_are_hidden_too():
    text = "environment:\n  - DB_PASSWORD=" + "s3cr3t" + "pass\n  POSTGRES_PASSWORD: " + "pg" + "pass99\n"
    out, count = redact_secret_values(text, config_like=True)
    assert count == 2
    assert "s3cr3t" not in out and "pg" + "pass99" not in out


def test_placeholders_numbers_and_variable_references_are_left_alone():
    text = "\n".join(
        [
            "DB_PASSWORD=${DB_PASSWORD}",
            "API_KEY=your-key-here",
            "TOKEN_EXPIRE_MINUTES=30",
            "SECRET_KEY=change-me",
            "JWT_ALGORITHM_SECRET_FLAG=true",
            "PASSWORD=",
        ]
    )
    out, count = redact_secret_values(text, config_like=True)
    assert count == 0
    assert out == text


def test_code_string_literal_is_hidden_but_comparisons_are_not_touched():
    code = 'SECRET_KEY = "' + "abc" + 'def123456"\nif password == "' + "letmein" + '":\n    pass\n'
    out, count = redact_secret_values(code, config_like=False)
    assert count == 1
    assert 'SECRET_KEY = "<redacted>"' in out
    assert 'if password == "letmein":' in out  # a comparison, left as it was


def test_code_that_reads_a_variable_is_not_changed():
    code = 'password = request.form["password"]\ntoken = os.getenv("TOKEN")\n'
    out, count = redact_secret_values(code, config_like=False)
    assert count == 0 and out == code


def test_pipeline_reports_redactions_and_stores_hidden_text(tmp_path, settings):
    root = tmp_path / "demo"
    (root / "src" / "main" / "resources").mkdir(parents=True)
    (root / "src" / "main" / "resources" / "application.properties").write_text(PROPS, encoding="utf-8")

    result = chunk_repo(RepoConfig("demo"), root, settings)

    assert len(result.chunks) == 1
    assert "hunter" not in result.chunks[0].text
    assert len(result.redacted) == 1
    assert "hunter" not in result.redacted[0]  # the report never repeats the value


def test_find_secret_still_catches_key_shapes():
    assert find_secret("x = '" + "AIza" + "C" * 35 + "'") == "google api key"
    assert find_secret("nothing to see here") is None
