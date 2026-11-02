"""A simple check for keys and tokens, so they are never embedded or sent to a model.

This is a safety net, not a security scanner. It catches the common key shapes.
It returns the NAME of the pattern that matched, never the value itself.
"""
from __future__ import annotations

import re

PATTERNS = {
    "google api key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    # (?<![A-Za-z0-9]) stops matches inside words like "task-based-..."
    "sk- style api key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{20,}"),
    "github token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "aws access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "slack token": re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def find_secret(text: str) -> str | None:
    for label, pattern in PATTERNS.items():
        if pattern.search(text):
            return label
    return None


# ---------------------------------------------------------------- redaction
# Passwords in config files (application.properties, docker-compose.yml, .env.example)
# do not have a special shape. We find them by the NAME of the setting and
# replace the value with <redacted>. The chunk keeps its place and its line
# numbers, only the value is hidden.

SECRET_NAME = r"(?:password|passwd|secret|token|api[_-]?key|private[_-]?key)"
CONFIG_LINE = re.compile(
    rf"(?im)^(?P<head>[ \t\-]*(?P<key>[\w.\-/]*{SECRET_NAME}[\w.\-]*)[ \t]*[=:][ \t]*)(?P<value>\S.*?)[ \t]*$"
)
# In code we only touch a quoted literal that is assigned directly: name = "value"
CODE_LITERAL = re.compile(
    rf"(?i)(?P<head>\b(?P<key>\w*{SECRET_NAME}\w*)[ \t]*(?<![=!<>])[:=](?!=)[ \t]*)(?P<quote>[\"'])(?P<value>[^\"'\s]{{6,}})(?P=quote)"
)
CONFIG_LANGUAGES = {"yaml", "properties", "env-example", "toml", "ini", "json", "dockerfile", "shell", "xml"}
HARMLESS_KEY_WORDS = ("url", "path", "expire", "minute", "second", "algorithm", "header", "endpoint", "timeout", "length", "enabled")
PLACEHOLDER = re.compile(r"(?i)(change|replace|your|example|placeholder|dummy|xxx|todo)")
REDACTED = "<redacted>"


def _is_placeholder(key: str, value: str) -> bool:
    bare = value.strip().strip("\"'")
    if not bare or bare == REDACTED:
        return True
    if bare.startswith(("${", "$", "<", "{{")):  # a reference to another variable
        return True
    if bare.lower() in {"true", "false", "null", "none", "~"} or bare.isdigit():
        return True
    if any(word in key.lower() for word in HARMLESS_KEY_WORDS):
        return True
    return bool(PLACEHOLDER.search(bare))


def redact_secret_values(text: str, config_like: bool) -> tuple[str, int]:
    """Hide values of settings that are named like a secret. Returns (new text, how many)."""
    count = 0

    def config_sub(match: re.Match) -> str:
        nonlocal count
        if _is_placeholder(match.group("key"), match.group("value")):
            return match.group(0)
        count += 1
        return match.group("head") + REDACTED

    def code_sub(match: re.Match) -> str:
        nonlocal count
        if _is_placeholder(match.group("key"), match.group("value")):
            return match.group(0)
        count += 1
        quote = match.group("quote")
        return f"{match.group('head')}{quote}{REDACTED}{quote}"

    pattern, replace = (CONFIG_LINE, config_sub) if config_like else (CODE_LITERAL, code_sub)
    return pattern.sub(replace, text), count
