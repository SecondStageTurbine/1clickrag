# SPDX-License-Identifier: MPL-2.0
"""Every setting, its value now, and where that value came from.

The configuration surface grew a long way past what anyone will read a file to
discover. Someone wondering why search feels wrong should be able to see that
reranking is off, or that OCR was never switched on, without opening .env,
remembering which of forty variables governs it, and knowing that an absent
line means a default they also have to remember.

Nothing here is hand-maintained, because a hand-maintained list of settings is
wrong within a month. The fields come from the Config dataclass, so a setting
added there appears here; the prose comes from .env.example, which already
documents every variable and is already shipped; and the values come from the
running Config, so what is shown is what the process is actually using rather
than what the file says.

Two things it will not do. It does not show secrets - an API key is reported as
present or absent and never echoed into a browser tab that might be screen-
shared. And it is read-only: changing a setting here would mean writing .env
and restarting from inside a request, which is a great deal of machinery for
something a text editor already does, and it would let a stray request take the
index offline.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import Any

from . import manifest as manifest_module

# Attributes whose environment variable does not follow RAG_<ATTRIBUTE>.
EXCEPTIONS = {
    "repo_path": "RAG_REPO_MOUNT",
}

# Settings that are computed rather than read, so "not set" is the normal state
# and the effective value is the interesting one.
DERIVED = {"watch", "rescan_minutes", "exclude_dirs", "exclude_globs"}

SECTION = re.compile(r"^#\s*-{2,}\s*(.+?)\s*-{2,}\s*$")
ASSIGNMENT = re.compile(r"^#?\s*(RAG_[A-Z0-9_]+)\s*=(.*)$")


def env_name(attribute: str) -> str:
    return EXCEPTIONS.get(attribute, f"RAG_{attribute.upper()}")


def _is_secret(attribute: str) -> bool:
    """Whether a setting's value must never be printed.

    Matched on whole words rather than substrings, because a substring rule
    redacts things that are not secrets: "token" appears in
    `chat_max_tokens`, whose value is 4096 and whose redaction only hides a
    number someone needs. Hiding a real setting is a quieter failure than
    showing one, so the rule is narrow and explicit.
    """
    parts = attribute.lower().split("_")
    if "password" in parts or "secret" in parts or "credential" in parts:
        return True
    if attribute.lower().endswith("api_key") or attribute.lower().endswith("_key"):
        return True
    # A singular "token" is a credential; "tokens" is a count of them.
    return parts[-1] == "token"


def _documentation(path: str) -> dict[str, dict]:
    """Descriptions and documented defaults, read from .env.example.

    That file is the canonical explanation of every variable and ships beside
    the code, so it is the natural source rather than a second copy that would
    drift from it.
    """
    docs: dict[str, dict] = {}
    if not os.path.isfile(path):
        return docs
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return docs

    section = ""
    prose: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        heading = SECTION.match(line)
        if heading:
            section = heading.group(1)
            prose = []
            continue
        if not line.strip():
            prose = []
            continue
        assignment = ASSIGNMENT.match(line)
        if assignment:
            name, default = assignment.group(1), assignment.group(2).strip()
            docs.setdefault(name, {
                "section": section,
                # The first paragraph carries the point; the rest is detail
                # that belongs in the file rather than a table cell.
                "description": " ".join(prose).strip(),
                "documented_default": default,
            })
            prose = []
            continue
        if line.startswith("#"):
            prose.append(line.lstrip("#").strip())
    return docs


def _shown(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return value


def snapshot(cfg, example_path: str = "") -> dict:
    """Every setting with its value, source and effect."""
    example = example_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env.example"
    )
    docs = _documentation(example)

    # No "default" column, deliberately. The dataclass field default looks like
    # one and is not: it is evaluated at import, after .env has been read into
    # the environment, so for every env-backed setting it is just the current
    # value under another name - RAG_OCR showed "default True" on a machine
    # that had switched it on, when the built-in default is False.
    # .env.example is no better a source: its lines are examples of how to set
    # a thing (#RAG_OCR=1), not statements of what it is when unset.
    # What is both true and useful is whether the value was set explicitly, so
    # that is what is reported.
    rebuild = set(manifest_module.REBUILD_FIELDS)
    content = set(manifest_module.CONTENT_FIELDS)

    items: list[dict] = []
    for field in dataclasses.fields(cfg):
        name = field.name
        variable = env_name(name)
        value = getattr(cfg, name)
        raw = os.environ.get(variable)
        secret = _is_secret(name)

        entry = {
            "setting": variable,
            "attribute": name,
            "value": "(set)" if secret and value else ("(not set)" if secret else _shown(value)),
            "secret": secret,
            # .env is loaded into the environment at import, so this covers a
            # value from either place - which is what "did I change this"
            # actually means to someone reading it.
            "set_explicitly": raw is not None,
            # Computed rather than read - a path built from another setting, a
            # thread count from the core count, a watcher default that depends
            # on whether the corpus is a UNC path. "Not set" is normal for
            # these and the effective value is the whole story.
            "derived": name in DERIVED,
            "section": docs.get(variable, {}).get("section", "other"),
            "description": docs.get(variable, {}).get("description", ""),
        }
        if name in rebuild:
            entry["changing_it"] = "needs `reindex -Full` - it changes every vector"
        elif name in content:
            entry["changing_it"] = "applies to files as they are next indexed"
        items.append(entry)

    items.sort(key=lambda item: (item["section"] or "other", item["setting"]))
    sections: dict[str, list[dict]] = {}
    for item in items:
        sections.setdefault(item["section"] or "other", []).append(item)

    return {
        "count": len(items),
        "changed": sum(1 for item in items if item["set_explicitly"]),
        "env_file": os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        ),
        "sections": [
            {"name": name, "settings": entries} for name, entries in sections.items()
        ],
    }
