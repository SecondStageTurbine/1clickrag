# SPDX-License-Identifier: MPL-2.0
"""Server-owned chat policies and deterministic retrieval plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


POLICY_DIR = Path(__file__).with_name("policies")
COMMON_POLICY = (POLICY_DIR / "common.txt").read_text(encoding="utf-8").strip()


def _policy(name: str) -> str:
    # Fail visibly at startup if an update omitted a required policy file.
    return (POLICY_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class ChatMode:
    id: str
    label: str
    description: str
    policy: str
    min_top_k: int = 0
    min_hops: int = 0
    min_context_chars: int = 0
    force_expand: bool = False
    follow_references: bool = False
    peer_review: bool = False


GENERAL = ChatMode(
    id="general",
    label="General document Q&A",
    description="Answer from the passages retrieved for the user's question.",
    policy="",
)

FAULT_TRACE = ChatMode(
    id="fault_trace",
    label="Supplier fault-code maintenance trace",
    description=(
        "Trace supplier fault codes through preparation, figures, blocks, "
        "diagram branches, cross-sheet connectors, and corrective actions."
    ),
    policy="""
Task policy: supplier fault-code maintenance trace

Given one or more supplier fault codes, identify each exact code and its
description, required preparation, referenced figure, starting block, and
starting sheet. Follow explicit diagram relationships in execution order,
including YES/NO branches and TO/FROM PAGE or SHEET connectors. Continue until
the retrieved evidence reaches corrective actions or an unresolved link.

Present an ordered maintenance matrix with these columns: sequence; condition
or test; YES action; NO action; figure/sheet/block; confidence; citation. Then
list preparations, terminal corrective actions, and unresolved branches.

Do not call a referenced figure, sheet, block, or procedure missing merely
because it was absent from the first passages. Use SEARCH for that exact
reference first. Treat unlabeled `flows_to` decision edges as unresolved, not
as YES or NO. Distinguish explicit relationships from inference, never invent
a branch or maintenance action, and cite every step.
""".strip(),
    min_top_k=15,
    min_hops=3,
    min_context_chars=24_000,
    force_expand=True,
    follow_references=True,
)

DIAGNOSTIC_REVIEW = ChatMode(
    "diagnostic_review", "Diagnostic artifact review",
    "Review FMECA, ICD, IDD, and Self-Test Report evidence and gaps.",
    _policy("diagnostic_review"), min_top_k=15, min_context_chars=24000,
)
SYMPTOM_MATRIX = ChatMode(
    "symptom_failure_matrix", "Symptom–failure matrix",
    "Map documented symptoms to failure modes, separating explicit and inferred links.",
    _policy("symptom_failure_matrix"), min_top_k=15, min_context_chars=24000,
)
DIAGNOSTIC_PEER_REVIEW = ChatMode(
    "diagnostic_peer_review", "Diagnostic peer review",
    "Review the last completed non-peer answer against its original evidence.",
    _policy("diagnostic_peer_review"), peer_review=True,
)
SECURITY_PEER_REVIEW = ChatMode(
    "security_peer_review", "Source-marking peer review",
    "Check source-marking fidelity; does not determine classification or authorize release.",
    _policy("security_peer_review"), peer_review=True,
)
MODES = {mode.id: mode for mode in (
    GENERAL, FAULT_TRACE, DIAGNOSTIC_REVIEW, SYMPTOM_MATRIX,
    DIAGNOSTIC_PEER_REVIEW, SECURITY_PEER_REVIEW,
)}

FAULT_CODE = re.compile(r"\b\d{7}\b")
FIGURE_BLOCK = re.compile(
    r"\b(?:go\s+to\s+)?figure\s+([A-Za-z0-9.-]+)\s*,?\s*"
    r"(?:then\s+)?(?:go\s+to\s+)?block\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
FIGURE = re.compile(r"\bfigure\s+([A-Za-z0-9.-]+)", re.IGNORECASE)
SHEET = re.compile(
    r"\b(?:to|from)\s+(?:sheet|sht\.?)\s*[:#-]?\s*(\d+)\b",
    re.IGNORECASE,
)


def get_mode(mode_id: str) -> ChatMode | None:
    return MODES.get((mode_id or "general").strip().casefold())


def fault_trace_followups(question: str, context: str) -> list[str]:
    """Exact searches implied by a fault-code result and its diagram links."""
    codes = list(dict.fromkeys(FAULT_CODE.findall(question)))
    if not codes:
        codes = list(dict.fromkeys(FAULT_CODE.findall(context)))[:6]
    code_prefix = " ".join(codes[:6])
    figure_blocks: list[tuple[str, str]] = []
    folded_context = context.casefold()
    for code in codes:
        start = 0
        while (position := folded_context.find(code.casefold(), start)) >= 0:
            window = context[position:position + 700]
            match = FIGURE_BLOCK.search(window)
            if match and match.groups() not in figure_blocks:
                figure_blocks.append(match.groups())
            start = position + len(code)
    if not figure_blocks:
        figure_blocks = list(dict.fromkeys(FIGURE_BLOCK.findall(context)))[:2]
    figures = [figure for figure, _block in figure_blocks]
    if not figures:
        figures = list(dict.fromkeys(FIGURE.findall(context)))[:2]

    queries: list[str] = []
    for figure, block in figure_blocks:
        queries.append(f"{code_prefix} Figure {figure} Block {block}".strip())
    # A sheet number is local to its figure. Prefer figures tied to an explicit
    # starting block; applying a bare sheet number across every figure creates
    # plausible but unrelated maintenance paths.
    for sheet in dict.fromkeys(SHEET.findall(context)):
        for figure in figures[:2]:
            queries.append(f"Figure {figure} Sheet {sheet}")
    return list(dict.fromkeys(queries))
