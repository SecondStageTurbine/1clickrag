# SPDX-License-Identifier: MPL-2.0
"""Entity extraction, by pattern rather than by model.

What comes out of here is deliberately not "named entity recognition". There is
no model, no download and no new dependency: this is regex and a stop list over
text that is already in memory during ingest, which is the only kind of
extraction that fits a tool promising one process and an offline install.

The trade is recall and precision, and the design leans on two things to make
that survivable:

* The graph is used for *retrieval*, not for assertions. A wrong edge widens
  the candidate set by one chunk and the reranker drops it. A wrong triple in a
  knowledge base would be a false claim - which is exactly why this module
  extracts mentions and lets co-occurrence carry the relation, rather than
  guessing at "A supplies B" from a verb it happened to match.
* Anything appearing in most documents is dropped at query time by document
  frequency (see graph.py). Headers, footers and boilerplate are the bulk of
  what a pattern-based extractor gets wrong on a document corpus, and they are
  precisely what a df ceiling removes without a hand-maintained list.
"""

from __future__ import annotations

import re
from collections import Counter

__all__ = ["Mention", "extract_entities", "normalise_key"]

# A token: a word, an acronym, or an identifier with internal punctuation
# (INC0042317, CHORD-1234, MIL-STD-810H, 12345-02).
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&./_-]*")

# Sentence-ending punctuation, for suppressing the capital that merely starts a
# sentence. Without this the extractor's most common "entity" is the first word
# of every paragraph.
SENTENCE_END = frozenset(".!?:;")

# Words that may sit *inside* a multi-word name without breaking the run.
# "the" is deliberately absent: a capitalised "The" mid-run nearly always starts
# a new phrase ("Dana Okafor The Customer" is two names, not one).
CONNECTORS = frozenset(
    {"of", "for", "and", "de", "del", "van", "von", "der", "la", "le", "&"}
)

# Single capitalised words that are almost never the thing being named. Kept
# short on purpose: the df ceiling in graph.py is the general mechanism, and a
# long hand-written list is the kind that rots.
COMMON_WORDS = frozenset(
    """
    a an the and or but if then so as of to in on at by for with from is are was
    were be been being do does did have has had can could would should will
    shall may might must this that these those there here it its we you they he
    she him her his our your their who whom whose what when where which why how
    not no yes all any both each few more most other some such only own same too
    very just also however therefore thus hence while during before after above
    below between under over again further once because although though since
    until unless about against among within without upon toward towards per via
    use used using make made get got give given take taken see seen say said
    show shown find found need needed want wanted work works working
    every one two three four five six seven eight nine ten another many much
    several various first second third fourth fifth next last new old good great
    best better worse worst less least little lot lots enough
    either neither others someone anyone everyone nobody something anything
    everything nothing moreover meanwhile instead rather perhaps maybe often
    always never sometimes usually typically generally normally finally now
    today yesterday tomorrow later earlier soon still yet already almost
    even ever else quite really actually simply merely indeed
    problem problems solution solutions answer answers question questions
    example examples result results reason reasons thing things way ways
    part parts case cases point points fact facts idea ideas
    """.split()
)

# Roman numerals: "Volume III", "Part IV". Caught here rather than by the
# structural list because the numeral is what survives once the run is split.
ROMAN = re.compile(r"^[IVXLCDM]{1,7}$")

# I'd, don't, we've - the capital belongs to a contraction, not a name.
CONTRACTION = re.compile(r"^[A-Za-z]+'[a-z]{1,2}$")

# Above this an all-capitals word is a heading in a document that shouts its
# headings, not an acronym. NGPCN is 5; FOUNDATIONS is 11.
MAX_ACRONYM_CHARS = 8

# Document furniture. These start a caption or a heading, and everything after
# them is a number, not a name.
STRUCTURAL = frozenset(
    """
    figure fig table tab section sect chapter chap appendix annex exhibit
    attachment page pages note notes revision rev draft version ver volume vol
    part paragraph para item step steps clause article schedule
    contents abstract introduction conclusion summary references glossary
    acknowledgements acknowledgments index copyright confidential proprietary
    """.split()
)

MONTHS = frozenset(
    """
    january february march april may june july august september october november
    december jan feb mar apr jun jul aug sep sept oct nov dec
    monday tuesday wednesday thursday friday saturday sunday
    mon tue tues wed thu thur thurs fri sat sun
    """.split()
)

# All-caps tokens that pass the acronym shape but carry no identity.
ACRONYM_STOP = frozenset(
    """
    THE AND FOR NOT ALL ANY USE SEE NOTE PAGE FIG TAB PDF DOC DOCX XLS XLSX PPT
    PPTX CSV TXT HTML HTTP HTTPS URL WWW COM NET ORG GOV EDU MIL INC LLC LTD
    YES NO NEW OLD END TOP BOT MAX MIN AVG SUM TBD TBA NA N/A ETC IE EG VS
    ONE TWO SIX TEN AM PM UTC GMT EST PST CST MST ISO
    """.split()
)

MAX_WORDS = 4  # longest multi-word name kept
MIN_CHARS = 3
MAX_PER_CHUNK = 60  # bound on how much one chunk can add to the graph


class Mention:
    """One entity found in one chunk."""

    __slots__ = ("name", "key", "kind", "count")

    def __init__(self, name: str, key: str, kind: str, count: int) -> None:
        self.name = name
        self.key = key
        self.kind = kind
        self.count = count

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Mention({self.name!r}, {self.kind}, x{self.count})"


def normalise_key(name: str) -> str:
    """The form two spellings of the same thing must share.

    Case and spacing only. Deliberately not stemming or acronym expansion:
    "NG PCN" -> "NGPCN" is a corpus-specific judgement, and the client's
    synonym map is where a human records it.
    """
    key = re.sub(r"\s+", " ", name).strip().casefold()
    if key.endswith("'s"):
        key = key[:-2]
    return key.strip(" -.,")


def _is_upper_token(word: str) -> bool:
    return bool(word) and word.isupper() and any(char.isalpha() for char in word)


def _is_acronym(word: str) -> bool:
    # ISO9001 and NGPCN qualify; Iso9001, IBMs and FOUNDATIONS do not.
    if len(word) < 3 or len(word) > MAX_ACRONYM_CHARS or not _is_upper_token(word):
        return False
    if ROMAN.match(word):
        return False
    return word.casefold() not in COMMON_WORDS


def _is_identifier(word: str) -> bool:
    """A ticket ID, part number, standard or filename - not a measurement.

    This is the class of thing embeddings are worst at and exact search is best
    at, so it is worth catching even when the surrounding sentence is
    unparseable. The capital-letter requirement is what separates NW-2200 and
    INC0042317 from "40-ton", "500-character" and "384-dimensional", which are
    prose about a number and match nothing anyone would search for.
    """
    if len(word) < 4:
        return False
    # A filename: supplier_contract_1994.pdf, search.py. The extension must
    # start with a letter, or "0.65" reads as a file called 0 of type 65.
    if re.search(r"\.[A-Za-z][A-Za-z0-9]{1,3}$", word):
        return True
    digits = sum(char.isdigit() for char in word)
    if digits >= 2 and any(char.isupper() for char in word):
        return True
    return bool(re.fullmatch(r"\d{2,}(?:[-.]\d{2,})+", word))


def _acceptable(name: str, words: list[str], kind: str, only_at_sentence_start: bool) -> bool:
    if len(name) < MIN_CHARS:
        return False

    lowered = [word.casefold() for word in words]
    if lowered[0] in STRUCTURAL:
        return False
    if all(word in MONTHS for word in lowered):
        return False

    if kind == "acronym":
        return name not in ACRONYM_STOP

    if kind == "identifier":
        return True

    if len(words) == 1:
        # One capitalised word carries almost no signal unless it is used like a
        # name somewhere other than the start of a sentence.
        if lowered[0] in COMMON_WORDS or lowered[0] in MONTHS:
            return False
        if only_at_sentence_start:
            return False
        if not any(char.isalpha() for char in name):
            return False
    return True


def _tokens(text: str) -> list[tuple[str, bool]]:
    """Every token with a flag for "this one starts a sentence"."""
    out: list[tuple[str, bool]] = []
    at_start = True
    position = 0
    for match in TOKEN.finditer(text):
        gap = text[position : match.start()]
        if position and (any(char in SENTENCE_END for char in gap) or "\n" in gap):
            at_start = True
        out.append((match.group(0).strip(".,'"), at_start))
        at_start = False
        position = match.end()
    return out


def extract_entities(text: str, max_entities: int = MAX_PER_CHUNK) -> list[Mention]:
    """Every entity mentioned in one chunk, with how often it occurs."""
    tokens = _tokens(text)
    counts: Counter[tuple[str, str, str]] = Counter()
    # A single capitalised word is only kept if it is used somewhere other than
    # the head of a sentence, so this tracks both facts per candidate.
    seen_mid_sentence: dict[str, bool] = {}

    # A document that shouts its headings ("RAG FROM SCRATCH") yields one
    # spurious acronym per word. Neighbouring capitals are the tell, so inside
    # such a run only genuinely acronym-shaped words are kept.
    shouted = [False] * len(tokens)
    start = 0
    while start < len(tokens):
        if not _is_upper_token(tokens[start][0]):
            start += 1
            continue
        end = start
        while end + 1 < len(tokens) and _is_upper_token(tokens[end + 1][0]):
            end += 1
        if end > start:
            for position in range(start, end + 1):
                shouted[position] = True
        start = end + 1

    index = 0
    while index < len(tokens):
        word, at_start = tokens[index]
        if not word or CONTRACTION.match(word):
            index += 1
            continue

        if _is_identifier(word):
            key = normalise_key(word)
            counts[(word, key, "identifier")] += 1
            seen_mid_sentence[key] = True
            index += 1
            continue

        if _is_upper_token(word):
            limit = 5 if shouted[index] else MAX_ACRONYM_CHARS
            if _is_acronym(word) and len(word) <= limit:
                key = normalise_key(word)
                counts[(word, key, "acronym")] += 1
                seen_mid_sentence[key] = True
            index += 1
            continue

        if not (word[0].isupper() and word[0].isalpha()) or ROMAN.match(word):
            index += 1
            continue

        # Greedy run of capitalised words, connectors allowed between them.
        run = [word]
        cursor = index + 1
        while cursor < len(tokens) and len(run) < MAX_WORDS:
            candidate, candidate_starts_sentence = tokens[cursor]
            if not candidate or candidate_starts_sentence:
                # "...asked Marcus. It was" must not become "Marcus It".
                break
            if candidate.casefold() in CONNECTORS:
                # Only if a capitalised word follows: "Department of Defense"
                # continues, "Board of the" does not.
                if (
                    cursor + 1 < len(tokens)
                    and tokens[cursor + 1][0]
                    and not tokens[cursor + 1][1]
                    and tokens[cursor + 1][0][0].isupper()
                    and len(run) + 2 <= MAX_WORDS
                ):
                    run.append(candidate)
                    run.append(tokens[cursor + 1][0])
                    cursor += 2
                    continue
                break
            if (
                candidate[0].isupper()
                and candidate[0].isalpha()
                and not _is_upper_token(candidate)
                and not CONTRACTION.match(candidate)
            ):
                run.append(candidate)
                cursor += 1
                continue
            break

        # A run may have picked up filler at either end - "And Dana", "From
        # Scratch", "The Regex One". The name is what is left after trimming it.
        while run and (run[0].casefold() in COMMON_WORDS or run[0].casefold() in CONNECTORS):
            run.pop(0)
        while run and (run[-1].casefold() in COMMON_WORDS or run[-1].casefold() in CONNECTORS):
            run.pop()
        if not run:
            index = cursor if cursor > index else index + 1
            continue

        name = " ".join(run)
        key = normalise_key(name)
        counts[(name, key, "name")] += 1
        if not at_start:
            seen_mid_sentence[key] = True
        seen_mid_sentence.setdefault(key, False)
        index = cursor if cursor > index else index + 1

    mentions: list[Mention] = []
    for (name, key, kind), count in counts.items():
        words = name.split()
        if not _acceptable(name, words, kind, not seen_mid_sentence.get(key, True)):
            continue
        mentions.append(Mention(name, key, kind, count))

    # Longer, more frequent names first, so the per-chunk cap drops the noise
    # rather than the subject.
    mentions.sort(key=lambda m: (-m.count, -len(m.name)))
    return mentions[:max_entities]
