# SPDX-License-Identifier: MPL-2.0
"""Searching for what the page calls a thing, not what you called it.

The failure this exists for: ask "what does the System say when it first
appears?" of a corpus where the System's panels are lettered DAILY QUEST,
PENALTY and ALARM - and never once print the word "System" - and every arm
misses. The vector arm matches passages where characters *discuss* the System,
because that is what the question sounds like. The keyword arm matches the
literal word, which is in those same discussions and not in the panels. Both
are working correctly and the answer is nowhere in the results.

Nothing about that is exotic. Any question phrased in the reader's vocabulary
rather than the document's hits it, and a searcher who already knew the
document's wording would not have needed to search.

Two techniques here, both deterministic and both needing no language model,
which matters because the chat pane's model is whatever the user picked and
may be small, slow or absent:

**Content terms** drop the question's scaffolding - "what", "does", "say",
"when" - leaving what is actually being asked about. Cheap, and it stops a
long question diluting the keyword arm's BM25 scores.

**Feedback terms** are the part that crosses the vocabulary gap. Take the
first pass's best passages, find the words that distinguish them, and search
again with those. Classic pseudo-relevance feedback: the first pass need not
contain the answer, only be *about the right subject*, and the vocabulary it
introduces is the document's own rather than the asker's. Its risk is equally
classic - if the first pass is off-topic the second inherits the drift - which
is why the result is fused as one more ranking rather than replacing anything.
"""

from __future__ import annotations

import re
from collections import Counter

TERM = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{1,}")

# Question scaffolding and discourse glue. Deliberately not a general English
# stopword list: words like "first" and "before" carry real meaning in a
# question about when something happened, and dropping them loses the point of
# the question.
NOISE = {
    "a", "about", "after", "all", "am", "an", "and", "any", "anyone", "are",
    "as", "at", "be", "because", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "doing", "done", "for", "from", "get", "give", "gives",
    "had", "has", "have", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "just", "know", "let", "like",
    "list", "make", "may", "me", "mean", "means", "might", "more", "most",
    "much", "must", "my", "of", "on", "one", "or", "our", "out", "over",
    "please", "said", "say", "says", "see", "shall", "she", "should", "show",
    "so", "some", "such", "tell", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "thing", "things", "this", "those",
    "to", "up", "us", "use", "used", "very", "was", "way", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your",
    # Words about documents rather than in them. Someone writing "which
    # passage in the document mentions X" is describing the search, and
    # matching those words retrieves whatever happens to be self-referential.
    "chapter", "chapters", "document", "documents", "file", "files", "page",
    "pages", "passage", "passages", "section", "sections", "text",
}


def content_terms(query: str, limit: int = 12) -> list[str]:
    """The question with its scaffolding removed."""
    seen: list[str] = []
    for match in TERM.findall(query):
        word = match.lower().strip("'-")
        if len(word) < 3 or word in NOISE or word in seen:
            continue
        seen.append(word)
        if len(seen) >= limit:
            break
    return seen


def feedback_terms(texts: list[str], exclude: list[str], limit: int = 8,
                   min_documents: int = 2, frequency=None,
                   corpus_chunks: int = 0, max_share: float = 0.08,
                   min_corpus: int = 3) -> list[str]:
    """Words that distinguish the first pass's passages, for a second one.

    Ranked by how many of those passages a word appears in, not by raw count:
    a word repeated forty times on one page describes that page, while a word
    on three of five pages describes the subject. `min_documents` is what stops
    a single unlucky passage dictating the follow-up search.

    Appearing in the sample is not enough on its own, though. The first version
    of this harvested "it's", "that's", "sung" and "webtoon" - the last being
    the studio credit printed on the final page of every chapter. They were in
    the sample because they are in *everything*, which is exactly what makes
    them worthless for finding anything. `frequency` supplies each candidate's
    corpus-wide chunk count and `max_share` drops the ones that are simply
    common. A word in under 8% of chunks says something; a word in half of
    them is furniture.
    """
    appearances: Counter[str] = Counter()
    weight: Counter[str] = Counter()
    skip = set(exclude)

    for text in texts:
        here = set()
        for match in TERM.findall(text):
            word = match.lower().strip("'-")
            # Four, not three: OCR of artwork produces a lot of short debris
            # ("ito", "3a", "OALARM"), and short fragments are where it lands.
            if len(word) < 4 or word in NOISE or word in skip:
                continue
            # Contractions carry no subject at all and survive every other
            # filter, because "it's" really is in most passages.
            if "'" in match:
                continue
            here.add(word)
            weight[word] += 1
        for word in here:
            appearances[word] += 1

    candidates = [w for w, n in appearances.items() if n >= min_documents]
    if not candidates:
        # One passage is all there was. Better a narrow follow-up than none,
        # so fall back to the most repeated words in it.
        candidates = [w for w, _ in weight.most_common(limit * 3)]

    if frequency is not None and corpus_chunks > 0 and candidates:
        ceiling = max(2, int(corpus_chunks * max_share))
        counts = frequency(candidates)
        # A band, not a ceiling. Filtering only for rarity picks the rarest
        # words in the corpus, and on OCR'd text those are the scanning errors:
        # "abovemep", "actoally", "c0mmanders", each appearing exactly once
        # because it is not a word. A term worth searching for was read
        # correctly somewhere else too, so it has to appear a few times.
        kept = [
            w for w in candidates
            if min_corpus <= counts.get(w, 0) <= ceiling
        ]
        if kept:
            candidates = kept
        else:
            # A corpus about one subject makes its own vocabulary common, and
            # a narrow follow-up beats none - but still drop the singletons.
            candidates = [w for w in candidates if counts.get(w, 0) >= min_corpus]
            if not candidates:
                return []
        # How many of the sampled passages a word appeared in is the signal;
        # corpus frequency only breaks ties, rarer first.
        candidates.sort(key=lambda w: (-appearances[w], counts.get(w, 0), w))
        return candidates[:limit]

    candidates.sort(key=lambda w: (-appearances[w], -weight[w], w))
    return candidates[:limit]
