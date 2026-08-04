<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# One-command local RAG

Point it at a folder of documents and get a private, searchable index of them,
running entirely on your machine. Vector store, embedding model, indexer, HTTP
API and a browser UI - one command, no accounts, no API keys, no cloud, nothing
leaves the box.

**Windows: double-click `Rag.bat`.** It asks which folder of documents to make
searchable, then does everything else. That is the whole setup.

Or from a PowerShell prompt, if you prefer naming the folder up front:

```powershell
.\rag-up.ps1 -Folder "S:\Team\Documents"
```

```bash
# Linux / macOS / WSL
./rag-up.sh --folder /mnt/share/documents
```

Then open **<http://127.0.0.1:49404>** and ask questions in plain language.

The folder is remembered in `.env`, so afterwards `Rag.bat` (or
`.\rag-up.ps1`) needs no arguments. `Rag.bat` also passes arguments through -
`Rag.bat status`, `Rag.bat down` - and sets `-ExecutionPolicy Bypass` for that
one invocation, which is what makes a script out of a zip runnable on a managed
machine without changing anything on it.

**The only prerequisite is Python 3.10+.** No Docker, no database server, no
model daemon - the whole stack is one Python process. The script checks for
Python and tells you exactly what to do if it is missing.

## What it can read

| Kind | Formats |
| --- | --- |
| Documents | `.pdf` (per-page), `.docx`, `.odt`, `.rtf` |
| Spreadsheets | `.xlsx`, `.xlsm`, `.xls`, `.ods`, `.csv`, `.tsv` |
| Slides | `.pptx` (per-slide), `.odp` |
| Email | `.eml`, `.msg` (Outlook) |
| Web / markup | `.html`, `.htm`, `.xml`, `.md`, `.txt` |
| Code & config | `.rs`, `.py`, `.c`, `.sh`, `.ps1`, `.toml`, `.yaml`, `.json`, and more |

Tables in Word, every sheet in a workbook, and slide text all become searchable
text. PDF hits carry a `[page N]` marker and slide hits a `[slide N]` marker, so
a result tells you where in the file to look.

**Running headers and footers are stripped from PDFs.** A book whose every page
carries "COMPANY CONFIDENTIAL - VOLUME I" puts those words in every chunk of the
document, so any query mentioning them matches the whole book uniformly and the
one relevant page has nothing to stand out with. Only lines repeating near the
edge of most pages are removed, and only in documents of four pages or more;
`RAG_PDF_KEEP_BOILERPLATE=1` disables it.

**Inside `.zip` archives too.** Each document in an archive is indexed and
cited individually under a virtual path:

```
contracts/2026-tender.zip!schedule-a/incident-2411.pdf
```

so a hit tells you both the archive and the file within it. Archives nested
inside archives are deliberately not opened - one level covers the real cases,
while unbounded recursion is how a zip bomb becomes a disk-space incident.
Guards: `RAG_ARCHIVE_MAX_BYTES` (500 MB), `RAG_ARCHIVE_MAX_MEMBERS` (2000), and
`RAG_ARCHIVES=0` to switch it off entirely. Encrypted or corrupt entries are
skipped with a log line, not fatal.

**Adding a file type needs no code** if it is already plain text:

```ini
# .env
RAG_EXTRA_TEXT_EXTS=.sql,.ini,.log
RAG_EXTRA_TEXT_EXTS=.sql=sql,.cfg=conf     # or name the language tag yourself
```

The tag is what `language_filter` matches on, defaulting to the extension
without its dot, and it also overrides a built-in mapping if you name one. For a
binary format needing real extraction, add a function to `app/extract.py` and
register it in `_EXTRACTORS` - about fifteen lines, and a missing library skips
that format rather than breaking the ingest.

**Not readable:** the legacy binary `.doc` and `.ppt` formats have no dependable
pure-Python reader. Rather than skip them silently, the ingest logs how many it
found - re-save those as `.docx`/`.pptx` to include them.

Each reader is optional at runtime: if a library is missing, that format is
skipped with an explanatory log line instead of failing the whole ingest.

## Why there is no daemon

Two pieces normally push a RAG setup toward containers; neither actually needs
one:

- **Vector store** - `qdrant-client` has a file-backed embedded mode. Same query
  engine and same API as the Qdrant server, but it keeps its segments in
  `.data/qdrant` and runs inside our process.
- **Embeddings** - `fastembed` runs the model in-process through ONNX. Weights
  are downloaded once into `.data/models` and reused offline afterwards.

So `rag-up` creates a virtualenv, installs dependencies, indexes the folder, and
serves. That is the entire install.

## What the one command actually does

1. Creates `.venv` and installs dependencies — **first run only**;
   afterwards it reinstalls only when `requirements*.txt` actually change.
2. Downloads the embedding model (`nomic-embed-text-v1.5`, ~520 MB) into
   `.data/models` — **first run only**; later starts are fully offline.
3. Walks the repository, chunks it, embeds it, and writes the vector index.
4. Starts a filesystem watcher so edits re-index automatically (2 s debounce).
5. Waits until `/health` reports `healthy`, prints the URL, and opens the UI.

First run takes a few minutes (dependency install + model download + full
ingest). Every later `rag-up` is seconds — the index and the model are on disk
and are reused.

## Using it

**Browser** — <http://127.0.0.1:49404>. Query box, language filter, path-prefix
filter, live index status, and a re-index button.

**Shell**

```powershell
.\rag-up.ps1 query "where is the IPC rendezvous done?"
.\rag-up.ps1 status
.\rag-up.ps1 reindex          # -Full to rebuild from scratch
.\rag-up.ps1 logs
.\rag-up.ps1 down             # -Wipe also drops the index and model cache
```

**HTTP**

```powershell
Invoke-RestMethod http://127.0.0.1:49404/health
# {"status":"healthy","mode":"native","embeddings":true,"qdrant":true,"chunks":16743,...}

$body = @{query='per-CPU TSS allocation'; top_k=5} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:49404/search -Method POST -Body $body -ContentType "application/json"

# full chunk text instead of a snippet, filtered
$body = @{query='page table walk'; top_k=5; language_filter='rs'; path_prefix='kernel/src'} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:49404/search/full -Method POST -Body $body -ContentType "application/json"
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Browser search UI |
| `/health` | GET | `status`, `embeddings`, `qdrant`, `chunks`, `model` |
| `/stats` | GET | Index size, repo, backend, last ingest, uptime |
| `/search` | POST | Ranked hits, snippet-truncated |
| `/search/full` | POST | Ranked hits, full chunk text |
| `/context` | POST | The same hits, assembled into a citable block for a generator |
| `/reindex` | POST | `{"full": false}` — background re-ingest |
| `/api-docs` | GET | Generated OpenAPI docs |

Search body fields: `query` (required), `top_k` (default 5), `language_filter`
(`rs`, `markdown`, `toml`, `ld`, `yaml`, `json`, `conf`, `sh`, `ps1`, `py`,
`asm`), `path_prefix` (e.g. `kernel/src`).

**Check `.status`, not `.ollama`.** Native mode has no Ollama, so `/health`
reports `embeddings` and `embed_backend` instead; `ollama` appears only when
that backend is actually in use.

## Pointing it at a different repository

The corpus is the only thing that is not self-contained — this is the part you
supply. Copy `.env.example` to `.env`:

```ini
RAG_REPO_MOUNT=E:/some/other/project
RAG_REPO_LABEL=SomeOtherProject
RAG_COLLECTION=some-other-project    # keeps indexes side by side
```

Or just re-run with `-Folder`. The indexer only ever reads the folder; it
never writes to it.

## Moving this folder somewhere else

This folder is self-contained and location-independent: copy it to your
Desktop, a tools directory, another drive — the scripts resolve everything
relative to themselves, and `.venv/` and `.data/` travel with the folder.

**One thing must change: the corpus.** The default corpus is "whatever
directory contains this one", which is almost never what you want. `Rag.bat`
asks for the folder and writes it for you; to set it by hand, put the path in
`.env`:

```ini
RAG_REPO_MOUNT=S:/Team/Documents
RAG_REPO_LABEL=Team share
```

If you forget, the server does not silently index your Desktop — it logs a
warning, exposes it as `warning` in `/health`, and shows it in the UI status
line. Moving the folder after it has already indexed is fine too: the index in
`.data/` moves with it and is reused.

A relocated copy is also how you run **several indexes at once** — one folder
per corpus, each with its own `.env` giving it a distinct `RAG_PORT` and
`RAG_COLLECTION`. The launcher reads `RAG_PORT` from `.env`, so
`.\rag-up.ps1 status|query|down` all talk to the right instance.

## Taking it to another PC

If the other machine has Python and normal internet access, just copy the
folder **without `.venv`** and run `.\rag-up.ps1`. It rebuilds the virtualenv
and downloads what it needs.

For a locked-down machine — no PyPI, no huggingface.co, or no internet at all —
prepare the folder first **on the machine that does have access**:

```powershell
.\rag-up.ps1 bundle
```

That downloads every wheel into `vendor\wheels\` (~70 MB) and pre-downloads the
embedding model into `.data\models\`. Then package it as a single file:

```powershell
.\rag-up.ps1 package
```

That writes `rag-portable.zip` (~600 MB) beside the folder, excluding `.venv`
(machine-specific, rebuilt on arrival) and `.data\qdrant` (the index of *this*
machine's corpus - so the zip does not carry a verbatim copy of local documents
somewhere they should not go). Copy the zip, unzip it on the other PC, and run
`.\rag-up.ps1 -Folder "..."`.

If you would rather copy the folder directly:

```powershell
robocopy . D:\rag /E /XD .venv .data\qdrant
```

**Match the Python minor version.** Binary wheels (onnxruntime, numpy,
pydantic-core, lxml) are built per Python minor version, so wheels fetched on a
3.14 machine will not install on a 3.12 one - and that only surfaces at the
target, where there is no network to recover. If the versions differ, add the
target's before travelling:

```powershell
.\rag-up.ps1 bundle -ForPython '3.12,3.13'
```

On the other PC, edit `.env` so `RAG_REPO_MOUNT` points at wherever the repo
lives there, then `.\rag-up.ps1`. It installs from the vendored wheels
(`--no-index`) and loads the bundled model, so it never reaches the network.

### What to copy, and what not to

| Item | Copy it? | Why |
| --- | --- | --- |
| `app/`, `*.ps1`, `*.sh`, `requirements*.txt` | **yes** | the program |
| `.env` | **yes**, then edit | `RAG_REPO_MOUNT` differs per machine |
| `vendor/wheels/` | yes, for an offline target | dependencies without PyPI |
| `.data/models/` | yes, for an offline target | ~520 MB, avoids huggingface.co |
| `.data/qdrant/` | optional | a prebuilt index — skips the first ingest |
| `.venv/` | **no** | records absolute paths to the Python that built it |

`.venv` is the one that bites: copied across, it looks present and fails in
confusing ways. `rag-up` probes it, and rebuilds it automatically if it cannot
run — but leaving it out of the copy is faster and cleaner.

Carrying `.data/qdrant/` over is safe: it is a single SQLite file, and the paths
inside it are repo-relative, so the index stays valid even though the repo sits
somewhere else on the other machine. Leave it behind and the first start simply
re-indexes.

**One caveat on `bundle`:** wheels are specific to the OS, CPU architecture and
Python minor version they were downloaded for. Windows → Windows on the same
Python 3.x is fine. Crossing OSes means re-running the download on the target,
or passing pip's `--platform` / `--python-version` flags.

Also check with whoever owns the work machine before putting a copy of a private
repository's index on it — the chunk text is stored verbatim in `.data/qdrant`,
so that file contains your source code.

## Docker mode (optional)

If you would rather have the stack isolated in containers — or you already run
Ollama and want the model shared with other tools — the compose path is still
there and fully supported:

```powershell
.\rag-up.ps1 -Docker          # bash: ./rag-up.sh --docker
```

That starts a Qdrant server, an Ollama server, and the API container, with
every port bound to `127.0.0.1` only and the repo bind-mounted **read-only**.
Same endpoints, same UI. It is heavier (three containers, ~2 GB of images) and
byte-reproducible across machines, which is the reason to choose it. Native
mode is the default because it asks less of the machine.

The two modes keep separate indexes (one on the host, one in a Docker volume),
so switching does not corrupt anything, and the first start in a mode you have
not used before will index from scratch.

## What gets indexed

Rust, Markdown, TOML, YAML, JSON, linker scripts, shell/PowerShell, Python,
assembly, C, and `.conf` files. Skipped: `.git`, `target/`, `node_modules`,
`.venv`, `.data`, `archive/`, lockfiles, logs, images, archives,
binaries, and anything over 1 MB. Chunking is structure-aware — Rust splits at
item boundaries (`fn`/`impl`/`struct`/`mod`, carrying the doc comment above it),
Markdown splits at headings — so a hit usually starts at the top of the thing
you were looking for rather than mid-body. All of it is tunable through `.env`
(see `app/config.py`).

## Network drives and shares

Both work. A mapped drive (`S:\Team\Docs`) and a UNC path
(`\\fileserver\Team\Docs`) are walked and read exactly like local folders,
and the folder picker returns either. Four things differ in practice:

- **Watching is off by default on a UNC path.** SMB does not deliver change
  notifications dependably, so a watcher there appears to work while quietly
  missing edits - worse than no watcher, because the index looks live when it
  is stale. Re-index on demand with `Rag.bat reindex`, or force watching on with
  `RAG_WATCH=1` if your share does support it. Mapped drive letters are not
  auto-detected as network; set `RAG_WATCH=0` yourself if `S:` is a share.
- **It is slower**, though usually not dominant: embedding is CPU-bound and
  costs more than the read for most documents. Large PDFs over a slow link are
  the exception.
- **Files you cannot read are skipped**, logged, and counted in the ingest's
  error total. A permissions patchwork degrades the index rather than stopping
  it.
- **A dropped connection does not abort the run.** Failures are caught per file,
  so the ingest continues and picks up the rest on the next `reindex`.

Point it at the subtree you actually search rather than the root of a
department share - it is the difference between a coffee break and an overnight
run, and the index quality is better for it.

## How long the first index takes

Embedding is the whole cost, it is CPU-bound, and it happens once. Measured on
this project's own repo: **~3.3 chunks/sec** on a desktop CPU with the default
model. A document is typically a handful of chunks, so as a planning figure:

| Corpus | Rough chunks | First index |
| --- | --- | --- |
| 500 documents | ~3,000 | ~15 min |
| 5,000 documents | ~30,000 | ~2.5 hours |
| 20,000 documents | ~120,000 | ~10 hours |

It is incremental afterwards - only changed files are re-embedded - and
`status` shows the chunk count climbing throughout, so a long run is visibly
progressing rather than hung. Leave it running; it survives you closing the
window that started it.

Three ways to make it faster, in order of effect:

```ini
# .env
RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5     # ~5x faster, 384-dim, weaker recall
RAG_EMBED_BATCH=64                         # more throughput, more memory
RAG_EMBED_THREADS=0                        # hand thread choice back to onnxruntime
```

Threads already default to the machine's core count, so there is usually
nothing to set. Whether that beats onnxruntime's own choice is machine-specific
and worth trying both ways if a large ingest drags.

Changing the model changes the vector width, so follow it with
`.\rag-up.ps1 reindex -Full`. Narrowing the corpus is the other lever - point
`-Folder` at the subtree you actually search rather than a whole shared drive.

## When the right passage will not come top: reranking

The vector search is a *bi-encoder*. Query and chunk are embedded separately and
compared by distance, which is what makes it fast enough to scan a whole corpus
- and is also its ceiling. The two texts never meet, so the score measures
topical proximity, not whether the passage answers the question.

The failure this produces is specific and recognisable: ask a book about text
chunking "how should the book be split into volumes?" and every page about
splitting text scores highly, because they genuinely are about splitting. No
amount of better extraction or chunking fixes it - the passages really are
similar, in the only sense a bi-encoder can measure.

A **cross-encoder** reads the query and the passage together and scores the
pair, which is far better at that distinction. It is also far too slow to run
over a whole corpus, so it re-orders the candidates the vector search already
found - retrieve wide and cheap, then re-order narrow and accurate.

```ini
# .env
RAG_RERANK=1
#RAG_RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2   # 82 MB, the default
#RAG_RERANK_CANDIDATES=40                          # how many the vector search feeds it
```

Restart - **no reindex needed**, this is query-time only. `/health` reports
`rerank` and `rerank_model` so you can confirm it is on.

The costs are real: a second or two per query on CPU, and another model. Off by
default for that reason. It is most worth turning on when feeding a generator,
where the top few chunks decide the answer, and least worth it when you are
browsing results yourself and can see rank 4 as easily as rank 1.

Results carry both scores when it is on - `score` from the cross-encoder and
`vector_score` from the search - so a large disagreement between them is visible
rather than hidden.

**Cross-encoder scores are logits, and the sign carries meaning.** Unlike the
vector score (cosine, always 0-1), a reranked hit is scored roughly "does this
passage answer this query": strongly positive means yes, negative means no.
Observed on a real corpus - the passage answering a well-aimed query scored
**+6.1**, while the best hit for a query the corpus does not really answer scored
**-2.9**, with everything below it clustered at the same value.

That is directly useful when feeding a generator. If nothing in `/context`
scores above zero, the corpus probably does not contain the answer, and the
prompt should say so rather than let the model compose something confident from
passages that merely share vocabulary. A threshold near 0 is a reasonable
starting point.

**A reranker can only re-order what retrieval found.** If every result comes
back with the same low cross-encoder score, that is the model saying *none of
these answer the question* - the passage that does was not among the candidates.
Raising `RAG_RERANK_CANDIDATES` widens the pool it gets to see; the cost is
linear, since each candidate is a separate cross-encoder pass.

**Duplicate documents waste the same way.** A draft kept beside its final
version puts near-identical text in two files, and both will score alike and
occupy two slots. De-duplication is within a file, not across files - two copies
of a book are two legitimately different documents as far as the index is
concerned. Exclude the folder instead:

```ini
RAG_EXCLUDE_DIRS=_to_delete,_drafts
```

That list is *added* to the built-in exclusions rather than replacing them.

## Feeding an LLM (the "G" in RAG)

This service is the **retrieval** half only. It finds passages; it runs no
language model and needs none installed. Whatever writes the prose - Gemma,
Llama, a hosted API - lives outside it, and everything here works exactly the
same whether or not one is present.

`POST /context` is the seam. It runs the same search as `/search/full` and
returns the hits pre-assembled:

```powershell
$body = @{query='who are the recurring characters?'; top_k=8; max_chars=8000} | ConvertTo-Json
$r = Invoke-RestMethod -Uri http://127.0.0.1:49404/context -Method POST -Body $body -ContentType "application/json"
$r.context     # numbered, citable text block - paste into a prompt
$r.sources     # [{n, path, start_line, end_line, score, citation}]
```

The block looks like this, so the model can cite by number and a reader can
check the claim against the file:

```
[1] 00_Book_Bible/RAG_Book_Bible.md:183-194
Readers should understand these roles instinctively...

[2] Volume_I/frontmatter/introduction.md:3-6
...
```

`max_chars` caps the block for the model's context window. Whole chunks are
dropped from the bottom to fit and the count comes back in
`chunks_dropped_for_budget` - a chunk is never cut in half, because a generator
handed a sentence that stops mid-clause will finish it from imagination.

Two things worth telling whoever wires up the generator:

- **Never build a prompt from `/search`.** It truncates each hit to 400
  characters for the UI. `/context` and `/search/full` return whole chunks.
- **`top_k` of 5-8 is the usual sweet spot.** Past that the relevant passage
  tends to drown rather than the answer improving.

## If results look wrong

Retrieval quality is the one thing worth tuning, and there are three dials.

**1. The path is part of the vector.** Each chunk is embedded as
`dir/sub/file.pdf` + text, which is why "where is the IPC rendezvous done?"
finds the right file. The cost is that directory names match content queries:
a folder called `Volume_I` pulls everything beneath it toward any question
mentioning volumes. If content questions keep landing in the wrong folder:

```ini
RAG_EMBED_PATH=off      # or "name" to keep the filename, drop the directories
```

then `Rag.bat reindex -Full`. Worth A/B-ing on questions whose answers you
already know.

**2. Chunk size.** Plain `.txt` has no structural boundaries, so it splits on
blank lines and chunks can end up short and context-poor. `RAG_CHUNK_CHARS`
(default 1600) gives each chunk more to match against.

**3. The model.** `mixedbread-ai/mxbai-embed-large-v1` retrieves better than the
default at 1024 dimensions and roughly half the speed. Both need a `-Full`
reindex, since the vector width changes.

Before reaching for any of them, raise `top_k` to 20 and look at where the right
passage actually lands. If it is at rank 8, that is a ranking problem worth
tuning. If it is absent entirely, the passage was never chunked the way you
expect - check what the file looks like in the results, not what you assume it
contains.

## When to use it

Use RAG for conceptual questions — *"what is our refund policy for damaged
goods?"*, *"where is the capability revoke walk?"* — anywhere you don't know the
file. Use `grep` for exact symbols or strings, or when you know the path: it is
faster and more precise for those. The two are complements, not competitors.

## Checking the corpus without installing anything

```bash
python3 selftest.py          # what would be indexed, and how it chunks
```

Walks the repo with the real config and chunker, using only the standard
library — no virtualenv, no model, no containers. Useful for confirming the
include/exclude rules before a long first ingest.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `cannot be loaded because running scripts is disabled` / `not digitally signed` | Files out of a zip carry Windows' mark-of-the-web. `Get-ChildItem -Recurse \| Unblock-File`, then `Set-ExecutionPolicy -Scope Process Bypass -Force` (that window only, no admin, machine unchanged). If group policy forbids even that: `powershell -ExecutionPolicy Bypass -File .\rag-up.ps1 -Folder "..."`. |
| `Python 3.9+ was not found` | `winget install Python.Python.3.12`, or use `-Docker`. |
| `startup failed: Could not load model ...` | The model download needs one-time internet access to huggingface.co. Behind a proxy, set `HTTPS_PROXY` before `rag-up`. |
| First run looks stuck | It is installing deps or downloading the model. `.\rag-up.ps1 logs`. |
| `status: empty` | Ingest found nothing. Check `RAG_REPO_MOUNT`, then `.\rag-up.ps1 reindex`. |
| Port 49404 in use | Another RAG is running (`.\rag-up.ps1 down`), or set `RAG_PORT`. |
| Results feel stale | The watcher may be off on some filesystems — `.\rag-up.ps1 reindex`. |
| `python -m app.ingest` fails with a lock error | The server holds the embedded store. Re-index through it: `.\rag-up.ps1 reindex`. |
| Want better code recall | `RAG_EMBED_MODEL=jinaai/jina-embeddings-v2-base-code` in `.env`, then `.\rag-up.ps1 reindex -Full`. |
| Want a smaller/faster model | `RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5` (~67 MB, 384-dim), then reindex `-Full`. |

Nothing here is a blocker for work: if the RAG is down, `grep` and direct file
reads always work.
