<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# One-command local RAG

Point it at a folder of documents and get a private, searchable index of them,
running entirely on your machine. Vector store, embedding model, keyword index,
entity graph, indexer, HTTP API and a browser UI - one command, no accounts, no
API keys, no cloud, nothing leaves the box.

**Windows: double-click `Rag.bat`.** It asks which folder of documents to make
searchable, then runs through the handful of extras that are off by default -
reranking, OCR for scanned pages, a chat pane, starting at logon - one line
each, Enter to take the suggestion. Then it does everything else. That is the
whole setup, and `.\rag-up.ps1 setup` revisits those answers later.

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

**The only prerequisite is Python 3.11+** (3.10 runs, but the offline bundle
cannot be resolved for it — `qdrant-client`'s own dependencies need 3.11).
No Docker, no database server, no
model daemon - the whole stack is one Python process. The script checks for
Python and tells you exactly what to do if it is missing.

## What you get

| | What it does |
| --- | --- |
| **Semantic search** | ask in plain language; finds the passage, not the keyword — [tuning](#if-results-look-wrong) |
| **Exact search, fused in** | ticket IDs, part numbers, error codes — the strings embeddings are worst at — [how](#keyword-search-and-the-entity-graph) |
| **Asks again in the document's words** | when your wording and the page's disagree, it searches a second time using the page's — [how](#when-your-words-and-the-documents-words-differ) |
| **Counts, as well as finds** | "how many forms were signed this year" is computed over the index, not guessed from a few passages — [how](#how-many-are-there) |
| **Judges a spreadsheet** | apply one policy to every row of a 16,000-line sheet, with the criterion recorded per verdict — [how](#deciding-one-thing-about-every-row-of-a-spreadsheet) |
| **Multi-hop** | follow a name from one document to the others mentioning it — [how](#keyword-search-and-the-entity-graph) |
| **Reads almost anything** | PDF, Word, Excel, PowerPoint, email, HTML, code, and inside `.zip` — [formats](#what-it-can-read) |
| **Reads scans too** | pages whose text is pixels, via OCR that needs no internet and no extra binary — [OCR](#scanned-pages-ocr) |
| **Stays current** | a watcher for live edits, a reconcile for whatever changed while it was off — [how](#keeping-it-running-and-keeping-it-current) |
| **Stays running** | one command registers it to start at logon and revive itself — [autostart](#autostart) |
| **Cited answers** | `/context` returns numbered passages, cited by file, lines and section — [the "G" in RAG](#feeding-an-llm-the-g-in-rag) |
| **A chat pane, if you want one** | pick from a dropdown of whatever this machine has — Claude, Codex, Ollama, any OpenAI-compatible endpoint — [chat](#chat-in-the-browser) |
| **Nothing lost quietly** | changes are queued to disk, retried with backoff, and dead-lettered where you can see them — [how](#nothing-is-indexed-on-a-promise) |
| **Measurable** | score a retrieval change against questions you know the answers to, instead of guessing — [how](#knowing-whether-a-change-helped) |
| **Nothing hidden in a config file** | a tab listing every setting, its value now, and where that value came from — [how](#using-it) |
| **Bring your own embeddings** | point it at a hosted or company-internal embeddings service instead of the built-in model — [how](#somebody-elses-embedding-service) |
| **Runs on the GPU if you have one** | optional CUDA embedding, with a check that proves it rather than assuming — [how](#embedding-on-the-gpu) |
| **Cheap to rebuild** | a full reindex reuses the extracted text instead of re-reading every document — [how](#scanned-pages-ocr) |
| **Says when it is stale** | change a setting that shapes the vectors and it tells you the index no longer matches — [how](#when-a-setting-changes-under-a-built-index) |
| **A shell client** | project-scoped questions, two query modes, one standard prompt — [PowerShell client](#the-powershell-client-rag-clientpsm1) |
| **Travels offline** | one zip carries the code, the wheels and the models to an air-gapped PC — [taking it elsewhere](#taking-it-to-another-pc) |

It runs no language model and needs none installed. It finds and cites the
passages; whatever writes prose from them is your choice, lives outside this,
and is optional — with none configured, everything above works unchanged.

## What it can read

| Kind | Formats |
| --- | --- |
| Documents | `.pdf` (per-page), `.docx`, `.odt`, `.rtf` |
| Spreadsheets | `.xlsx`, `.xlsm`, `.xls`, `.ods`, `.csv`, `.tsv` |
| Slides | `.pptx` (per-slide), `.odp` |
| Email | `.eml`, `.msg` (Outlook) |
| Web / markup | `.html`, `.htm`, `.xml`, `.md`, `.txt` |
| Code & config | `.rs`, `.py`, `.c`, `.sh`, `.ps1`, `.toml`, `.yaml`, `.json`, and more |
| Scanned pages | any PDF page with no text layer, and `.cbz` — [with OCR on](#scanned-pages-ocr) |

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

### Scanned pages (OCR)

A scanned contract, a photographed report, a comic: the words are on the page,
but as pixels. `pypdf` returns nothing for them, so by default those pages index
as empty — and a search that finds nothing looks identical whether the corpus
lacks the answer or the reader could not see it. You do get a log line saying
how many pages were skipped for having no text layer.

`RAG_OCR=1` in `rag\.env` reads them. It is **off by default** because it is
about a thousand times slower than reading a text layer — seconds per page
against microseconds — and starting a multi-hour job on someone's first index
unasked would be indefensible. Measured on a webtoon corpus here: **2.8 s and
~500 characters per page at 0.83 mean confidence**, so ~5,000 pages is about
four hours, once — and once is the operative word. Modification times mean a
later reconcile never redoes it, and since the extracted text is kept (below), a
full reindex does not either.

Only pages **with no text layer** are sent to OCR, so a typed report with
scanned appendices costs OCR on the appendices alone. Turning it on also makes
`.cbz` comic archives readable — those are page images and nothing else.

Two things decide whether the output is usable, and both are the opposite of
the obvious setting:

- **Pages are rendered at native resolution, never upscaled.** Rendering a
  720px-wide page at 2× made recognition slower *and* worse — 1.44 s against
  0.32 s, confidence 0.77 against 0.81. There is no extra detail to find in an
  upsampled bitmap, only more pixels to search.
- **Tall pages are sliced into bands, not shrunk to fit.** A webtoon page can be
  20,000 pixels tall; squeezing that into the detector's window flattens the
  text to a couple of hundred pixels wide. The same page came back as
  `LMNG CANNOTENTERUN-ESS SNENPERWSSICNBY` squashed and
  `LIVING CANNOT ENTER UNLESS GIVEN PERMISSION BY` sliced, for 20% more time.
  Bands also arrive in reading order, which one whole-page pass does not
  guarantee. `RAG_OCR_BAND` and `RAG_OCR_OVERLAP` tune it; the overlap stops a
  line falling in a cut from being lost, and the duplicate it creates is
  removed by comparing letters only, so `EXPECTED THIS` and `EXPECTEDTHIS` count
  as one.

`RAG_OCR_MIN_CONFIDENCE` (0.5) drops low-scoring readings, which matters on
artwork: a detector pointed at a drawing finds "text" in hatching and panel
borders. Raise it if junk gets in, lower it if faint scans are missed.

**A rebuild does not read the pages again.** Indexing is two jobs of wildly
different cost — turning a PDF into text takes seconds, or minutes with OCR,
while chunking and embedding that text takes milliseconds — and a full reindex
used to redo both. But `reindex -Full` exists for when *chunking or embedding*
changes, and none of those alter what the words on page 12 are. So extracted
text is kept and reused. Measured on 12 scanned PDFs: **215s cold, 0.1s on the
rebuild**, byte-identical output.

It is keyed by the SHA-256 of the file's bytes, not its timestamp. Timestamps
are cheaper and are what the incremental scan already trusts, but this cache
has to survive `reindex -Full` — and distrusting timestamps is the whole reason
someone runs that. It is keyed by the extraction settings too, so turning OCR
on or changing its confidence bypasses the old entries without discarding them;
switch back and they are still there. `GET /stats` reports its size and hit
rate. The first rebuild after upgrading still pays once, since nothing is
cached yet.

Like the keyword index, it holds verbatim document text, so it is never
committed, never packaged, and dropped by `down -Wipe`.

The engine is RapidOCR on the onnxruntime this already uses for embeddings, and
its models ride inside its own wheel — so an offline install gains OCR without
fetching anything, and the bundle grows by about 65 MB. **Expect stylised text
to be imperfect.** Comic lettering and sound effects are among the harder cases;
`THE HLNTER GUILD'S` is a real line of output. It is good enough to search, not
to reprint.

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

The keyword index and entity graph that came later kept the same rule: they are
one SQLite file (`.data/graph.db`) built on `sqlite3` and FTS5 from the standard
library, so they add no dependency, no daemon, and nothing to an offline
install.

So `rag-up` creates a virtualenv, installs dependencies, indexes the folder, and
serves. That is the entire install.

## What the one command actually does

1. Creates `.venv` and installs dependencies — **first run only**;
   afterwards it reinstalls only when `requirements*.txt` actually change.
2. Downloads the embedding model (`nomic-embed-text-v1.5`, ~520 MB) into
   `.data/models` — **first run only**; later starts are fully offline.
3. Walks the folder, chunks it, embeds it, and writes the vector index — plus
   the keyword index and the entity graph, in the same pass.
4. Starts a filesystem watcher so edits re-index automatically (2 s debounce).
5. Waits until `/health` reports `healthy`, prints the URL, and opens the UI.

On every later start it also reconciles the index against the folder, so
anything added, edited or deleted while it was off is picked up. Unchanged files
are skipped on modification time, so that costs a directory walk, not a re-embed.

First run takes a few minutes (dependency install + model download + full
ingest). Every later `rag-up` is seconds — the index and the model are on disk
and are reused.

## Using it

**Upgrading an existing install** — unzip a newer release over the old folder,
then delete `.venv` so it rebuilds against the bundled wheels:

```powershell
.\rag-up.ps1 down            # release the index lock
# unzip over the top, replacing files
Remove-Item .venv -Recurse -Force
.\rag-up.ps1
```

Your `.env` survives, because the package deliberately contains none — and so
does your index, which is excluded from the zip along with the keyword index,
the extraction cache and the queue. Check the Settings tab afterwards to see
what the running configuration actually is.

**Browser** — <http://127.0.0.1:49404>. Query box, language and path-prefix
filters, a keyword toggle, a hops selector with a when-to-use guide, live index
status, and a re-index button. Results show which arms found them. A second tab
holds a [chat pane](#chat-in-the-browser), if a generator is configured.

A third tab lists **every setting and its value right now** — filterable, with
"only what I have changed" for the short version. The configuration surface is
past what anyone will read a file to discover, and someone wondering why search
feels wrong should be able to see that reranking is off without opening
`rag\.env` and remembering which of eighty-odd variables governs it. Each row
says whether the value came from `rag\.env` or is built in, and whether changing
it needs a reindex. It is read-only, and API keys are shown as set or not set —
never printed, because a browser tab gets screen-shared. Same data on
`GET /settings`.

**Shell**

```powershell
.\rag-up.ps1 query "where is the IPC rendezvous done?"
.\rag-up.ps1 ask "how does the boot handoff work?"   # the same hits as a ready prompt
.\rag-up.ps1 setup            # revisit the optional extras: OCR, chat, reranking
.\rag-up.ps1 status
.\rag-up.ps1 autostart        # start at logon and stay up (-Remove to undo)
.\rag-up.ps1 reindex          # only re-embeds what changed; -Full rebuilds
.\rag-up.ps1 logs
.\rag-up.ps1 down             # -Wipe also drops the index and model cache
```

Two more that are Python rather than PowerShell, because they are diagnostics
rather than everyday commands:

```powershell
python -m app.evaluate --compare before.json   # did that change help?
python -m app.gpucheck                         # is the GPU really doing the work?
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
| `/settings` | GET | Every setting, its value now, and where it came from |
| `/search` | POST | Ranked hits, snippet-truncated |
| `/search/full` | POST | Ranked hits, full chunk text |
| `/context` | POST | The same hits, assembled into a citable block for a generator |
| `/chat` | POST | A cited answer, streamed. 501 unless a generator is configured |
| `/chat/config` | GET | Which generator answers by default, and where it sends the passages |
| `/chat/models` | GET | Every generator this machine can reach, for the dropdown |
| `/corpus/count` | POST | How many documents match, grouped by year, folder or extension |
| `/classify` | POST | Decide one question about every row of a spreadsheet |
| `/classify` | GET | Progress, rate and summary; `?verdicts=true` for the rows |
| `/classify/cancel` | POST | Stop after the batch in flight, keeping what was decided |
| `/classify/csv` | GET | The verdicts as a CSV |
| `/queue` | GET | Pending, retrying and dead-lettered index work |
| `/queue/retry` | POST | Requeue dead letters, optionally one `path` |
| `/entities` | GET | Names the index knows about, most-mentioned first |
| `/graph/neighbors` | POST | What shares documents with a name, with citations |
| `/graph/path` | POST | How two names connect, hop by hop |
| `/reindex` | POST | `{"full": false}` — background re-ingest |
| `/api-docs` | GET | Generated OpenAPI docs |

Search body fields: `query` (required), `top_k` (default 5), `language_filter`
(`rs`, `markdown`, `toml`, `ld`, `yaml`, `json`, `conf`, `sh`, `ps1`, `py`,
`asm`), `path_prefix` (e.g. `kernel/src`), `hybrid` (fuse BM25 with the vector
ranking, default on), `hops` (graph expansion, default off).

**Check `.status`, not `.ollama`.** Native mode has no Ollama, so `/health`
reports `embeddings` and `embed_backend` instead; `ollama` appears only when
that backend is actually in use.

## Keeping it running, and keeping it current

Three separate mechanisms, because no one of them covers everything:

**The watcher** re-indexes a file seconds after it is saved, and drops it from
both stores when it is deleted or moved away. It lives inside the server
process, so it only runs while the server does.

**The startup reconcile** catches everything the watcher could not see because
nothing was running: files added, edited, renamed or deleted while the machine
was off. It compares each file's modification time against what the index
recorded, so unchanged files are skipped and a quiet corpus costs a directory
walk and no embedding.

**The periodic rescan** is the same sweep on a timer, for corpora where the
watcher cannot be trusted. It defaults to every 15 minutes when watching is off
(a UNC share) and off when the watcher is running; `RAG_RESCAN_MINUTES` sets it
either way.

That also makes an ordinary re-index cheap — it re-embeds only what changed:

```
ingest complete: 3 indexed, 79 unchanged, 1 removed, 9 skipped
```

### Autostart

```powershell
.\rag-up.ps1 -Folder "S:\Team\Documents"   # once, so the corpus is in .env
.\rag-up.ps1 autostart                     # start at logon, stay up
.\rag-up.ps1 autostart -Remove             # undo
```

This registers a Scheduled Task that starts the server at logon **and re-runs
every 10 minutes**. The repeat is the point: a task that only fired at logon
would leave a crashed server dead until the next reboot, whereas re-running the
launcher costs nothing when the server is healthy — it prints one line and
exits — and revives it when it is not. `-EveryMinutes` changes the interval.

`autostart -System` runs it at boot as SYSTEM instead, so it comes up with no
one logged in. That is genuine 24/7, with one catch worth understanding:
**SYSTEM has no mapped drives and no share credentials**, so if your documents
live on a network path it will index nothing. For a share, use the default
per-user task, which runs as you.

The task is refused if no corpus is recorded in `.env` — a scheduled run cannot
see the environment of the shell you registered it from, so it would fail every
ten minutes in a hidden window with nobody watching.

**What still needs a hand:** a *directory* deleted wholesale is not always
reported per-file by Windows, so its documents are pruned by the next reconcile
rather than instantly. Startup and the periodic rescan both cover it.

### Nothing is indexed on a promise

A change is **written down before it is acted on**. The watcher records what it
saw in a small SQLite queue; a worker drains it. That matters because the moment
a file changes is exactly the moment it is least readable — Word still holds the
handle, a network copy has landed in pieces — and indexing inline meant one
failed read discarded the change for good.

Failures back off (5s, 10s, 20s…) and retry. After `RAG_QUEUE_MAX_ATTEMPTS` an
item is parked in a dead-letter table, where it stops burning retries and starts
being evidence:

```powershell
Invoke-RestMethod http://127.0.0.1:49404/queue        # pending, retrying, dead
Invoke-RestMethod http://127.0.0.1:49404/queue/retry -Method POST `
  -Body '{}' -ContentType 'application/json'          # requeue after a fix
```

The queue is on disk, so work survives a crash or a reboot — the point of
running unattended for weeks.

## Sections, not just line ranges

Each chunk records the heading it sits **under**, which is different from the
label describing the chunk itself. Two things follow.

**Citations say where in the document a passage lives:**

```
[2] Volume_I/ch08_cutting_documents_into_pieces.md:25-40 (The Problem)
```

**And a hit can be widened back to its section.** A chunk is a retrieval unit,
not a unit of meaning: a long section is cut into several, so the paragraph that
matched is frequently the middle of an argument whose beginning is the chunk
above it. With `expand` on (the default), `/context` rebuilds the contiguous run
of chunks sharing that heading and cites the widened range:

```powershell
$body = @{query='why does chunk size matter?'; top_k=4; expand=$true} | ConvertTo-Json
$r = Invoke-RestMethod -Uri http://127.0.0.1:49404/context -Method POST -Body $body -ContentType 'application/json'
$r.sources | Select-Object n, citation, heading, expanded
```

Only the *contiguous* run: a heading repeated later in the file is a different
section with the same name, and splicing the two would invent a passage that
does not exist. An expanded section is capped at half the character budget, so
one long section cannot crowd out every other answer. `RAG_CONTEXT_EXPAND=0`, or
`"expand": false` per request, turns it off — worth doing for wide surveys where
breadth beats depth.

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

That writes `rag-portable.zip` (~780 MB) beside the folder. Copy it, unzip on
the other PC, and run `Rag.bat`.

Left out on purpose:

| Excluded | Why |
| --- | --- |
| `.venv` | records absolute paths to the Python that built it; rebuilt on arrival |
| `.data\qdrant` | the index of *this* machine's corpus |
| `.data\graph.db` | the same documents again — FTS5 stores a verbatim copy of everything it indexes |
| `.data\context-cache` | cached passages, also verbatim document text |
| `.git` | a tool being delivered, not a checkout being cloned |
| `rag.pid`, `rag.log` | describe a run on the machine that built the package |

The corpus path is stripped from the packaged `.env` too, so the target asks for
its own folder instead of pointing at a path that does not exist there. Between
them, that is what lets you hand someone the zip without handing them your
documents.

If you would rather copy the folder directly:

```powershell
robocopy . D:\rag /E /XD .venv .data\qdrant .data\context-cache /XF graph.db* rag.pid rag.log*
```

**Match the Python minor version.** Binary wheels (onnxruntime, numpy,
pydantic-core, lxml) are built per Python minor version, so wheels fetched on a
3.14 machine will not install on a 3.12 one - and that only surfaces at the
target, where there is no network to recover. If the versions differ, add the
target's before travelling:

```powershell
.\rag-up.ps1 bundle -ForPython '3.12,3.13'
```

On the other PC, run `Rag.bat` and answer the folder question (or set
`RAG_REPO_MOUNT` in `.env` yourself). It installs from the vendored wheels
(`--no-index`) and loads the bundled model, so it never reaches the network.
Then, if it should stay up on its own:

```powershell
.\rag-up.ps1 autostart
```

### What to copy, and what not to

| Item | Copy it? | Why |
| --- | --- | --- |
| `app/`, `*.ps1`, `*.sh`, `requirements*.txt` | **yes** | the program |
| `.env` | **yes**, then edit | `RAG_REPO_MOUNT` differs per machine |
| `rag-client.psm1`, `rag-projects.example.json` | **yes** | the PowerShell client and its config template |
| `vendor/wheels/` | yes, for an offline target | dependencies without PyPI |
| `.data/models/` | yes, for an offline target | ~610 MB, avoids huggingface.co |
| `.data/qdrant/`, `.data/graph.db` | optional, and only to a machine allowed to hold the documents | a prebuilt index — skips the first ingest, but both contain the corpus |
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

### Somebody else's embedding service

`RAG_EMBED_BACKEND=http` points the index at an OpenAI-shaped `/embeddings`
endpoint — hosted, or a company-internal gateway — instead of the in-process
model:

```ini
# .env
RAG_EMBED_BACKEND=http
RAG_EMBED_URL=https://ai.internal/v1
RAG_EMBED_MODEL=whatever-the-service-calls-it
RAG_EMBED_API_KEY=...
RAG_EMBED_DOC_PREFIX=search_document:
RAG_EMBED_QUERY_PREFIX=search_query:
```

Those last two matter more than they look. Strong retrieval models are trained
asymmetrically — nomic expects `search_document:` on passages and
`search_query:` on questions, e5 wants `passage:` and `query:`. `fastembed`
applies them for you; a raw HTTP endpoint does not, so leaving them empty
against such a model degrades ranking while appearing to work. Set them to
whatever the service's model was trained with.

Replies are sorted by the `index` the server reports rather than by arrival,
because responses are not guaranteed to come back in order and a shuffled one
would attach every vector to the wrong chunk — nothing would error, search
would simply return nonsense. Batches are capped (`RAG_EMBED_HTTP_BATCH`) and
retried with backoff, since an ingest against a rate-limited gateway is the
case this has to survive rather than the exception.

To decide whether a different service is actually better, index each into its
own `RAG_COLLECTION` and use [the eval harness](#knowing-whether-a-change-helped)
— that turns "it feels sharper" into a number.

### Running the models on the GPU

`RAG_GPU=1` runs both ONNX models on CUDA — the embedder and the cross-encoder
reranker.

It speeds up a first index over a large corpus, which is the obvious win.
What it does for *query* latency is less obvious, and worth spelling out
because the intuitive answer is wrong:

Embedding the question is a rounding error — one short forward pass. Reranking
is one forward pass *per retrieved candidate*, over passages of
`RAG_CHUNK_CHARS` each, repeated for every re-search round. So with reranking
off a search is milliseconds and the GPU changes nothing you can feel; with it
on, the reranker is essentially the entire cost.

Measured on one box (RTX 5090, CUDA 13.3, 150 candidates), via
`python -m app.gpucheck`:

| | CPU | GPU | |
| --- | --- | --- | --- |
| embedding | 8.19 ms/row | 0.47 ms/row | 17.3× |
| reranking | 1.40 s/query | 0.06 s/query | **23.4×** |
| end-to-end `POST /search` | 12.2 s | 0.36 s | 34× |

The end-to-end figure beats both component ratios because a search reranks more
than once — the query-expansion round pays the cost again.

> Until v1.10.0 this setting was `RAG_EMBED_GPU` and reached only the embedder.
> Turning it on therefore did nothing measurable for query latency, and the
> docs here said so — which read like a fact about GPUs when it was a missing
> argument. `RAG_EMBED_GPU` is still honoured; it meant "use my GPU" and now
> gets all of it.

It needs `onnxruntime-gpu`, which **replaces** `onnxruntime` rather than sitting
beside it — they share an install directory, so having both silently shadows
the GPU build and uninstalling either breaks the other. Install it deliberately,
after the normal requirements:

```powershell
.\rag-up.ps1 gpu
```

That swaps `onnxruntime` for `onnxruntime-gpu`, runs the proof below, and only
writes `RAG_GPU=1` if it passes. By hand it is:

```powershell
pip uninstall -y onnxruntime
pip install --force-reinstall "onnxruntime-gpu[cuda,cudnn]"
```

**For machines with no internet** — vendor the wheels before you travel, on a
machine that has it:

```powershell
.\rag-up.ps1 bundle -Gpu       # base wheels + models + CUDA, both generations
.\rag-up.ps1 package -Gpu      # include them in the zip
```

On the other machines there is **nothing to run**. Unzip, double-click
`Rag.bat`, and the installer detects the card, reads the CUDA version its
driver supports, installs the matching wheels, proves they work, and only then
writes `RAG_GPU=1`. That holds for a scripted `-Folder` rollout too, which
never reaches the setup questions.

None of it is guesswork about the target, which matters because every wrong
guess here fails the same quiet way — a fall back to the CPU that reports
success:

| decided on the target | how |
| --- | --- |
| is there a GPU | `nvidia-smi` |
| CUDA 12 or 13 | the version in the `nvidia-smi` header, which is the highest the *driver* supports |
| which Python | pip resolves against the vendored wheels |
| did it actually work | `app.gpucheck`, before anything is enabled |

Sizes, which decide whether this is one zip or two:

| | |
| --- | --- |
| the seven `nvidia-*` wheels (`py3-none-win_amd64`) | 1,080 MB per CUDA generation, one copy serves every Python |
| `onnxruntime-gpu` (per interpreter) | 230 MB each |
| one generation, Python 3.11–3.14 | **~2.0 GB** |
| both generations, Python 3.11–3.14 | **~4.0 GB** |

`-Cuda 13` or `-Cuda 12` halves it when the fleet is known to be uniform;
`both` is the default because a mixed fleet is the case that silently
half-works. The Python versions are taken from whatever the base wheel set
already covers, so the two cannot drift apart — override with `-ForPython`.

The pack is checked before you leave: `bundle -Gpu` asks pip to resolve
`requirements-gpu.txt` against nothing but the vendored directory, once per
Python version per CUDA generation, and names any combination that fails.
Counting wheels proves nothing — a set can hold every package and still not
resolve, and the place that discovers it is otherwise the target machine,
offline.

Both switches are opt-in. Without them the CUDA wheels are neither fetched nor
packaged, so a machine with no GPU still gets a 371 MB wheel set.

The `[cuda,cudnn]` extras matter. The bare wheel does not bundle CUDA or cuDNN,
and without them onnxruntime offers `CUDAExecutionProvider`, accepts it, then
logs `Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.*`
and runs on the CPU — which looks exactly like having no GPU.

Match the build to the machine's CUDA: the default PyPI wheel wants CUDA 13,
while CUDA 12 hosts need the separate index documented in `.env.example`.

On Windows those extras install their DLLs under `site-packages/nvidia/*/bin`,
which is *not* on the DLL search path — so the libraries are present, correct
and invisible, and you get the same silent CPU fallback. The server calls
`onnxruntime.preload_dlls()` before building any session to fix that, so this
needs nothing from you; it is noted only because the symptom is indistinguishable
from a missing GPU and cost a round of debugging here.

**Check it actually took**, because the obvious check lies. When the driver
libraries are missing, onnxruntime logs a warning and runs on the CPU rather
than failing, so a benchmark reports a small speedup that is CPU on both sides.
`onnxruntime.get_available_providers()` saying `CUDAExecutionProvider` only
means a library is present on disk.

```powershell
python -m app.gpucheck
```

That asks for evidence a CPU run cannot fabricate: which provider the *loaded
session* chose, whether GPU memory actually moved, whether it is faster once
warmed up, and whether the output still matches the CPU's — a silently
different embedding would poison an index, and a reranker that scores subtly
differently would reorder results, in ways nothing would report. It also runs a
negative control with CPU forced and must see CPU, because a check that cannot
fail proves nothing. Exit code is non-zero until every part passes, and the
failure says which one did not.

It checks **both** models and reports them separately (`--what rerank` for just
the cross-encoder). Read the reranker's verdict first: it is the one that
decides how long a search takes, so a green light on the embedder alone is a
green light on the wrong lamp — which is precisely how the reranker went six
releases with no provider wiring at all.

`GET /health` reports `embed_provider` and `rerank_provider` for the same
reason, as a standing reminder of what onnxruntime chose rather than what was
requested.

**On a host with several GPUs**, set `RAG_GPU_DEVICE`. A provider named as a
bare string means device 0, so without this every process that never considered
the question lands on the same card — usually the one someone else is training
on. Requesting a device that does not exist doesn't fail either; it falls back
to CPU. `/health` therefore also reports `embed_device` and `rerank_device` as
the sessions actually bound them, and `python -m app.gpucheck --device 1` probes
a specific card.

Changing the model changes the vector width, so follow it with
`.\rag-up.ps1 reindex -Full` — and it will tell you if you forget, since the
settings that built the index are [recorded and
checked](#when-a-setting-changes-under-a-built-index). That rebuild re-embeds
but does not re-read the documents, so on a corpus of scans it costs minutes
rather than the hours the first pass did. Narrowing the corpus is the other
lever — point `-Folder` at the subtree you actually search rather than a whole
shared drive.

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

## Chat in the browser

The Chat tab is that seam with the wiring already done: it retrieves, hands the
passages to a model you pick, and streams back an answer whose claims carry
`[1]` `[2]` markers you can click to see the file and lines behind them.

**An "Answered by" dropdown lists whatever this machine can reach**, and the
list is discovered rather than configured — a list in a config file is wrong
the moment someone pulls a new model. Ollama is asked what it holds and `PATH`
is checked for signed-in coding agents, so `ollama pull deepseek-r1` puts
Deepseek in the dropdown on the next page load with nothing to edit. Embedding
models are filtered out: they answer `/api/chat` with nonsense rather than an
error, and this indexer pulls one itself, so offering it was a guaranteed
puzzle rather than a hypothetical one. Your choice is remembered across
reloads.

**Claude Code and Codex work as backends** if they are installed. They are
command line tools rather than HTTP services, so the server drives them as
subprocesses — someone with a paid-up agent already on the machine should not
have to go and find an API key to use it from the pane.

**It stays hidden until there is something to use**, and that is a real default
rather than a prompt to set something up. With nothing found, the tab does not
appear, `POST /chat` answers 501, and search behaves exactly as it did. That
matters because this installs on machines with no internet and no local model,
where a chat box that failed on every message would be worse than no chat box.

To point it at something the discovery cannot find — a hosted API, or a model
on another machine — set it in `rag\.env`:

```ini
# A model on this machine or the local network
RAG_CHAT_PROVIDER=ollama
RAG_CHAT_MODEL=gemma3:12b
RAG_CHAT_URL=http://gpu-box.example.lan:11434

# Anything speaking OpenAI's protocol - OpenAI and its Codex models, and
# equally llama.cpp's server, LM Studio, vLLM, text-generation-webui
RAG_CHAT_PROVIDER=openai
RAG_CHAT_MODEL=gpt-4o-mini
RAG_CHAT_URL=https://api.openai.com/v1
RAG_CHAT_API_KEY=sk-...

# Claude
RAG_CHAT_PROVIDER=anthropic
RAG_CHAT_MODEL=claude-opus-5
RAG_CHAT_API_KEY=sk-ant-...
```

Restart the server after changing these. No new Python package is needed for any
of them: all three are HTTP, spoken through the `httpx` this already depends on,
so an offline install gains a chat pane without gaining a single wheel.

**Where your documents go is printed above the conversation.** Retrieval never
leaves this process - the index, the embeddings and the files stay here - but
the question and the passages that answered it are sent to whatever model you
named. For a local model that is the same machine. For a hosted API it is not,
and the pane says so in orange, with the endpoint spelled out, because a tool
that quietly posts a private corpus to a third party is the one failure this
must not have. `rag-up package` strips `RAG_CHAT_API_KEY` from the zip for the
same reason.

Some details that decide whether the answers are any good:

- **The answer is written from the passages, and says when they fall short.**
  The instruction it works under is to report a retrieval miss rather than
  answer from general knowledge - a fluent wrong answer is indistinguishable
  from a right one, which is exactly what makes it expensive.
- **The filters above the box apply to chat too.** Language, path prefix,
  result count, keyword fusion and hops all shape what gets retrieved for each
  question, so you can scope a conversation to one folder.
- **Short follow-ups carry their subject.** "And the M10?" retrieves nothing on
  its own, so a question under 80 characters is searched together with the
  previous one. Crude, but it costs no extra model call and is easy to reason
  about when the citations look wrong.
- **Untick "search the index" for a question about the last answer.** "Shorten
  that" or "what does that acronym mean" needs no new retrieval, and skipping it
  keeps unrelated passages out of the prompt.
- **Nothing is stored.** The conversation lives in the browser tab; the server
  keeps no session, so a reload starts fresh and two tabs never collide.
- **Give a small local model room.** `RAG_CHAT_CONTEXT_CHARS` (12000) is the
  passage budget and `RAG_CHAT_TOP_K` (8) the passage count - lower both for a
  model with a small context window, raise them for a large one.

- **A reasoning model says it is thinking.** One streams its reasoning before
  its answer; that reasoning is not the answer and is dropped, but dropping it
  silently made a slow model look like a hung one — minutes passing with the
  stream alive and nothing on screen. The pane now says so.

`POST /chat` is the same thing over HTTP if you would rather drive it yourself:
send `{"messages": [{"role": "user", "content": "..."}]}`, optionally with
`"backend"` from the list below, and read back server-sent events — `sources`
once, then a `delta` per fragment, then `done`, with `status` while a model
thinks or searches again. `GET /chat/models` lists what is available and which
answers by default; `GET /chat/config` describes that default, or reports that
there is none.

## The PowerShell client (`rag-client.psm1`)

Everything above is the raw HTTP contract. `rag-client.psm1` is the same
contract with the repetitive parts filled in: project scoping, two query modes,
acronym normalisation, one standard prompt, a literal-search fallback and a
per-project context cache. It adds nothing to the server and runs no language
model - what it returns is the prompt you feed to one.

```powershell
Import-Module .\rag-client.psm1 -DisableNameChecking

Set-RagProject -Name CHORD -PathPrefix 'Projects/CHORD' -Alias ngpcn
Ask-Rag 'How is CHORD NGPCN deployed to prod?'    # scope inferred from the alias
(Ask-Rag 'who owns the prod runbook?').Prompt | Set-Clipboard
```

`-DisableNameChecking` only silences PowerShell's warning that `Ask-` is not
one of its approved verbs.

Or without importing anything:

```powershell
.\rag-up.ps1 ask "how is CHORD deployed?" -Project CHORD          # prints the prompt
.\rag-up.ps1 ask "explain the architecture" -Project CHORD -Deep
```

**Project scoping.** Querying the whole of Documents for a question about one
project is mostly noise. Projects live in `rag-projects.json` (copy
`rag-projects.example.json`, or let `Set-RagProject` write it) and map a name to
a `path_prefix`, some aliases and an optional default language filter. A
question that names a project or one of its aliases is scoped to it
automatically; `-Project` forces it and `-NoProject` searches everything.
`Get-RagPrefix` lists the corpus subdirectories, so the prefixes can be read off
the real layout rather than guessed.

**Two modes.** `Quick` (the default, `top_k` 5) is lookup: *where is this
documented?* `Deep` (`top_k` 25, `max_chars` 20000) is for overviews and
explanations. `Ask-RagQuick` and `Ask-RagDeep` are the shorthands, and the
numbers live under `modes` in the config file.

**One prompt.** Every question is assembled into the same instruction block -
answer only from the numbered context, cite by number, say you do not know
rather than filling the gap. Consistency here does more for answer quality than
any single retrieval knob.

**Grep fallback.** When retrieval returns nothing - or, with reranking off,
scores below `fallback.min_score` - the client also runs a literal search over
the same subtree (ripgrep if it is on PATH, `Select-String` otherwise) and
appends those lines as extra numbered evidence, marked `origin = grep`. Short
queries stay a literal phrase, which is the ticket-ID and error-code case this
exists for; longer ones become an alternation over their most distinctive words.
Plain-text file types only: grep cannot read a PDF.

**Context cache.** The 20-30 chunks that answer "what is this, who owns it,
which environments" are the same for most questions about a project.

```powershell
Update-RagContextCache -Project CHORD      # once, or after a big document drop
Ask-RagDeep 'why does staging differ from prod?' -Project CHORD -UseCache
```

Cached background is capped at `cache_share` (default 40%) of the character
budget so the hits that actually answer the question are never the ones dropped,
and reading a cache older than a week warns.

**Query rewriting.** `synonyms` in the config normalises variants to one
canonical form (`NG PCN`, `Next Gen PCN` -> `NGPCN`) before embedding, per
project via `expand` or globally; the project name is appended when the question
does not already contain it; and a question over `long_query_chars` is reduced to
its content words, since long prose embeds toward its own filler. `-NoRewrite`
turns all of it off, and `Expand-RagQuery` shows what a question becomes.

| Function | Purpose |
| --- | --- |
| `Ask-Rag`, `Ask-RagQuick`, `Ask-RagDeep` | Retrieve, then assemble the standard prompt |
| `Get-RagContext` | The same retrieval without the prompt wrapper |
| `Invoke-RagSearch` | Ranked hits, project-scoped (`-Full` for whole chunks) |
| `Find-RagLiteral` | Literal search over the corpus, scoped the same way |
| `Get-RagProject`, `Set-RagProject`, `Remove-RagProject` | Manage projects |
| `Get-RagPrefix` | Candidate `path_prefix` values from the real corpus layout |
| `Get-RagEntity` | Names the index knows about (`-Kind identifier` for exact-match strings) |
| `Get-RagNeighbor` | What shares documents with a name, with evidence citations |
| `Get-RagPath` | How two names connect, hop by hop |
| `Update-RagContextCache`, `Get-RagContextCache`, `Clear-RagContextCache` | Per-project background context |
| `Expand-RagQuery` | Show what a question becomes before it is embedded |

`-Hops 1` on `Ask-Rag`, `Get-RagContext` and `Invoke-RagSearch` turns on graph
expansion for that question; `-NoHybrid` drops back to vectors alone.

`Ask-Rag` returns an object: `.Prompt`, `.Context`, `.Sources`, `.ChunksUsed`,
`.ChunksDropped`, `.Fallback`, `.Ranking`, `.Graph`, plus the scope it resolved. `-Raw` returns the
prompt string alone; `-OutFile` writes it. `$env:RAG_API` or `-Api` points the
client at a server on another port or host.

## Keyword search and the entity graph

Vector search has two blind spots, and both are fixed by one SQLite file next
to the vectors (`.data/graph.db`). `sqlite3` is in the standard library and the
bundled interpreter has FTS5, so this costs no dependency, no daemon and
nothing in an offline install. It is built during ingest — a corpus indexed
before this existed needs `Rag.bat reindex -Full` to populate it.

**1. Exact strings.** Embeddings are worst at precisely what people search for
most confidently: ticket IDs, part numbers, standards, error codes. `NW-2200`
means nothing in vector space. So the same chunks are also indexed for BM25,
and the two rankings are fused rather than chosen between:

```powershell
$body = @{query='NW-2200 pressure fault'; top_k=8} | ConvertTo-Json
$r = Invoke-RestMethod -Uri http://127.0.0.1:49404/search -Method POST -Body $body -ContentType 'application/json'
$r.ranking                      # vector, keyword, rerank - what produced the order
$r.results[0].origins           # which arms found this particular chunk
```

Ranks are fused, not scores: a cosine similarity, a BM25 value and an entity
count share no scale, and normalising them against each other invents a
comparison that does not exist. A chunk two arms both liked outranks one that a
single arm liked slightly more. `"hybrid": false` in the body turns it off.

**Check `.ranking` before reading `.score`.** It is a cosine similarity with
one arm, a fused rank weight with several, and an unbounded cross-encoder logit
when reranking is on — the three are not comparable, and only the caller knows
which one it is looking at.

**2. Multi-hop.** *"Which supplier makes the part that failed?"* is two lookups
chained through a shared name, and one similarity search cannot do it. During
ingest, each chunk's entity mentions are recorded; `hops` resolves the names in
a question to entities, walks out through the documents they share, and adds
those chunks as candidates:

```powershell
$body = @{query='what does the part that failed at Harborline cost?'; top_k=8; hops=1} | ConvertTo-Json
$r = Invoke-RestMethod -Uri http://127.0.0.1:49404/context -Method POST -Body $body -ContentType 'application/json'
$r.graph.seeds      # Harborline, NW-2200
$r.graph.reached    # SKU, List Prices, Hydraulic, Lead Time, NW-2260 ...
```

There are deliberately **no triples**. A stored "A supplies B" would be a claim
this stack cannot check without a language model, and extracting it reliably
needs a dependency parser — a hundred megabytes per Python version in a bundle
that has to install offline. Co-occurrence claims only that two names appear
together, which is all retrieval needs: the graph widens the candidate set and
the reranker narrows it again. Entity extraction is regex and a stop list over
text already in memory, so ingest costs a few percent more, not a multiple.

Edges are derived from the mentions table rather than stored, which is why an
incremental re-ingest cannot leave a stale one behind, and why every edge comes
with the passages it rests on:

```powershell
Get-RagNeighbor 'NW-2200' -Hops 1        # SKU, List Prices, Hydraulic, Lead Time
Get-RagEntity -Kind identifier            # what the index would match exactly
Get-RagPath -From 'Harborline' -To 'NW-9001'
```

**What to expect.** The keyword arm earns its place immediately — on a test
corpus here it returned five or six chunks per query that the vector arm never
surfaced. The graph arm is a recall mechanism: it changes what enters the
candidate pool more than it changes the top few, especially with reranking on
and `RAG_RERANK_CANDIDATES` set high, since the cross-encoder is then already
reading a wide pool. Turn `hops` on for "who else was involved", "what depends
on this", "trace this part across documents"; leave it off for questions about
one passage.

**The df ceiling matters.** An entity in more than `RAG_GRAPH_MAX_DF` of the
documents (default 20%) is treated as boilerplate and dropped from the graph —
a header, a footer, or the corpus's own subject connects everything to
everything. On a mixed document share this is what keeps the graph clean. On a
single-subject corpus it is too aggressive: index one book and its main
characters appear in most chapters, so the default hides exactly the names you
would ask about. `GET /entities?include_common=true` shows what is being
filtered, and the ceiling itself.

**Known limits.** Entity extraction is patterns, not a model: it catches
capitalised names, acronyms and identifier-shaped strings, and misses lower-case
ones (a domain in an email body, a product written in running text). PDF text
with hyphenation and wrapped lines extracts worse than Markdown. `/graph/path`
is best-effort — on prose it can route through a weak link, so read the
evidence citations before believing a chain.

## "How many are there?"

Some questions are not about what documents say, and no amount of better
retrieval answers them:

> How many CISAR forms have been signed this year?

Search returns the handful of passages most like the question. Counting needs
every matching document and none of their text. Asked this, a chat model gets
five chunks, correctly notices they contain no total, and says so — sometimes
adding that it cannot see your files, which is the wrong explanation for the
right refusal.

So counts are computed over the index and handed to the model as a fact:

```powershell
$body = @{match='CISAR'; group_by='year'} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:49404/corpus/count -Method POST -Body $body -ContentType "application/json"
# matched: 142, total_indexed: 5000
# groups:  2026: 42, 2025: 60, 2024: 40
```

In the Chat tab this happens by itself — ask "how many CISAR forms were signed
this year" and the model requests the count, the server computes it, and the
answer quotes a number it did not have to work out. **The model is never asked
to tally a list**: a language model counting twenty-nine filenames is a coin
toss, and `SELECT COUNT` is not.

`match` is a substring of the path, or a glob if it carries `*` or `?`.
`group_by` takes `year`, `month`, `folder`, `extension` or `none`, and
`path_prefix` scopes to a subtree.

Three things worth knowing about what the number means:

- **Documents are counted, not chunks.** One PDF is many chunks, and "how many
  forms" means files. The distinction is invisible until it is wrong by a
  factor of four.
- **Dates come from filenames first.** A form archive names its files by the
  date on the form, and `Signed - A.CISAR.20260225.pdf` was signed in 2026
  whatever the filesystem thinks — a file copied to a new machine this year has
  not been signed this year. Only when *no* filename in the matching set
  carries a date does modification time get used at all, and the reply says so.
  Files without a date are counted as `undated` rather than guessed at.
- **It counts what is indexed**, which is not quite what is in the folder: a
  document that failed to extract is on disk but not in the index. The reply
  gives the indexed total alongside the match count so the two are comparable.

Today's date is included in the facts given to the model, because "this year"
is unanswerable without a clock and a model has none.

## Deciding one thing about every row of a spreadsheet

A different shape of question again: sixteen thousand failure modes, one
document defining what makes a failure mode built-in-test applicable, and a
verdict needed for each row. Done a row at a time through retrieve-and-ask it
cost about 1.5 seconds a row — roughly six and three quarter hours, before
anyone disagrees with a verdict and wants it re-run.

Almost none of that time was doing anything useful, and where it went decides
what to fix. **Embedding a row costs 27 ms and searching for it a few more —
under two per cent.** The rest is a model writing one verdict. So a faster
embedder, or a GPU under it, buys nothing here. Three things do:

- **The criteria never change.** Every row retrieved substantially the same
  passages. Retrieved once and reused, that work disappears — and every row is
  judged against identical wording, which matters more for a determination
  someone has to defend than the speed does.
- **Rows repeat.** The same failure mode recurs across many line items with
  different identifiers. Decided once per distinct case and mapped back, the
  model sees a fraction of the sheet.
- **A verdict is small.** One call carries many rows, so the fixed cost of a
  call is paid per batch rather than per row.

```powershell
$body = @{
  path          = 'FMEA.xlsx'
  columns       = @('Failure Mode', 'Local Effect')
  criteria_query= 'when is a failure mode built-in test BIT applicable'
  labels        = @('applicable', 'not applicable')
  id_column     = 'FM_ID'
  batch_size    = 50
  concurrency   = 4
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:49404/classify -Method POST -Body $body -ContentType "application/json"
# rows: 16000, distinct_cases: 168, repeats_collapsed: 15832
```

`GET /classify` reports progress and a rate; `POST /classify/cancel` stops after
the batch in flight and keeps what was decided; `GET /classify/csv` downloads
the verdicts with the columns that identify each row. One job runs at a time,
on a worker thread, so search stays answerable throughout.

Every verdict carries **which criterion decided it**, because "applicable"
alone is an opinion and "applicable, by the continuous-monitoring clause" is a
determination.

**Batch size is an accuracy dial that happens to also make things faster.**
Past some width a model starts pattern-matching across rows instead of judging
each on its merits, and where that begins depends on the model and the rows.
Point `truth_column` at a column of known-correct answers and the reply carries
an agreement rate — run 1, 10, 25, 50 and watch where it falls off. That is how
"we ran it at 50" becomes an auditable choice rather than a guess.

**Nothing is dropped silently.** A batch whose reply cannot be parsed, or comes
back missing rows, is retried in halves and finally one row at a time; a row
that fails alone is recorded with its reason. Sixteen thousand determinations
with four unexplained gaps is a worse outcome than one that took longer.

Two practical notes. **Use an HTTP backend, not a CLI one** — `claude` and
`codex` spawn a process per call, five to seven seconds that is invisible in
chat and an hour of pure overhead across several hundred batches. And
`concurrency` only pays against a model server that genuinely runs requests in
parallel; past that, requests queue and the only thing that grows is how long
each appears to take.

## When your words and the document's words differ

Retrieval matches the wording of your question against the wording of the
documents. When the two disagree about the same thing, the passage that answers
you is simply absent — and nothing says so. A manual that says "torque
specification" does not match a question about "how tight". A status window
lettered `DAILY QUEST` does not match a question about "the System". Every arm
works correctly and the answer is nowhere in the results.

Two things address it, and they fail differently, which is why there are two.

**A deterministic pass, with no model involved.** Your question is stripped of
its scaffolding — "what", "does", "when" — for one keyword search, then the
vocabulary *that* pass turned up is used for another, and both are fused in as
extra rankings. This is pseudo-relevance feedback, it costs two more BM25
queries against SQLite, and it works with any chat model or none. It improves
`POST /search` and `POST /context` as much as the chat pane.

Which words get reused took three attempts, and the failures are worth knowing
if you tune it. Ranking by raw frequency harvested `it's`, `that's` and the
studio credit printed on every chapter's last page — words in the sample
because they are in *everything*. Filtering for rarity instead produced
`abovemep` and `actoally`: on OCR'd pages the rarest words are the scanning
errors, each appearing once because it is not a word. What works is a band —
common enough to have been read correctly somewhere, rare enough to still
discriminate. `RAG_EXPAND_MAX_SHARE` is the ceiling, and there is a floor of
three chunks underneath it.

**The model can ask for a different search.** When the passages fall short it
may reply `SEARCH: <terms>`, guessing what the documents themselves would call
the thing; the server runs that and hands back new passages. This is the part
that crosses a genuine vocabulary gap, which the deterministic pass cannot —
the bridging words are in neither your question nor the first results, so no
amount of re-reading them will produce it.

It is a line of text rather than a tool API, so any instruction-following model
can use it, including a small local one. Bounded by `RAG_CHAT_MAX_SEARCHES`
(two) because each round costs a full generation and a model that kept
searching would never answer. Both rounds' passages are renumbered into one
list, so a citation in the answer still resolves against what you were shown.

## When a setting changes under a built index

Several settings decide what ends up in the vectors — the embedding model, the
chunk size, which label rides along with the text. Change one and the chunks
already indexed keep whatever the old setting produced, while everything
indexed afterwards uses the new one. You end up with an index built two
different ways, and nothing says so: search keeps working and returns slightly
worse answers, blaming nothing.

So the settings are written down beside the index when it is built and checked
against the running configuration at startup:

```
⚠ The index was built with different settings: the embedded label
  ('symbol' -> 'section'). Chunks indexed before and after this change are not
  comparable - run `reindex -Full` to rebuild them consistently.
```

It appears in the browser header, in the server log, and as `stale` on
`/health`; `/stats` carries the full before-and-after detail.

Two kinds of drift are distinguished, because the consequences differ. Changing
**the model, chunk size, vector width or embedded label** makes old and new
chunks incomparable, and only a full reindex fixes it. Changing **what gets
extracted** — OCR, archives, extra text extensions — corrects itself file by
file as each is next indexed, so a reindex only hurries it along. The message
says which of the two you are looking at.

Drift is reported, never repaired. Rebuilding an index is minutes to hours of
your time and that is your call to make, not the tool's. An index built before
this existed has no record, which is reported as unknown and never warned
about — being nagged into a two-hour rebuild to satisfy a bookkeeping file
would be a poor trade.

## Knowing whether a change helped

Every knob here — reranking, hybrid fusion, hops, query expansion, what goes
into the embedded text — is a judgement call, and judging one by running a few
searches and squinting tells you almost nothing. Five questions can only tell
you a change helped those five. The ones it quietly broke are, by definition,
the ones nobody typed.

So: a list of questions with the documents that ought to answer them.

```powershell
Copy-Item rag-eval.example.json rag-eval.json   # then edit it
python -m app.evaluate --save before.json
# ...change something, reindex if it touched embedding...
python -m app.evaluate --compare before.json
```

```
       #1 -> #1    how do I keep it running after I log out?
 DN    #3 -> #4    which model writes the answers?
     MISS -> #2    what does the System say when it first appears?

  10 questions   hit@1 70%  hit@3 90%  hit@5 100%   MRR 0.808
  was:           hit@1 70%  hit@3 100% hit@5 100%   MRR 0.817
  MRR -0.008, 2 question(s) changed rank
```

**hit@k** is the share of questions with a right answer in the top k — what a
reader experiences, since nobody reads past the first screen. **MRR** is the
mean reciprocal rank: 1.0 if the answer is always first, 0.5 if always second.
It sees a hit moving from rank 5 to rank 2, which hit@k cannot.

No language model is involved and none is needed. This grades **retrieval** —
whether the right passage was found — which is the half that must work before
any generator has a chance. Grading the prose an LLM writes is a different and
far more expensive problem, and not this one.

A dozen questions you already know the answers to is enough to start. The best
ones come from real disappointments: every time a search fails you, add it, and
that failure can never quietly come back. Phrase them as someone would actually
type them — a question worded the way the document is worded passes trivially
and measures nothing.

`--set` changes any search setting for the run, so a knob can be A/B'd without
touching `.env`:

```powershell
python -m app.evaluate --set hybrid=false --save no-keyword.json
python -m app.evaluate --compare no-keyword.json
```

Expectations are substrings matched against a result's path, section or symbol
— not exact citations, because line ranges shift whenever chunking changes, and
a golden set needing a rewrite after every change will not be maintained.

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
faster and more precise for those. The two are complements, not competitors -
`Find-RagLiteral` is the grep half, scoped to the same project subtree, and the
client falls back to it automatically when retrieval comes up empty.

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
| `Python 3.9+ was not found` | `winget install Python.Python.3.12`, or use `-Docker`. The offline bundle needs 3.11 or newer. |
| `startup failed: Could not load model ...` | The model download needs one-time internet access to huggingface.co. Behind a proxy, set `HTTPS_PROXY` before `rag-up`. |
| First run looks stuck | It is installing deps or downloading the model. `.\rag-up.ps1 logs`. |
| `status: empty` | Ingest found nothing. Check `RAG_REPO_MOUNT`, then `.\rag-up.ps1 reindex`. |
| Port 49404 in use | Another RAG is running (`.\rag-up.ps1 down`), or set `RAG_PORT`. |
| Results feel stale | The watcher is off on UNC shares by default. Restarting reconciles; otherwise set `RAG_RESCAN_MINUTES=15`, or run `.\rag-up.ps1 reindex` (which now only re-embeds what changed). |
| Nothing is running after a reboot | Nothing starts it unless you asked: `.\rag-up.ps1 autostart`. |
| An exact ID or part number is not found | The keyword index is built during ingest. On an index predating it, `.\rag-up.ps1 reindex -Full`. Check `/health` shows `"graph": true`. |
| `hops` reaches nothing | The names in the question matched no indexed entity, or they are above the boilerplate ceiling. `GET /entities?include_common=true` shows both, and `RAG_GRAPH_MAX_DF` raises the ceiling — the default is too strict for a single-subject corpus. |
| Offline install fails on the target | Its Python minor version has no wheels in the bundle. It ships cp311–cp314; add others with `.\rag-up.ps1 bundle -ForPython '3.10'` before travelling. |
| `python -m app.ingest` fails with a lock error | The server holds the embedded store. Re-index through it: `.\rag-up.ps1 reindex`. |
| Want better code recall | `RAG_EMBED_MODEL=jinaai/jina-embeddings-v2-base-code` in `.env`, then `.\rag-up.ps1 reindex -Full`. |
| Want a smaller/faster model | `RAG_EMBED_MODEL=BAAI/bge-small-en-v1.5` (~67 MB, 384-dim), then reindex `-Full`. |
| A warning says the index was built with different settings | A setting that shapes the vectors moved. `reindex -Full` rebuilds consistently; the message says which setting and whether it needs a rebuild at all. |
| The Chat tab is missing | Nothing was found to answer with. Install Ollama and pull a model, or set `RAG_CHAT_PROVIDER` — `GET /chat/models` lists what it can see. |
| Chat times out on a local model | A large model can take minutes per answer. The error names `RAG_CHAT_TIMEOUT`; raise it, or pick a smaller model from the dropdown. |
| A question finds nothing, but you know it is in there | Your wording and the page's may not overlap. Search the words printed on the page, or widen `top_k` — and add the question to `rag-eval.json` so the fix is measurable. |

Nothing here is a blocker for work: if the RAG is down, `grep` and direct file
reads always work.
