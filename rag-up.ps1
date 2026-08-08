# SPDX-License-Identifier: MPL-2.0
#
# KEEP THIS FILE PURE ASCII. Windows PowerShell 5.1 decodes a .ps1 without a BOM
# as Windows-1252, not UTF-8, so a UTF-8 em-dash (E2 80 94) arrives as three
# characters ending in a DOUBLE QUOTE. Inside a double-quoted string that quote
# closes the string early and the trailing one opens a runaway string that eats
# everything up to the next quote - including whole function definitions, which
# then fail at call time with a baffling "term is not recognized" error. That
# cost a debugging round on 2026-08-04; use plain ASCII hyphens and quotes.
<#
.SYNOPSIS
  One-command local RAG (Windows / PowerShell).

  Default is NATIVE mode: one Python process, no Docker, no daemons. The
  embedding model runs in-process (ONNX) and the vector store is file-backed.
  -Docker switches to the containerised stack (Qdrant server + Ollama).

.EXAMPLE
  .\rag-up.ps1 -Folder 'S:\Team\Documents'   # index a folder (remembered in .env)
  .\rag-up.ps1                  # start it, wait until the index is live, open the UI
  .\rag-up.ps1 -Docker          # containerised stack instead
  .\rag-up.ps1 status           # health + index size
  .\rag-up.ps1 reindex          # incremental re-ingest  (-Full to rebuild)
  .\rag-up.ps1 setup            # revisit the optional extras (OCR, chat, reranking)
  .\rag-up.ps1 autostart        # start at logon and stay up  (-Remove to undo)
  .\rag-up.ps1 query "where is the IPC rendezvous done?"
  .\rag-up.ps1 ask "how is CHORD deployed?" -Project CHORD   # a prompt, ready for an LLM
  .\rag-up.ps1 bundle           # vendor wheels + model so it installs with no internet
  .\rag-up.ps1 package          # zip the whole thing up for the trip to another PC
  .\rag-up.ps1 logs             # tail the server log
  .\rag-up.ps1 down             # stop   (-Wipe also drops the index and model cache)
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('up', 'setup', 'status', 'query', 'ask', 'reindex', 'autostart',
                 'bundle', 'package', 'logs', 'down')]
    [string]$Command = 'up',

    [Parameter(Position = 1)]
    [string]$Query,

    # `ask` only: scope the retrieval to a project configured in
    # rag-projects.json. Omitted, the project is inferred from the question.
    [string]$Project,

    # -Folder is the whole setup step: point it at the documents you want
    # searchable and it remembers the choice in .env for next time.
    [string]$Folder,

    # Extra interpreter versions to fetch wheels for, e.g. -ForPython '3.12,3.13'.
    # Binary wheels are built per Python minor version; if the target machine
    # runs a different one than this machine, the offline install fails there.
    [string]$ForPython,

    [switch]$Docker,
    [switch]$Full,
    # `autostart` only: -Remove unregisters it, -System runs at boot with no
    # one logged in (needs admin), -EveryMinutes sets the keepalive interval.
    [switch]$Remove,
    [switch]$System,
    [int]$EveryMinutes = 10,
    # `ask` only: deep-dive mode - a wider retrieval window for overviews.
    [switch]$Deep,
    # `ask` only: reuse the project's cached background context.
    [switch]$UseCache,
    [switch]$Wipe,
    [switch]$NoBrowser,

    # Never open the folder picker; fail instead. For scripted/unattended runs.
    [switch]$NoPrompt
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Corpus handling. Defined here rather than further down because the -Folder
# block below runs at load time, and PowerShell only knows functions already
# executed above the call site.
function Set-Corpus {
    param([string]$Path)

    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue)
    if (-not $resolved) { Write-Error "folder not found: $Path" }

    # ProviderPath, not Path: for a UNC share Resolve-Path returns the
    # provider-qualified form (Microsoft.PowerShell.Core\FileSystem::\\srv\share),
    # which is meaningless to Python. ProviderPath is the plain filesystem path.
    $corpus = $resolved.ProviderPath

    $envPath = Join-Path $PSScriptRoot '.env'
    $keep = @()
    if (Test-Path $envPath) {
        $keep = Get-Content $envPath | Where-Object { $_ -notmatch '^\s*RAG_(REPO_MOUNT|REPO_LABEL)\s*=' }
    }
    # Written verbatim, backslashes and all. The .env reader does no escape
    # processing, and rewriting separators would mangle a UNC path's leading \\.
    $keep += "RAG_REPO_MOUNT=$corpus"
    $keep += "RAG_REPO_LABEL=$(Split-Path $corpus -Leaf)"
    Set-Content -Path $envPath -Value $keep -Encoding ascii

    Write-Host "==> corpus set to $corpus" -ForegroundColor Cyan
    Write-Host '    (remembered in rag\.env - future runs need no -Folder)'
}

function Set-EnvValue {
    # Rewrite one key in rag\.env, leaving every other line alone. An empty
    # value removes the key rather than writing a blank one, so "not set" and
    # "set to nothing" stay distinguishable.
    param([string]$Key, [string]$Value)

    $envPath = Join-Path $PSScriptRoot '.env'
    $keep = @()
    if (Test-Path $envPath) {
        $pattern = '^\s*' + [regex]::Escape($Key) + '\s*='
        $keep = Get-Content $envPath | Where-Object { $_ -notmatch $pattern }
    }
    if ($Value -ne '') { $keep += "$Key=$Value" }
    Set-Content -Path $envPath -Value $keep -Encoding ascii
}

function Read-YesNo {
    param([string]$Prompt, [bool]$Default)

    if ($Default) { $hint = '[Y/n]' } else { $hint = '[y/N]' }
    while ($true) {
        $answer = Read-Host "$Prompt $hint"
        if (-not $answer) { return $Default }
        $answer = $answer.Trim().ToLower()
        if ($answer -eq 'y' -or $answer -eq 'yes') { return $true }
        if ($answer -eq 'n' -or $answer -eq 'no') { return $false }
        Write-Host '    answer y or n' -ForegroundColor Yellow
    }
}

function Test-CorpusHasScans {
    # Does this corpus hold anything OCR would help with? Asking about scanned
    # pages in a folder of Markdown wastes the one question budget this has.
    # Deliberately bounded: `Select-Object -First 1` stops the pipeline on the
    # first hit, so a share with a million files costs one match, not a walk.
    param([string]$Path)

    foreach ($pattern in @('*.pdf', '*.cbz')) {
        try {
            $hit = Get-ChildItem -LiteralPath $Path -Recurse -File -Filter $pattern `
                -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($hit) { return $true }
        } catch { }
    }
    return $false
}

function Find-LocalOllama {
    # The common case for a local model, and the one worth detecting: if a host
    # is already serving models there is no reason to make anyone type a URL.
    foreach ($url in @('http://127.0.0.1:11434')) {
        try {
            $tags = Invoke-RestMethod "$url/api/tags" -TimeoutSec 3
            $names = @($tags.models | ForEach-Object { $_.name })
            if ($names.Count -gt 0) {
                return [pscustomobject]@{ Url = $url; Models = $names }
            }
        } catch { }
    }
    return $null
}

function Invoke-Setup {
    <#
      The optional extras, asked once at install time.

      Everything here defaults to off, and each default is deliberate: they
      cost time, money or a system change that nobody should discover after
      the fact. But off-by-default settings buried in a .env file are, in
      practice, off forever - so the install asks, states the cost in a line,
      and takes Enter for an answer.

      Answers are written to rag\.env either way. Recording a "no" as an
      explicit 0 is the point: it is a decision, not an absence, and it is
      why this does not ask again on every start.
    #>
    $corpus = Get-ConfiguredCorpus

    Write-Host ''
    Write-Host '==> optional extras' -ForegroundColor Cyan
    Write-Host '    All of these are off by default. Press Enter to take the'
    Write-Host '    suggestion in brackets; change your mind later with'
    Write-Host '    .\rag-up.ps1 setup, or by editing rag\.env.'
    Write-Host ''

    # --- reranking --------------------------------------------------------
    Write-Host '  Reranking' -ForegroundColor White
    Write-Host '    Reads each candidate passage against your question and reorders'
    Write-Host '    them. Fixes "topically similar, but not the answer", which better'
    Write-Host '    chunking cannot. Costs a second or two per search, no reindex.'
    if (Read-YesNo '    Turn reranking on?' $true) {
        Set-EnvValue 'RAG_RERANK' '1'
        Write-Host '    on' -ForegroundColor Green
    } else {
        Set-EnvValue 'RAG_RERANK' '0'
        Write-Host '    off' -ForegroundColor DarkGray
    }
    Write-Host ''

    # --- OCR --------------------------------------------------------------
    # Only worth asking when the corpus actually holds pages that might be
    # images. On a folder of Markdown the question is noise.
    if ($corpus -and (Test-CorpusHasScans -Path $corpus)) {
        Write-Host '  OCR for scanned pages' -ForegroundColor White
        Write-Host '    Your folder has PDFs. If any of their pages are scans or'
        Write-Host '    photographs, the words are pixels and index as nothing - a'
        Write-Host '    search then finds nothing whether or not the answer is there.'
        Write-Host '    Reading them is slow: about 3 seconds a page, so a few thousand'
        Write-Host '    scanned pages is an overnight first index. Pages that already'
        Write-Host '    have real text are untouched, and it never redoes a file.'
        if (Read-YesNo '    Read scanned pages?' $false) {
            Set-EnvValue 'RAG_OCR' '1'
            Write-Host '    on - the first index will take a while' -ForegroundColor Green
        } else {
            Set-EnvValue 'RAG_OCR' '0'
            Write-Host '    off - the log will say how many pages were skipped' -ForegroundColor DarkGray
        }
        Write-Host ''
    }

    # --- the chat pane ----------------------------------------------------
    Write-Host '  Chat pane' -ForegroundColor White
    Write-Host '    Search always works on its own. A chat tab additionally writes'
    Write-Host '    cited answers, using a model you supply - this ships with none.'
    $ollama = Find-LocalOllama
    if ($ollama) {
        $shown = ($ollama.Models | Select-Object -First 6) -join ', '
        Write-Host "    Found Ollama on this machine: $shown"
        Write-Host '    Nothing would leave this machine.'
        if (Read-YesNo '    Use it for chat?' $true) {
            $model = Read-Host "    Which model? [$($ollama.Models[0])]"
            if (-not $model) { $model = $ollama.Models[0] }
            Set-EnvValue 'RAG_CHAT_PROVIDER' 'ollama'
            Set-EnvValue 'RAG_CHAT_URL' $ollama.Url
            Set-EnvValue 'RAG_CHAT_MODEL' $model.Trim()
            Write-Host "    on - $model" -ForegroundColor Green
        } else {
            Set-EnvValue 'RAG_CHAT_PROVIDER' ''
            Write-Host '    off' -ForegroundColor DarkGray
        }
    } else {
        Write-Host '    No local model found on this machine.'
        if (Read-YesNo '    Point it at one now?' $false) {
            Write-Host '    1) Claude          2) OpenAI-compatible   3) Ollama elsewhere'
            $pick = Read-Host '    Which? [1/2/3]'
            switch ($pick.Trim()) {
                '1' {
                    Set-EnvValue 'RAG_CHAT_PROVIDER' 'anthropic'
                    $model = Read-Host '    Model? [claude-opus-5]'
                    if (-not $model) { $model = 'claude-opus-5' }
                    Set-EnvValue 'RAG_CHAT_MODEL' $model.Trim()
                    $key = Read-Host '    API key'
                    if ($key) { Set-EnvValue 'RAG_CHAT_API_KEY' $key.Trim() }
                    Write-Host '    on - note your questions and the matching passages' -ForegroundColor Yellow
                    Write-Host '    will be sent to api.anthropic.com' -ForegroundColor Yellow
                }
                '2' {
                    Set-EnvValue 'RAG_CHAT_PROVIDER' 'openai'
                    $url = Read-Host '    Base URL? [https://api.openai.com/v1]'
                    if (-not $url) { $url = 'https://api.openai.com/v1' }
                    Set-EnvValue 'RAG_CHAT_URL' $url.Trim()
                    $model = Read-Host '    Model name'
                    if ($model) { Set-EnvValue 'RAG_CHAT_MODEL' $model.Trim() }
                    $key = Read-Host '    API key (blank if the server needs none)'
                    if ($key) { Set-EnvValue 'RAG_CHAT_API_KEY' $key.Trim() }
                    Write-Host '    on' -ForegroundColor Green
                }
                '3' {
                    Set-EnvValue 'RAG_CHAT_PROVIDER' 'ollama'
                    $url = Read-Host '    Ollama URL? [http://127.0.0.1:11434]'
                    if (-not $url) { $url = 'http://127.0.0.1:11434' }
                    Set-EnvValue 'RAG_CHAT_URL' $url.Trim()
                    $model = Read-Host '    Model name'
                    if ($model) { Set-EnvValue 'RAG_CHAT_MODEL' $model.Trim() }
                    Write-Host '    on' -ForegroundColor Green
                }
                default {
                    Write-Host '    skipped' -ForegroundColor DarkGray
                }
            }
        } else {
            Write-Host '    off - the Chat tab stays hidden' -ForegroundColor DarkGray
        }
    }
    Write-Host ''

    # --- autostart --------------------------------------------------------
    # Asked last because it is the only answer that changes anything outside
    # this folder.
    Write-Host '  Start automatically' -ForegroundColor White
    Write-Host '    Registers a Windows task so the index is running whenever you are'
    Write-Host '    logged in, and comes back if it stops. Without it the index only'
    Write-Host '    keeps up with your documents while this window is open.'
    Write-Host '    Undo any time with .\rag-up.ps1 autostart -Remove.'
    $wantsAutostart = Read-YesNo '    Start at logon?' $false
    Write-Host ''

    Write-Host '==> saved to rag\.env' -ForegroundColor Cyan
    return $wantsAutostart
}

function Get-ConfiguredCorpus {
    if ($env:RAG_REPO_MOUNT) { return $env:RAG_REPO_MOUNT }
    $envPath = Join-Path $PSScriptRoot '.env'
    if (Test-Path $envPath) {
        $match = Select-String -Path $envPath -Pattern '^\s*RAG_REPO_MOUNT\s*=\s*(.+)$' |
            Select-Object -Last 1
        if ($match) { return $match.Matches[0].Groups[1].Value.Trim() }
    }
    return $null
}

function Select-CorpusFolder {
    # A native folder dialog, so double-clicking Rag.bat asks the one question
    # this tool actually needs answered. FolderBrowserDialog requires an STA
    # thread: Windows PowerShell 5.1 is STA, PowerShell 7 is not, so a typed
    # path is the fallback rather than an unexplained failure.
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description = 'Choose the folder of documents to make searchable'
        $dialog.ShowNewFolderButton = $false
        if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            return $dialog.SelectedPath
        }
        return $null
    } catch {
        Write-Host 'Could not open a folder picker on this host.' -ForegroundColor Yellow
        $typed = Read-Host 'Paste the full path to your documents folder'
        if ($typed) { return $typed.Trim('"') }
        return $null
    }
}

if ($Folder) { Set-Corpus -Path $Folder }

# The server reads RAG_PORT from the environment or from .env; this script has
# to resolve it the same way or it polls a port nothing is listening on.
$Port = $env:RAG_PORT
if (-not $Port -and (Test-Path (Join-Path $PSScriptRoot '.env'))) {
    $match = Select-String -Path (Join-Path $PSScriptRoot '.env') `
        -Pattern '^\s*RAG_PORT\s*=\s*(\d+)' | Select-Object -Last 1
    if ($match) { $Port = $match.Matches[0].Groups[1].Value }
}
# Must match the RAG_PORT default in app/config.py.
if (-not $Port) { $Port = '49404' }

$Api = "http://127.0.0.1:$Port"
$Data = Join-Path $PSScriptRoot '.data'
$Venv = Join-Path $PSScriptRoot '.venv'
$VenvPy = Join-Path $Venv 'Scripts\python.exe'
$Wheels = Join-Path $PSScriptRoot 'vendor\wheels'
$PidFile = Join-Path $Data 'rag.pid'
$LogFile = Join-Path $Data 'rag.log'

# ---------------------------------------------------------------- helpers

function Get-Health {
    try { return Invoke-RestMethod -Uri "$Api/health" -TimeoutSec 5 } catch { return $null }
}

# Something else may already own this port - notably the hand-rolled RAG this
# replaces, which also lives on 8404 and also answers /health with
# status=healthy. Without this check the launcher reports another service's
# health as its own success while our server has failed to bind.
function Test-IsOurs {
    param($Health)
    return ($Health -and $Health.service -eq 'rag-local')
}

function Assert-PortFree {
    $health = Get-Health
    if (-not $health) { return }
    if (Test-IsOurs $health) { return }
    Write-Host ''
    Write-Host "Port $Port is already serving something else." -ForegroundColor Red
    Write-Host 'Its /health replies, but it is not this RAG (no service=rag-local).'
    Write-Host 'It is most likely the older hand-rolled RAG stack. Either stop that,'
    Write-Host 'or give this one its own port by putting a line in rag\.env. Pick from'
    Write-Host 'the private range 49152-65535, which nothing standard claims:'
    Write-Host ''
    Write-Host '    RAG_PORT=49404'
    Write-Host ''
    Write-Host 'To see what holds the port:'
    Write-Host "    Get-NetTCPConnection -LocalPort $Port -State Listen |"
    Write-Host '        ForEach-Object { Get-Process -Id $_.OwningProcess }'
    Write-Error "port $Port is in use by another service."
}

function Wait-Healthy {
    # $Proc lets this notice a server that died mid-ingest instead of polling a
    # corpse until the deadline. RAG_START_TIMEOUT_MINUTES covers genuinely huge
    # corpora - embedding is CPU-bound and a large share can take hours.
    param($Proc)

    $minutes = 180
    if ($env:RAG_START_TIMEOUT_MINUTES) { $minutes = [int]$env:RAG_START_TIMEOUT_MINUTES }
    $deadline = (Get-Date).AddMinutes($minutes)
    $last = ''
    $lastChunks = -1
    while ((Get-Date) -lt $deadline) {
        if ($Proc -and $Proc.HasExited) {
            Write-Host ''
            Write-Host 'error: the server process exited while indexing. Last log lines:' -ForegroundColor Red
            Get-Content $LogFile, "$LogFile.err" -Tail 25 -ErrorAction SilentlyContinue
            return $false
        }
        $health = Get-Health
        if ($health -and -not (Test-IsOurs $health)) {
            Write-Host ''
            Write-Host "error: $Api is answering, but it is not this RAG." -ForegroundColor Red
            Write-Host 'Another service holds the port - see .\rag-up.ps1 status, or set'
            Write-Host 'RAG_PORT in rag\.env to move this one out of the way.'
            return $false
        }
        if ($health) {
            if ($health.status -ne $last) {
                Write-Host "    status: $($health.status)"
                $last = $health.status
            }
            # A big corpus spends a long time here; show it moving.
            if ($health.status -eq 'indexing' -and $health.chunks -ne $lastChunks) {
                Write-Host "      indexed so far: $($health.chunks) chunks"
                $lastChunks = $health.chunks
            }
            if ($health.status -eq 'healthy') {
                Write-Host ''
                Write-Host "    RAG is live:  $Api" -ForegroundColor Green
                Write-Host "    indexed:      $($health.chunks) chunks  ($($health.model))"
                Write-Host ''
                Write-Host '    Open it in a browser, or:'
                Write-Host "      .\rag-up.ps1 query 'where is the IPC rendezvous done?'"
                if (-not $NoBrowser) { Start-Process $Api }
                return $true
            }
            # A recorded bootstrap error is terminal - the model download
            # failed, or a dependency is misconfigured. Report it now rather
            # than polling for an hour against something that will never come up.
            if ($health.error) {
                Write-Host ''
                Write-Host "error: startup failed: $($health.error)" -ForegroundColor Red
                return $false
            }
        }
        Start-Sleep -Seconds 5
    }
    return $false
}

function Find-Python {
    # RAG_PYTHON pins a specific interpreter - the escape hatch when the default
    # one on PATH is too new (or too old) for a dependency to have wheels.
    $candidates = @()
    if ($env:RAG_PYTHON) { $candidates += $env:RAG_PYTHON }
    $candidates += @('python', 'python3', 'py')

    foreach ($candidate in $candidates) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        $exe = if ($cmd) { $cmd.Source } elseif (Test-Path $candidate) { $candidate } else { continue }
        try {
            & $exe -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { return $exe }
        } catch { }
    }

    Write-Host 'Python 3.10+ was not found on PATH.' -ForegroundColor Yellow
    Write-Host ''
    Write-Host 'Install it (winget install Python.Python.3.12, or python.org), or point'
    Write-Host 'at an existing install:'
    Write-Host "     `$env:RAG_PYTHON = 'C:\Python312\python.exe'"
    Write-Host 'Or run the containerised stack instead:   .\rag-up.ps1 -Docker'
    Write-Error 'no usable Python found.'
}

function Show-InstallHelp {
    Write-Host ''
    Write-Host 'Dependency install failed. Most often this is a Python version with no' -ForegroundColor Yellow
    Write-Host 'prebuilt wheels yet for one of the packages. Two things to try:' -ForegroundColor Yellow
    Write-Host ''
    Write-Host '  1. Use a different interpreter (3.12 is the safest bet):'
    Write-Host '       Remove-Item .venv -Recurse -Force'
    Write-Host "       `$env:RAG_PYTHON = 'C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe'"
    Write-Host '       .\rag-up.ps1'
    Write-Host ''
    Write-Host '  2. Or run the containerised stack, which brings its own Python:'
    Write-Host '       .\rag-up.ps1 -Docker'
}

function Initialize-Venv {
    # A virtualenv records absolute paths to the Python that built it, so one
    # copied from another machine looks present and then fails cryptically.
    # Probe it and rebuild rather than inflicting that error on the user.
    if (Test-Path $VenvPy) {
        & $VenvPy -c 'import sys' 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host '==> .venv cannot run on this machine (copied from another PC?) - rebuilding' -ForegroundColor Yellow
            Remove-Item $Venv -Recurse -Force
        }
    }

    if (-not (Test-Path $VenvPy)) {
        $py = Find-Python
        Write-Host "==> creating .venv ($(& $py --version))" -ForegroundColor Cyan
        & $py -m venv $Venv
        if ($LASTEXITCODE -ne 0) { Write-Error 'failed to create the virtualenv.' }
    }

    # Reinstall only when the requirements actually change.
    $stamp = Join-Path $Venv '.requirements-sha'
    $text = (Get-Content requirements.txt, requirements-native.txt -Raw) -join ''
    $sha = [System.BitConverter]::ToString(
        [System.Security.Cryptography.SHA256]::Create().ComputeHash(
            [System.Text.Encoding]::UTF8.GetBytes($text))).Replace('-', '')
    if (-not (Test-Path $stamp) -or (Get-Content $stamp -Raw).Trim() -ne $sha) {
        if ((Test-Path $Wheels) -and (Get-ChildItem $Wheels -File -ErrorAction SilentlyContinue)) {
            Write-Host "==> installing dependencies from $Wheels (offline)" -ForegroundColor Cyan
            & $VenvPy -m pip install --quiet --no-index --find-links $Wheels -r requirements-native.txt
        } else {
            Write-Host '==> installing dependencies (first run only)' -ForegroundColor Cyan
            & $VenvPy -m pip install --quiet --upgrade pip
            & $VenvPy -m pip install --quiet -r requirements-native.txt
        }
        if ($LASTEXITCODE -ne 0) { Show-InstallHelp; Write-Error 'dependency install failed.' }
        Set-Content -Path $stamp -Value $sha
    }
}

function Invoke-Pip {
    # pip reports ordinary things on stderr - a yanked version, a resolver note -
    # and this script runs with $ErrorActionPreference = 'Stop', which turns any
    # native stderr output into a terminating NativeCommandError. Without this,
    # one informational line aborts a bundle halfway through.
    param([string[]]$PipArgs)

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $VenvPy -m pip @PipArgs 2>&1
        return [pscustomobject]@{ Ok = ($LASTEXITCODE -eq 0); Output = $output }
    } finally {
        $ErrorActionPreference = $previous
    }
}

# Would an offline install actually succeed on each target interpreter?
#
# Counting wheels is not an answer: the set that broke a work PC had a wheel for
# every package and still could not resolve, because two of them pinned each
# other to versions that were never fetched together. The only honest check is
# to ask pip to resolve the requirements against nothing but vendor\wheels, for
# that Python, which is exactly what the target will do - and to do it here,
# where there is still a network to fix it with.
function Test-BundleResolves {
    param([string]$Versions)

    $targets = @()
    if ($Versions) { $targets += ($Versions -split ',' | Where-Object { $_ } | ForEach-Object { $_.Trim() }) }
    $here = & $VenvPy -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($targets -notcontains $here) { $targets = @($here) + $targets }

    Write-Host ''
    Write-Host '==> checking the bundle installs offline' -ForegroundColor Cyan
    $failed = @()
    foreach ($version in $targets) {
        $probe = Join-Path ([System.IO.Path]::GetTempPath()) ('rag-probe-' + [guid]::NewGuid().ToString('N'))
        $result = Invoke-Pip @(
            'download', '--no-index', '--find-links', $Wheels, '--only-binary=:all:',
            '--python-version', $version, '--platform', 'win_amd64',
            '-r', (Join-Path $PSScriptRoot 'requirements-native.txt'), '-d', $probe
        )
        $ok = $result.Ok
        $output = $result.Output
        Remove-Item $probe -Recurse -Force -ErrorAction SilentlyContinue
        if ($ok) {
            Write-Host "    Python $version : OK" -ForegroundColor Green
        } else {
            $failed += $version
            Write-Host "    Python $version : FAILS" -ForegroundColor Red
            $reason = $output | Where-Object { $_ -match 'Could not find|No matching distribution' } |
                Select-Object -First 2
            foreach ($line in $reason) { Write-Host "      $line" -ForegroundColor Yellow }
        }
    }
    if ($failed) {
        Write-Host ''
        Write-Host "    Add the missing wheels before travelling:" -ForegroundColor Yellow
        Write-Host "        .\rag-up.ps1 bundle -ForPython '$($failed -join ',')'"
    }
}

# Prepare this folder to be carried to a machine with no internet access:
# vendor the wheels and pre-download the model, so rag-up there needs neither
# PyPI nor huggingface.co.
function New-Bundle {
    Initialize-Venv
    New-Item -ItemType Directory -Force -Path $Wheels | Out-Null

    $here = & $VenvPy -c "import sys; print('%d.%d' % sys.version_info[:2])"
    Write-Host "==> building wheels into $Wheels (for Python $here)" -ForegroundColor Cyan
    # `pip wheel`, not `pip download`. One dependency in this set
    # (red-black-tree-mod, via extract-msg) is published only as an sdist, and
    # installing an sdist offline fails: pip's build isolation tries to fetch
    # setuptools from a network the target machine does not have. `pip wheel`
    # builds it here, so the bundle is wheels only and the target never builds.
    & $VenvPy -m pip wheel --quiet -r requirements-native.txt -w $Wheels
    if ($LASTEXITCODE -ne 0) { Write-Error 'wheel build failed.' }

    # Binary wheels (onnxruntime, numpy, pydantic-core, lxml) are built per
    # Python minor version. Wheels for THIS Python are useless on a machine
    # running a different one, and that only shows up at install time on the
    # target - with no network to recover from. Fetch the common versions too.
    # Only the version-specific distributions need re-fetching. A wheel tagged
    # cp314-cp314 runs on 3.14 alone; py3-none-any and cp3XX-abi3 wheels are
    # already portable across versions. Fetching per-distribution also sidesteps
    # a whole-file resolve, which cannot succeed under the --only-binary that
    # --python-version forces: this set contains an sdist-only dependency.
    # Name AND version, not just the name. Fetching by name alone takes whatever
    # is newest, which quietly produces a bundle that cannot install: pydantic
    # pins pydantic-core exactly, so a set holding pydantic-core 2.46.4 for the
    # interpreter that built it and 2.47.0 for every other one resolves on
    # exactly one machine and fails on the rest, at install time, offline,
    # where there is no network to recover with.
    $specific = Get-ChildItem $Wheels -Filter *.whl |
        Where-Object { $_.Name -match '-cp\d+-cp\d+' } |
        ForEach-Object {
            $parts = $_.Name -split '-'
            [pscustomobject]@{ Name = $parts[0]; Version = $parts[1] }
        } | Sort-Object Name, Version -Unique

    foreach ($version in ($ForPython -split ',' | Where-Object { $_ })) {
        $version = $version.Trim()
        Write-Host "==> also fetching binary wheels for Python $version" -ForegroundColor Cyan
        $before = (Get-ChildItem $Wheels -Filter *.whl).Count
        foreach ($dist in $specific) {
            $common = @('download', '--quiet', '--only-binary=:all:', '--no-deps',
                        '--python-version', $version, '--platform', 'win_amd64', '-d', $Wheels)
            $result = Invoke-Pip ($common + @("$($dist.Name)==$($dist.Version)"))
            if (-not $result.Ok) {
                # That exact version may have no wheel for an older interpreter
                # (numpy drops a Python version while the pinned one is newer).
                # Unpinned is right for anything nothing else pins exactly.
                $null = Invoke-Pip ($common + @($dist.Name))
            }
        }
        $added = (Get-ChildItem $Wheels -Filter *.whl).Count - $before
        Write-Host "    added $added wheel(s) for $version"
        if ($added -eq 0) {
            Write-Host '    (none resolved - that Python version may be too old for this set)' -ForegroundColor Yellow
        }
    }

    Test-BundleResolves -Versions $ForPython

    Write-Host '==> pre-downloading the embedding model' -ForegroundColor Cyan
    $env:RAG_MODE = 'native'
    & $VenvPy -c "from app.config import CONFIG; from app.embedder import make_embedder; make_embedder(CONFIG).prepare(); print('   model cached in', CONFIG.model_cache)"
    if ($LASTEXITCODE -ne 0) { Write-Error 'model download failed.' }

    # The reranker is fetched whether or not it is currently enabled: it is
    # small next to the embedding model, and the target machine may have no
    # network at all, so "turn on RAG_RERANK later" must not need one.
    Write-Host '==> pre-downloading the reranker (so it can be enabled offline)' -ForegroundColor Cyan
    & $VenvPy -c "from app.config import CONFIG; from app.reranker import Reranker; Reranker(CONFIG.rerank_model, CONFIG.model_cache).prepare(); print('   reranker cached:', CONFIG.rerank_model)"
    if ($LASTEXITCODE -ne 0) { Write-Host '    (reranker download failed - it can still be fetched later)' -ForegroundColor Yellow }

    Write-Host ''
    Write-Host '    This folder is now self-contained. Copy it to the other machine' -ForegroundColor Green
    Write-Host '    WITHOUT .venv (it is machine-specific and gets rebuilt there):'
    Write-Host ''
    Write-Host '      robocopy . D:\rag /E /XD .venv'
    Write-Host ''
    Write-Host "    Wheels here are for Python $here on this OS/CPU. If the target machine"
    Write-Host '    runs a different Python minor version, add them before you travel:'
    Write-Host "        .\rag-up.ps1 bundle -ForPython '3.12,3.13'"
}

# Zip everything the other machine needs into one file. Excludes .venv (records
# absolute paths to the Python that built it, so it is rebuilt on arrival),
# .data\qdrant (the index of THIS machine's corpus - so the zip does not carry a
# verbatim copy of local documents to somewhere it does not belong),
# .data\context-cache and .data\graph.db, which hold the same document text in
# plain form (FTS5 stores what it indexes), .git - this is a tool being
# delivered, not a checkout being cloned, and the whole history would otherwise
# ride along in every copy - and the pid, log and work-queue files, which
# describe a run on this machine and mean nothing on the next one.
function New-Package {
    $stage = Join-Path ([System.IO.Path]::GetTempPath()) ('rag-package-' + [guid]::NewGuid().ToString('N'))
    $zip = Join-Path (Split-Path $PSScriptRoot -Parent) 'rag-portable.zip'

    # ZipFile throws DirectoryNotFoundException if the destination folder is
    # absent, which is easy to hit on Windows: a redirected Desktop means
    # C:\Users\<you>\Desktop does not exist, it lives under OneDrive.
    $zipDir = Split-Path $zip -Parent
    if (-not (Test-Path $zipDir)) { New-Item -ItemType Directory -Force -Path $zipDir | Out-Null }

    if (-not (Test-Path $Wheels)) {
        Write-Host 'note: vendor\wheels is missing - run .\rag-up.ps1 bundle first if the' -ForegroundColor Yellow
        Write-Host '      target machine has no PyPI access.' -ForegroundColor Yellow
    }

    Write-Host '==> staging files' -ForegroundColor Cyan
    # /XF as well as /XD: the keyword index and entity graph are a single file
    # (plus its write-ahead log), and FTS5 keeps a verbatim copy of every chunk
    # it indexes - packing it would ship the corpus inside the tool. The runtime
    # state goes too: a pid from this machine, and a log naming this machine's
    # paths, are meaningless on the target and are overwritten by its first run
    # anyway. /XF takes a bare name pattern: robocopy rejects a fully qualified
    # path with a wildcard in it outright (invalid parameter), which /XD accepts
    # happily.
    $null = robocopy $PSScriptRoot $stage /E `
        /XD (Join-Path $PSScriptRoot '.venv') (Join-Path $Data 'qdrant') `
            (Join-Path $Data 'context-cache') (Join-Path $PSScriptRoot '.git') `
        /XF 'graph.db*' 'ingest-queue.db*' 'extract-cache.db*' 'rag.pid' `
            '*.log' '*.err' 'rag-eval.json'
    if ($LASTEXITCODE -ge 8) { Write-Error "robocopy failed (exit $LASTEXITCODE)" }
    # robocopy reports what it did in its exit code - 1 means "files copied",
    # which is success. Left alone it becomes the script's own exit code, so a
    # package that worked perfectly ends with Rag.bat printing "[exit code 1]"
    # and any caller checking the status treating it as a failure.
    $global:LASTEXITCODE = 0

    # Clear the packaged .env of everything that is a decision about THIS
    # machine, keeping the tuning that travels (port, threads, embedding model,
    # exclusions).
    #
    # The corpus, because the target points at a path that does not exist there
    # and - worse - the first-run folder picker never appears, since a corpus
    # looks configured. RAG_CHAT_API_KEY, because this zip gets copied between
    # machines and published as a release asset, and a key that has been in one
    # is compromised. And everything the setup questions own - OCR, reranking,
    # the chat provider - because inheriting them is silently wrong: a target
    # installed with -Folder never sees those questions, so it would arrive with
    # OCR on and a chat pane pointed at an Ollama on somebody else's localhost.
    # Off is the documented default; let the target choose for itself.
    $stagedEnv = Join-Path $stage '.env'
    if (Test-Path $stagedEnv) {
        $kept = Get-Content $stagedEnv |
            Where-Object { $_ -notmatch '^\s*RAG_(REPO_MOUNT|REPO_LABEL|CHAT_[A-Z_]+|OCR|RERANK)\s*=' }
        if ($kept) { Set-Content -Path $stagedEnv -Value $kept -Encoding ascii }
        else { Remove-Item $stagedEnv -Force }
        Write-Host '    (cleared the corpus, any API key, and the per-machine choices)'
    }

    Write-Host '==> compressing' -ForegroundColor Cyan
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    # Both assemblies: FileSystem carries ZipFile and ZipFileExtensions, while
    # ZipArchive and ZipArchiveMode live in System.IO.Compression itself.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    Add-Type -AssemblyName System.IO.Compression

    # Entry by entry rather than CreateFromDirectory, for one reason: on Windows
    # .NET writes the platform separator into the entry name, and the zip format
    # requires forward slashes (APPNOTE 4.4.17.1). Explorer and .NET read
    # "vendor\wheels\x.whl" back as a path anyway, so this is invisible here -
    # but Python's zipfile and Linux unzip treat the backslash as an ordinary
    # character, extracting one flat pile of oddly-named files instead of a
    # tree. This repo ships rag-up.sh, so that case is real.
    #
    # ZipFile rather than Compress-Archive either way: far quicker over a few
    # hundred megabytes.
    $archive = [System.IO.Compression.ZipFile]::Open(
        $zip, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        $prefix = $stage.TrimEnd('\').Length + 1
        foreach ($file in Get-ChildItem -LiteralPath $stage -Recurse -File -Force) {
            $entry = $file.FullName.Substring($prefix).Replace('\', '/')
            $null = [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $file.FullName, $entry,
                [System.IO.Compression.CompressionLevel]::Optimal)
        }
    } finally {
        $archive.Dispose()
    }
    Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue

    $mb = (Get-Item $zip).Length / 1MB
    Write-Host ''
    Write-Host ("    {0}  ({1:N0} MB)" -f $zip, $mb) -ForegroundColor Green
    Write-Host ''
    Write-Host '    On the other PC: unzip it, then'
    Write-Host '        .\rag-up.ps1 -Folder "S:\path\to\documents"'
}

$TaskName = 'LocalRAG'

function Register-Autostart {
    <#
      Keeps the server - and therefore the watcher inside it - running without
      anyone remembering to start it.

      The trigger repeats every few minutes on purpose. Starting at logon alone
      would leave a crashed server dead until the next reboot; re-running the
      launcher instead makes the schedule its own keepalive, because a run that
      finds the service healthy prints one line and exits. That is also why the
      "already running" test above keys on /health rather than the pid file.
    #>
    param([switch]$AsSystem, [int]$EveryMinutes = 10)

    # A scheduled run has none of this shell's environment, so a corpus set
    # only in RAG_REPO_MOUNT here is invisible there: the task would fail every
    # few minutes, silently, in a hidden window. .env is the one place both can
    # see, so require it before promising the user this will keep running.
    $corpus = $null
    $envPath = Join-Path $PSScriptRoot '.env'
    if (Test-Path $envPath) {
        $match = Select-String -Path $envPath -Pattern '^\s*RAG_REPO_MOUNT\s*=\s*(.+)$' |
            Select-Object -Last 1
        if ($match) { $corpus = $match.Matches[0].Groups[1].Value.Trim() }
    }
    if (-not $corpus) {
        Write-Host ''
        Write-Host 'No corpus is recorded in rag\.env, so a scheduled run would have' -ForegroundColor Yellow
        Write-Host 'nothing to index - it cannot see this shell environment.' -ForegroundColor Yellow
        Write-Host ''
        Write-Host 'Set it once, then register autostart:'
        Write-Host '    .\rag-up.ps1 -Folder "S:\path\to\documents"'
        Write-Host '    .\rag-up.ps1 autostart'
        Write-Error 'no corpus in .env - autostart not registered.'
        return
    }

    $psExe = (Get-Process -Id $PID).Path
    if (-not $psExe) { $psExe = 'powershell.exe' }
    $action = New-ScheduledTaskAction -Execute $psExe `
        -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass ' +
                   '-File "' + (Join-Path $PSScriptRoot 'rag-up.ps1') + '" -NoBrowser -NoPrompt') `
        -WorkingDirectory $PSScriptRoot

    if ($AsSystem) {
        # True 24/7: runs at boot with no one logged in. Needs admin to
        # register, and SYSTEM has no mapped drives and no share credentials -
        # for a UNC corpus this will index nothing, which is why it is not the
        # default.
        $identity = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
        $trigger = New-ScheduledTaskTrigger -AtStartup
    } else {
        $identity = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    }

    # Repetition is not exposed on AtLogOn/AtStartup triggers, so it is lifted
    # off a throwaway -Once trigger. "Forever" is an ABSENT duration: the
    # obvious [TimeSpan]::MaxValue serialises to P99999999DT23H59M59S, which
    # Task Scheduler rejects outright as out of range.
    $repeat = New-ScheduledTaskTrigger -Once -At (Get-Date) `
        -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes)
    $trigger.Repetition = $repeat.Repetition
    $trigger.Repetition.Duration = $null
    $trigger.Repetition.StopAtDurationEnd = $false

    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
    $settings.DisallowStartOnRemoteAppSession = $false
    $settings.Hidden = $true

    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Principal $identity -Settings $settings `
        -Description 'Local RAG: keeps the index server and its file watcher running.' | Out-Null

    Write-Host "==> autostart registered as scheduled task '$TaskName'" -ForegroundColor Cyan
    if ($AsSystem) {
        Write-Host '    runs at boot as SYSTEM - no login needed.'
        Write-Host '    NOTE: SYSTEM cannot see mapped drives or reach an SMB share as you.'
        Write-Host '          If your documents live on a network share, re-register without'
        Write-Host '          -System so it runs as you instead.' -ForegroundColor Yellow
    } else {
        Write-Host "    runs at your logon, as $env:USERNAME."
    }
    Write-Host "    re-checks every $EveryMinutes minute(s) and restarts the server if it died."
    Write-Host '    remove it with:  .\rag-up.ps1 autostart -Remove'
    Write-Host ''
    Write-Host '    starting it now...'
    Start-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}

function Unregister-Autostart {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "no scheduled task named '$TaskName' - nothing to remove."
        return
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "==> autostart removed ('$TaskName')" -ForegroundColor Cyan
    Write-Host '    the server keeps running until you stop it: .\rag-up.ps1 down'
}

function Get-RagProcess {
    if (-not (Test-Path $PidFile)) { return $null }
    $procId = (Get-Content $PidFile -Raw).Trim()
    return Get-Process -Id $procId -ErrorAction SilentlyContinue
}

function Invoke-Compose {
    param([string[]]$ComposeArgs)
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "docker is not on PATH. Native mode needs no Docker - just run: .\rag-up.ps1"
    }
    & docker compose @ComposeArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "docker compose $($ComposeArgs -join ' ') failed." }
}

# ------------------------------------------------------------------ modes

function Start-Native {
    New-Item -ItemType Directory -Force -Path $Data | Out-Null

    # Keyed on the health check rather than the pid file, because the keepalive
    # task re-runs this every few minutes and the pid file is the fragile half:
    # lose it (a wiped .data, a server started by hand) and the old test would
    # launch a second server, which then dies on the port or the storage lock
    # and leaves a pid file pointing at a corpse. Answering /health as
    # service=rag-local is proof enough that the job is already done.
    $existing = Get-RagProcess
    if (Test-IsOurs (Get-Health)) {
        $who = if ($existing) { "pid $($existing.Id)" } else { 'started elsewhere' }
        Write-Host "already running ($who) - $Api" -ForegroundColor Green
        if (-not $NoBrowser) { Start-Process $Api }
        return
    }

    Assert-PortFree
    Initialize-Venv

    Write-Host '==> starting the RAG server' -ForegroundColor Cyan
    Write-Host '    (first run downloads the embedding model, then indexes the repo;'
    Write-Host '     later starts reuse both and take seconds)'

    $env:RAG_MODE = 'native'
    $proc = Start-Process -FilePath $VenvPy -ArgumentList '-m', 'app.server' `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err"
    Set-Content -Path $PidFile -Value $proc.Id

    # Surface a startup crash instead of polling for an hour against a dead process.
    # Wait for OUR server specifically, not merely for something to answer.
    $waited = 0
    while (-not (Test-IsOurs (Get-Health))) {
        if ($proc.HasExited) {
            Write-Host 'error: the server exited during startup. Last lines:' -ForegroundColor Red
            Get-Content $LogFile, "$LogFile.err" -Tail 20 -ErrorAction SilentlyContinue
            Remove-Item $PidFile -ErrorAction SilentlyContinue
            return
        }
        Start-Sleep -Seconds 2
        $waited += 2
        if ($waited -gt 120) { Write-Error 'the server never answered /health.' }
    }

    if (Wait-Healthy -Proc $proc) {
        Write-Host "    log:  $LogFile      stop with: .\rag-up.ps1 down"
    } else {
        Write-Host ''
        Write-Host 'The server may still be running and indexing - this only means the' -ForegroundColor Yellow
        Write-Host 'wait gave up. Check with:  .\rag-up.ps1 status' -ForegroundColor Yellow
        Write-Host "Full log: $LogFile"
    }
}

function Start-DockerStack {
    Write-Host '==> building and starting the containerised stack' -ForegroundColor Cyan
    Invoke-Compose @('up', '-d', '--build')
    Write-Host '==> waiting for the index' -ForegroundColor Cyan
    if (-not (Wait-Healthy)) { Write-Error 'not healthy. Check: .\rag-up.ps1 -Docker logs' }
}

function Stop-Native {
    $proc = Get-RagProcess
    if ($proc) {
        Stop-Process -Id $proc.Id -Force
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        Write-Host 'stopped.'
    } else {
        Write-Host 'not running.'
    }
    if ($Wipe) {
        Remove-Item (Join-Path $Data 'qdrant'), (Join-Path $Data 'models') `
            -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host 'index and model cache removed.'
    }
}

# ------------------------------------------------------------------- main

switch ($Command) {
    'up' {
        # First run with no corpus: ask, rather than silently indexing whatever
        # directory this folder happens to sit in.
        $firstRun = -not (Get-ConfiguredCorpus)
        if ($firstRun) {
            if ($NoPrompt -or -not [Environment]::UserInteractive) {
                Write-Error 'no corpus configured. Pass -Folder "C:\path\to\documents".'
            }
            Write-Host ''
            Write-Host 'Which folder of documents should be searchable?' -ForegroundColor Cyan
            Write-Host '(a folder picker is opening - it is only ever read, never written to)'
            $picked = Select-CorpusFolder
            if (-not $picked) { Write-Error 'no folder chosen - nothing to index.' }
            Set-Corpus -Path $picked

            # The optional extras, while the corpus is known and before the
            # first index starts - OCR in particular has to be decided now,
            # because turning it on afterwards means indexing the scans again.
            $wantsAutostart = Invoke-Setup
            if ($wantsAutostart) { Register-Autostart }
        }
        if ($Docker) { Start-DockerStack } else { Start-Native }
    }

    'setup' {
        # The same questions the first run asks, for changing your mind. Note
        # that turning OCR on here does not reach back into what is already
        # indexed: `reindex` picks up the scans that were skipped.
        if ($NoPrompt -or -not [Environment]::UserInteractive) {
            Write-Error 'setup needs an interactive console. Edit rag\.env instead.'
        }
        if (-not (Get-ConfiguredCorpus)) {
            Write-Host 'No corpus set yet - run .\rag-up.ps1 first, or pass -Folder.' -ForegroundColor Yellow
        }
        $wantsAutostart = Invoke-Setup
        if ($wantsAutostart) { Register-Autostart }
        Write-Host ''
        Write-Host 'Restart to apply: .\rag-up.ps1 down, then .\rag-up.ps1'
        Write-Host 'If you turned OCR on, follow that with .\rag-up.ps1 reindex'
        Write-Host 'to read the scanned pages that were skipped before.'
    }

    'status' {
        try { Invoke-RestMethod -Uri "$Api/stats" -TimeoutSec 5 | Format-List }
        catch { Write-Host "not responding on $Api (start it: .\rag-up.ps1)" -ForegroundColor Yellow }
    }

    'query' {
        if (-not $Query) { Write-Error "usage: .\rag-up.ps1 query 'your question'" }
        $body = @{ query = $Query; top_k = 5 } | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "$Api/search" -Method POST -Body $body -ContentType 'application/json'
        foreach ($hit in $response.results) {
            Write-Host ''
            Write-Host "--- $($hit.path):$($hit.start_line)-$($hit.end_line)  (score $($hit.score))" -ForegroundColor Cyan
            Write-Host $hit.text
        }
    }

    'ask' {
        # Retrieval assembled into a grounded-answer prompt. Still no language
        # model here: what this prints is the input to one.
        if (-not $Query) { Write-Error "usage: .\rag-up.ps1 ask 'your question' [-Project NAME] [-Deep]" }
        Import-Module (Join-Path $PSScriptRoot 'rag-client.psm1') -Force -DisableNameChecking

        $askArgs = @{ Question = $Query; Api = $Api }
        if ($Project)  { $askArgs.Project = $Project }
        if ($Deep)     { $askArgs.Mode = 'Deep' }
        if ($UseCache) { $askArgs.UseCache = $true }
        $answer = Ask-Rag @askArgs

        $scope = 'whole corpus'
        if ($answer.PathPrefix) { $scope = $answer.PathPrefix }
        Write-Host ("==> {0} chunks, {1} chars, scope: {2}" -f $answer.ChunksUsed, $answer.Chars, $scope) -ForegroundColor Cyan
        foreach ($source in $answer.Sources) {
            Write-Host ("    [{0}] {1}" -f $source.n, $source.citation) -ForegroundColor DarkGray
        }
        Write-Host ''
        # Written to the pipeline, not the host, so it can be redirected into a
        # file or piped straight to whatever runs the model.
        $answer.Prompt
    }

    'autostart' {
        if ($Remove) { Unregister-Autostart }
        else { Register-Autostart -AsSystem:$System -EveryMinutes $EveryMinutes }
    }

    'bundle' { New-Bundle }

    'package' { New-Package }

    'reindex' {
        $body = @{ full = [bool]$Full } | ConvertTo-Json
        Invoke-RestMethod -Uri "$Api/reindex" -Method POST -Body $body -ContentType 'application/json'
    }

    'logs' {
        if ($Docker) { Invoke-Compose @('logs', '-f', 'rag-api') }
        else { Get-Content $LogFile -Wait -Tail 40 }
    }

    'down' {
        if ($Docker) {
            if ($Wipe) { Invoke-Compose @('down', '-v') } else { Invoke-Compose @('down') }
        } else { Stop-Native }
    }
}
