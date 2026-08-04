# SPDX-License-Identifier: MPL-2.0
#
# KEEP THIS FILE PURE ASCII. Windows PowerShell 5.1 decodes a .psm1 without a BOM
# as Windows-1252, so a UTF-8 em-dash arrives as three characters ending in a
# DOUBLE QUOTE, which closes a string early and swallows whole function bodies.
# Plain ASCII hyphens and quotes only.
<#
.SYNOPSIS
  Client-side conveniences over the local RAG HTTP API.

  The server is unchanged: everything here is query construction, prompt
  assembly and caching around the same /search, /search/full and /context
  endpoints. Nothing in this module runs a language model.

.EXAMPLE
  Import-Module .\rag-client.psm1

  Set-RagProject -Name CHORD -PathPrefix 'Projects/CHORD' -Alias ngpcn
  Ask-Rag 'How is CHORD NGPCN deployed to prod?'      # project inferred from the alias
  Ask-RagDeep 'Explain the NGPCN architecture.' -Project CHORD
  Ask-Rag 'deployment steps' -Project CHORD -Language markdown

  Update-RagContextCache -Project CHORD               # pre-build the background block
  Ask-Rag 'who owns the prod runbook?' -Project CHORD -UseCache
#>

Set-StrictMode -Version 2.0

$script:ModuleRoot   = $PSScriptRoot
$script:ConfigName   = 'rag-projects.json'
$script:DefaultApi   = 'http://127.0.0.1:49404'
$script:Config       = $null
$script:ConfigStamp  = $null
$script:StatsCache   = @{}
$script:HealthCache  = @{}

# One prompt, used for every question. A stable instruction block is most of
# what keeps a generator from filling gaps with invention, and having it live
# in one place means improving it improves every caller at once.
$script:PromptTemplate = @'
Use the numbered context below to answer the question.
Only use facts supported by the context. Cite the sources you used by their
number, like [2]. If the context does not contain the answer, say you do not
know - do not fill the gap from general knowledge.

{context}

Question: {question}
'@

# Extensions the literal-search fallback will look inside. Deliberately not the
# full indexed set: grep cannot read a PDF or an .xlsx, and pretending it can
# by scanning them as bytes produces noise, not evidence.
$script:GrepExtensions = @(
    '.md', '.markdown', '.txt', '.rst', '.json', '.yaml', '.yml', '.toml',
    '.conf', '.ini', '.cfg', '.csv', '.tsv', '.xml', '.html', '.htm',
    '.ps1', '.sh', '.py', '.rs', '.c', '.h', '.sql', '.bat'
)

# Dropped when a question is too long to embed reliably. Short, deliberately
# conservative: only words that carry no retrieval signal at all.
$script:StopWords = @(
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'so', 'as', 'of',
    'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'do', 'does', 'did', 'have', 'has', 'had',
    'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might', 'i',
    'we', 'you', 'they', 'it', 'me', 'my', 'our', 'your', 'their', 'please',
    'kindly', 'just', 'really', 'actually', 'basically', 'thanks', 'thank',
    'hi', 'hello', 'also', 'about', 'that', 'this', 'these', 'those', 'there',
    'here', 'what', 'when', 'where', 'which', 'who', 'whom', 'why', 'how'
)

# ---------------------------------------------------------------- utilities

function ConvertTo-RagHashtable {
    # ConvertFrom-Json returns PSCustomObjects on 5.1 (no -AsHashtable), and
    # nested property access on those is far more awkward than a hashtable.
    param($InputObject)

    if ($null -eq $InputObject) { return $null }
    if ($InputObject -is [System.Collections.IDictionary]) {
        $copy = @{}
        foreach ($key in $InputObject.Keys) {
            $copy[[string]$key] = ConvertTo-RagHashtable $InputObject[$key]
        }
        return $copy
    }
    if ($InputObject -is [System.Management.Automation.PSCustomObject]) {
        $copy = @{}
        foreach ($property in $InputObject.PSObject.Properties) {
            $copy[$property.Name] = ConvertTo-RagHashtable $property.Value
        }
        return $copy
    }
    if ($InputObject -is [object[]]) {
        return @(foreach ($item in $InputObject) { ConvertTo-RagHashtable $item })
    }
    return $InputObject
}

function Merge-RagHashtable {
    # Override wins, key by key, recursing into nested tables so a config file
    # that sets only modes.deep.top_k keeps every other default.
    param([hashtable]$Base, [hashtable]$Override)

    $merged = @{}
    foreach ($key in $Base.Keys) { $merged[$key] = $Base[$key] }
    if (-not $Override) { return $merged }

    foreach ($key in $Override.Keys) {
        if ($merged.ContainsKey($key) -and
            $merged[$key] -is [hashtable] -and
            $Override[$key] -is [hashtable]) {
            $merged[$key] = Merge-RagHashtable -Base $merged[$key] -Override $Override[$key]
        } else {
            $merged[$key] = $Override[$key]
        }
    }
    return $merged
}

function Get-RagValue {
    # Tolerant lookup: config files are hand-edited and a missing key should
    # fall back, not throw under Set-StrictMode.
    param([hashtable]$Table, [string]$Key, $Default = $null)

    if ($Table -and $Table.ContainsKey($Key) -and $null -ne $Table[$Key]) {
        return $Table[$Key]
    }
    return $Default
}

function New-RagDefaultConfig {
    @{
        api   = $script:DefaultApi
        modes = @{
            # Lookup: "I know this exists somewhere, find it." A small, precise
            # window - the answer is a location, not a synthesis.
            quick = @{ top_k = 5;  max_chars = 4000 }
            # Deep dive: "explain this whole thing." Wide enough that a
            # generator can actually summarise across documents.
            deep  = @{ top_k = 25; max_chars = 20000 }
        }
        synonyms = @{}
        projects = @{}
        fallback = @{
            enabled   = $true
            min_score = 0.35
            max_lines = 40
        }
        # Past this, a question is rewritten down to its content words before
        # being embedded. Long prose embeds toward its own filler.
        long_query_chars = 300
        # Ceiling on the share of the character budget that cached background
        # context may take under -UseCache. Without a ceiling the background
        # block fills a Quick-mode budget by itself and the hits that actually
        # answer the question are the ones dropped.
        cache_share = 0.4
        # Append the project name to the query when it is not already in it.
        augment_with_project = $true
    }
}

function Get-RagConfigPath {
    [CmdletBinding()]
    param()
    Join-Path $script:ModuleRoot $script:ConfigName
}

function Get-RagConfig {
    <#
    .SYNOPSIS
      The merged client configuration (defaults + rag-projects.json).
    #>
    [CmdletBinding()]
    param([switch]$Force)

    $path = Get-RagConfigPath
    $stamp = $null
    if (Test-Path -LiteralPath $path) {
        $stamp = (Get-Item -LiteralPath $path).LastWriteTimeUtc
    }
    if (-not $Force -and $script:Config -and $script:ConfigStamp -eq $stamp) {
        return $script:Config
    }

    $config = New-RagDefaultConfig
    if ($stamp) {
        try {
            $raw = Get-Content -LiteralPath $path -Raw -Encoding UTF8
            if ($raw.Trim()) {
                $parsed = ConvertTo-RagHashtable (ConvertFrom-Json $raw)
                $config = Merge-RagHashtable -Base $config -Override $parsed
            }
        } catch {
            Write-Warning "could not read $path : $($_.Exception.Message)"
        }
    }

    $script:Config = $config
    $script:ConfigStamp = $stamp
    return $config
}

function Save-RagConfig {
    param([hashtable]$Config)

    $path = Get-RagConfigPath
    ($Config | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $path -Encoding UTF8
    $script:Config = $null
    $script:ConfigStamp = $null
}

function Resolve-RagApi {
    param([string]$Api)

    if ($Api) { return $Api.TrimEnd('/') }
    if ($env:RAG_API) { return $env:RAG_API.TrimEnd('/') }
    return ([string](Get-RagValue (Get-RagConfig) 'api' $script:DefaultApi)).TrimEnd('/')
}

function Invoke-RagApi {
    # Invoke-RestMethod on 5.1 decodes a charset-less application/json response
    # as Latin-1, which turns every smart quote in a PDF-derived chunk into
    # mojibake. Reading the raw bytes and decoding UTF-8 ourselves avoids it.
    param(
        [string]$Route,
        [string]$Method = 'POST',
        $Body,
        [string]$Api,
        [int]$TimeoutSec = 180
    )

    $base = Resolve-RagApi $Api
    $uri = "$base$Route"
    $arguments = @{
        Uri             = $uri
        Method          = $Method
        TimeoutSec      = $TimeoutSec
        UseBasicParsing = $true
        ErrorAction     = 'Stop'
    }
    if ($null -ne $Body) {
        $arguments.Body = ($Body | ConvertTo-Json -Depth 6)
        $arguments.ContentType = 'application/json'
    }

    try {
        $response = Invoke-WebRequest @arguments
    } catch {
        $detail = $null
        try {
            if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
                $detail = $_.ErrorDetails.Message
            } elseif ($_.Exception.Response) {
                $stream = $_.Exception.Response.GetResponseStream()
                $reader = New-Object System.IO.StreamReader($stream)
                $detail = $reader.ReadToEnd()
                $reader.Close()
            }
        } catch { $detail = $null }

        if ($detail) {
            throw "RAG API $Route failed: $detail"
        }
        throw ("RAG API $uri is not answering ({0}). Start it with .\rag-up.ps1" -f $_.Exception.Message)
    }

    $text = $null
    if ($response.PSObject.Properties.Name -contains 'RawContentStream' -and $response.RawContentStream) {
        $text = [System.Text.Encoding]::UTF8.GetString($response.RawContentStream.ToArray())
    } else {
        $text = $response.Content
    }
    if (-not $text) { return $null }
    return ConvertFrom-Json $text
}

function Get-RagStatsCached {
    param([string]$Api)

    $base = Resolve-RagApi $Api
    if ($script:StatsCache.ContainsKey($base)) { return $script:StatsCache[$base] }
    $stats = Invoke-RagApi -Route '/stats' -Method 'GET' -Api $base -TimeoutSec 15
    $script:StatsCache[$base] = $stats
    return $stats
}

function Get-RagHealthCached {
    param([string]$Api)

    $base = Resolve-RagApi $Api
    if ($script:HealthCache.ContainsKey($base)) { return $script:HealthCache[$base] }
    $health = Invoke-RagApi -Route '/health' -Method 'GET' -Api $base -TimeoutSec 15
    $script:HealthCache[$base] = $health
    return $health
}

# ----------------------------------------------------------------- projects

function Get-RagProject {
    <#
    .SYNOPSIS
      List configured projects, or resolve one by name or alias.
    .DESCRIPTION
      With no -Name, returns every configured project. With -Name, returns the
      one that matches by name or alias (case-insensitive) and errors if there
      is no such project, listing the ones there are.
    #>
    [CmdletBinding()]
    param([Parameter(Position = 0)][string]$Name)

    $config = Get-RagConfig
    $projects = Get-RagValue $config 'projects' @{}

    if (-not $Name) {
        return @(foreach ($key in ($projects.Keys | Sort-Object)) {
            $entry = $projects[$key]
            [pscustomobject]@{
                Name       = $key
                PathPrefix = Get-RagValue $entry 'path_prefix' ''
                Language   = Get-RagValue $entry 'language_filter' $null
                Aliases    = @(Get-RagValue $entry 'aliases' @())
            }
        })
    }

    foreach ($key in $projects.Keys) {
        $entry = $projects[$key]
        $aliases = @(Get-RagValue $entry 'aliases' @())
        if ($key -ieq $Name -or ($aliases | Where-Object { $_ -ieq $Name })) {
            return [pscustomobject]@{
                Name       = $key
                PathPrefix = Get-RagValue $entry 'path_prefix' ''
                Language   = Get-RagValue $entry 'language_filter' $null
                Aliases    = $aliases
                Expand     = Get-RagValue $entry 'expand' @{}
            }
        }
    }

    $known = ($projects.Keys | Sort-Object) -join ', '
    if (-not $known) { $known = '(none configured yet - use Set-RagProject)' }
    throw "unknown project '$Name'. Configured: $known"
}

function Set-RagProject {
    <#
    .SYNOPSIS
      Create or update a project in rag-projects.json.
    .EXAMPLE
      Set-RagProject -Name CHORD -PathPrefix 'Projects/CHORD' -Alias ngpcn,'chord ngpcn'
    #>
    [CmdletBinding(SupportsShouldProcess = $true)]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Name,
        [Parameter(Position = 1)][string]$PathPrefix,
        [string[]]$Alias,
        [string]$Language,
        [hashtable]$Expand
    )

    $config = Get-RagConfig -Force
    $projects = Get-RagValue $config 'projects' @{}
    $entry = @{}
    foreach ($key in $projects.Keys) {
        if ($key -ieq $Name) { $Name = $key; $entry = $projects[$key]; break }
    }

    if ($PSBoundParameters.ContainsKey('PathPrefix')) { $entry['path_prefix'] = $PathPrefix }
    if ($PSBoundParameters.ContainsKey('Alias'))      { $entry['aliases'] = @($Alias) }
    if ($PSBoundParameters.ContainsKey('Language'))   { $entry['language_filter'] = $Language }
    if ($PSBoundParameters.ContainsKey('Expand'))     { $entry['expand'] = $Expand }

    $projects[$Name] = $entry
    $config['projects'] = $projects

    if ($PSCmdlet.ShouldProcess((Get-RagConfigPath), "save project '$Name'")) {
        Save-RagConfig -Config $config
        Get-RagProject -Name $Name
    }
}

function Remove-RagProject {
    <#
    .SYNOPSIS
      Delete a project from rag-projects.json.
    #>
    [CmdletBinding(SupportsShouldProcess = $true)]
    param([Parameter(Mandatory = $true, Position = 0)][string]$Name)

    $config = Get-RagConfig -Force
    $projects = Get-RagValue $config 'projects' @{}
    $match = $projects.Keys | Where-Object { $_ -ieq $Name } | Select-Object -First 1
    if (-not $match) { throw "unknown project '$Name'" }

    if ($PSCmdlet.ShouldProcess((Get-RagConfigPath), "remove project '$match'")) {
        $projects.Remove($match)
        $config['projects'] = $projects
        Save-RagConfig -Config $config
    }
}

function Get-RagPrefix {
    <#
    .SYNOPSIS
      List candidate path_prefix values by walking the indexed corpus.
    .DESCRIPTION
      Reads the corpus root from /stats and enumerates subdirectories, so
      setting up projects does not mean guessing what the layout looks like.
      Paths come back in the forward-slash, corpus-relative form the API
      matches on.
    #>
    [CmdletBinding()]
    param(
        [int]$Depth = 2,
        [string]$Api
    )

    $stats = Get-RagStatsCached -Api $Api
    $root = $stats.repo_path
    if (-not (Test-Path -LiteralPath $root)) {
        throw "corpus root from /stats is not reachable from here: $root"
    }

    $skip = @('.git', '.venv', 'venv', '.data', 'node_modules', '__pycache__', 'archive')
    Get-ChildItem -LiteralPath $root -Directory -Recurse -Depth ($Depth - 1) -ErrorAction SilentlyContinue |
        Where-Object {
            $relative = $_.FullName.Substring($root.Length).TrimStart('\', '/')
            $parts = $relative -split '[\\/]'
            -not ($parts | Where-Object { $skip -contains $_ })
        } |
        ForEach-Object {
            $relative = $_.FullName.Substring($root.Length).TrimStart('\', '/').Replace('\', '/')
            [pscustomobject]@{
                PathPrefix = $relative
                Depth      = ($relative -split '/').Count
                FullPath   = $_.FullName
            }
        } | Sort-Object PathPrefix
}

# ---------------------------------------------------------- query rewriting

function Expand-RagQuery {
    <#
    .SYNOPSIS
      Normalise acronym variants and add project context to a question.
    .DESCRIPTION
      Applies, in order: global synonyms from rag-projects.json, the project's
      own expansions, the project name itself when absent, and - only for
      questions over `long_query_chars` - a reduction to content words, since a
      long piece of prose embeds toward its own filler rather than its subject.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Query,
        [string]$Project,
        [switch]$NoAugment
    )

    $config = Get-RagConfig
    $result = $Query

    $maps = @()
    $maps += ,(Get-RagValue $config 'synonyms' @{})
    $projectEntry = $null
    if ($Project) {
        $projectEntry = Get-RagProject -Name $Project
        if ($projectEntry.PSObject.Properties.Name -contains 'Expand' -and $projectEntry.Expand) {
            $maps += ,(ConvertTo-RagHashtable $projectEntry.Expand)
        }
    }

    foreach ($map in $maps) {
        if (-not $map) { continue }
        foreach ($canonical in $map.Keys) {
            foreach ($variant in @($map[$canonical])) {
                if (-not $variant) { continue }
                # Whitespace in a variant matches any run of whitespace, so
                # "NG PCN" also catches "NG  PCN" and a line-wrapped copy-paste.
                $escaped = [regex]::Escape($variant) -replace '(\\\s)+', '\s+'
                $pattern = "(?i)\b$escaped\b"
                if ($result -match $pattern -and $result -notmatch "(?i)\b$([regex]::Escape($canonical))\b") {
                    $result = [regex]::Replace($result, $pattern, $canonical)
                }
            }
        }
    }

    $augment = [bool](Get-RagValue $config 'augment_with_project' $true)
    if ($projectEntry -and $augment -and -not $NoAugment) {
        if ($result -notmatch "(?i)\b$([regex]::Escape($projectEntry.Name))\b") {
            $result = "$result ($($projectEntry.Name))"
        }
    }

    $limit = [int](Get-RagValue $config 'long_query_chars' 300)
    if ($result.Length -gt $limit) {
        $words = [regex]::Split($result, '\s+') | Where-Object { $_ }
        $kept = @(foreach ($word in $words) {
            $bare = $word.Trim('.,;:!?()[]"''').ToLowerInvariant()
            if ($bare -and ($script:StopWords -notcontains $bare)) { $word }
        })
        $condensed = ($kept -join ' ')
        if ($condensed.Length -gt 40) {
            Write-Verbose "query condensed from $($result.Length) to $($condensed.Length) chars for embedding"
            $result = $condensed
        }
    }

    return $result
}

function Resolve-RagProjectFromQuery {
    param([string]$Query)

    $config = Get-RagConfig
    $projects = Get-RagValue $config 'projects' @{}
    foreach ($key in ($projects.Keys | Sort-Object { -$_.Length })) {
        $entry = $projects[$key]
        $needles = @($key) + @(Get-RagValue $entry 'aliases' @())
        foreach ($needle in $needles) {
            if (-not $needle) { continue }
            $escaped = [regex]::Escape($needle) -replace '(\\\s)+', '\s+'
            if ($Query -match "(?i)\b$escaped\b") { return $key }
        }
    }
    return $null
}

function Resolve-RagQueryPlan {
    # One place where every caller's arguments become the request body: mode
    # defaults, then the project's settings, then explicit parameters last.
    param(
        [string]$Query,
        [string]$Project,
        [string]$Mode = 'Quick',
        [int]$TopK,
        [int]$MaxChars,
        [string]$Language,
        [string]$PathPrefix,
        [switch]$NoRewrite,
        [switch]$NoProject
    )

    $config = Get-RagConfig
    $modes = Get-RagValue $config 'modes' @{}
    $modeKey = $Mode.ToLowerInvariant()
    $modeDefaults = Get-RagValue $modes $modeKey @{}

    $resolvedProject = $null
    if ($Project) {
        $resolvedProject = (Get-RagProject -Name $Project).Name
    } elseif (-not $NoProject -and -not $PathPrefix) {
        # Project-aware by default: a question that names CHORD should not be
        # answered from every folder in the corpus.
        $resolvedProject = Resolve-RagProjectFromQuery -Query $Query
        if ($resolvedProject) {
            Write-Verbose "project inferred from the question: $resolvedProject"
        }
    }

    $entry = $null
    if ($resolvedProject) { $entry = Get-RagProject -Name $resolvedProject }

    $effectivePrefix = $PathPrefix
    if (-not $effectivePrefix -and $entry) { $effectivePrefix = $entry.PathPrefix }
    $effectiveLanguage = $Language
    if (-not $effectiveLanguage -and $entry) { $effectiveLanguage = $entry.Language }

    $effectiveTopK = $TopK
    if (-not $effectiveTopK) { $effectiveTopK = [int](Get-RagValue $modeDefaults 'top_k' 5) }
    $effectiveMaxChars = $MaxChars
    if (-not $effectiveMaxChars) { $effectiveMaxChars = [int](Get-RagValue $modeDefaults 'max_chars' 8000) }

    $effectiveQuery = $Query
    if (-not $NoRewrite) {
        $expandArgs = @{ Query = $Query }
        if ($resolvedProject) { $expandArgs.Project = $resolvedProject }
        $effectiveQuery = Expand-RagQuery @expandArgs
    }

    return [pscustomobject]@{
        Query          = $Query
        EffectiveQuery = $effectiveQuery
        Project        = $resolvedProject
        PathPrefix     = $effectivePrefix
        Language       = $effectiveLanguage
        TopK           = $effectiveTopK
        MaxChars       = $effectiveMaxChars
        Mode           = $Mode
    }
}

# ----------------------------------------------------------------- searching

function Invoke-RagSearch {
    <#
    .SYNOPSIS
      Ranked hits for a question, scoped to a project by default.
    .DESCRIPTION
      A thin wrapper over POST /search (snippets) or POST /search/full (whole
      chunks) that fills in path_prefix, language_filter and top_k from the
      project configuration and the chosen mode.
    .EXAMPLE
      Invoke-RagSearch 'location of the NGPCN runbook' -Project CHORD
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Query,
        [string]$Project,
        [ValidateSet('Quick', 'Deep')][string]$Mode = 'Quick',
        [int]$TopK,
        [string]$Language,
        [string]$PathPrefix,
        [switch]$Full,
        [switch]$NoRewrite,
        [switch]$NoProject,
        [string]$Api
    )

    $plan = Resolve-RagQueryPlan -Query $Query -Project $Project -Mode $Mode -TopK $TopK `
        -Language $Language -PathPrefix $PathPrefix -NoRewrite:$NoRewrite -NoProject:$NoProject

    $body = @{ query = $plan.EffectiveQuery; top_k = $plan.TopK }
    if ($plan.Language)   { $body.language_filter = $plan.Language }
    if ($plan.PathPrefix) { $body.path_prefix = $plan.PathPrefix }

    $route = '/search'
    if ($Full) { $route = '/search/full' }
    $response = Invoke-RagApi -Route $route -Body $body -Api $Api

    foreach ($hit in $response.results) {
        [pscustomobject]@{
            Path      = $hit.path
            StartLine = $hit.start_line
            EndLine   = $hit.end_line
            Score     = $hit.score
            Language  = $hit.language
            Location  = $hit.location
            Text      = $hit.text
        }
    }
}

function ConvertTo-RagLiteralPattern {
    # A question is a bad grep pattern: no file contains the sentence. Short
    # queries stay a literal phrase (that is the ticket-ID and error-code case
    # the fallback exists for); longer ones become an alternation over their
    # most distinctive words, digits first, since those are the identifiers.
    param([string]$Query)

    $tokens = @(
        [regex]::Split($Query, '\s+') |
            ForEach-Object { $_.Trim('.,;:!?()[]"''') } |
            Where-Object { $_ }
    )
    $content = @($tokens | Where-Object { $script:StopWords -notcontains $_.ToLowerInvariant() })
    if ($content.Count -eq 0) { $content = $tokens }
    if ($content.Count -eq 0) { return [regex]::Escape($Query) }

    if ($content.Count -le 3) {
        $phrase = [regex]::Escape($content -join ' ')
        return ($phrase -replace '(\\\s)+', '\s+')
    }

    $ranked = $content |
        Sort-Object -Property @{ Expression = { if ($_ -match '\d') { 0 } else { 1 } } },
                              @{ Expression = { -$_.Length } } |
        Select-Object -First 4
    return (($ranked | ForEach-Object { [regex]::Escape($_) }) -join '|')
}

function Find-RagLiteral {
    <#
    .SYNOPSIS
      Literal text search over the same corpus, scoped the same way.
    .DESCRIPTION
      The complement to retrieval: ticket IDs, error codes and exact strings
      are found better by grep than by an embedding. Uses ripgrep when it is on
      PATH and Select-String otherwise, over the corpus root reported by /stats.

      Only plain-text file types are searched - grep cannot read a PDF or an
      .xlsx, and scanning them as bytes produces noise rather than evidence.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Pattern,
        [string]$Project,
        [string]$PathPrefix,
        [int]$MaxLines,
        [string]$Api
    )

    $config = Get-RagConfig
    if (-not $MaxLines) {
        $MaxLines = [int](Get-RagValue (Get-RagValue $config 'fallback' @{}) 'max_lines' 40)
    }

    $prefix = $PathPrefix
    if (-not $prefix -and $Project) { $prefix = (Get-RagProject -Name $Project).PathPrefix }

    $stats = Get-RagStatsCached -Api $Api
    $root = $stats.repo_path
    if (-not (Test-Path -LiteralPath $root)) {
        Write-Warning "corpus root $root is not reachable from here - skipping literal search"
        return
    }
    $searchRoot = $root
    if ($prefix) {
        $candidate = Join-Path $root ($prefix -replace '/', '\')
        if (Test-Path -LiteralPath $candidate) {
            $searchRoot = $candidate
        } else {
            Write-Warning "path prefix '$prefix' does not exist under $root - searching the whole corpus"
        }
    }

    $rootLength = $root.Length
    $emitted = 0
    $ripgrep = Get-Command rg -ErrorAction SilentlyContinue

    if ($ripgrep) {
        $globs = @()
        foreach ($extension in $script:GrepExtensions) { $globs += @('-g', "*$extension") }
        $arguments = @(
            '--line-number', '--no-heading', '--smart-case', '--color', 'never',
            '--max-count', '3'
        ) + $globs + @('-e', $Pattern, '--', $searchRoot)

        foreach ($line in (& $ripgrep.Path @arguments 2>$null)) {
            if ($emitted -ge $MaxLines) { break }
            if ($line -match '^(.+?):(\d+):(.*)$') {
                $emitted++
                $full = $Matches[1]
                $relative = $full
                if ($full.Length -gt $rootLength -and $full.StartsWith($root)) {
                    $relative = $full.Substring($rootLength).TrimStart('\', '/')
                }
                [pscustomobject]@{
                    Path = $relative.Replace('\', '/')
                    Line = [int]$Matches[2]
                    Text = $Matches[3].Trim()
                }
            }
        }
        return
    }

    Get-ChildItem -LiteralPath $searchRoot -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $script:GrepExtensions -contains $_.Extension.ToLowerInvariant() } |
        ForEach-Object {
            if ($emitted -ge $MaxLines) { return }
            $matched = Select-String -LiteralPath $_.FullName -Pattern $Pattern -ErrorAction SilentlyContinue |
                Select-Object -First 3
            foreach ($item in $matched) {
                if ($emitted -ge $MaxLines) { break }
                $emitted++
                $relative = $item.Path
                if ($relative.Length -gt $rootLength -and $relative.StartsWith($root)) {
                    $relative = $relative.Substring($rootLength).TrimStart('\', '/')
                }
                [pscustomobject]@{
                    Path = $relative.Replace('\', '/')
                    Line = $item.LineNumber
                    Text = $item.Line.Trim()
                }
            }
        }
}

# ------------------------------------------------------------------ context

function Split-RagContextBlock {
    # Turn an assembled /context block back into its parts, so cached and fresh
    # context can be merged, deduplicated and renumbered without the numbers in
    # one block colliding with the numbers in the other.
    param([string]$Context)

    if (-not $Context) { return @() }
    $matches = [regex]::Matches($Context, '(?ms)^\[(\d+)\]\s+(\S+)\r?\n(.*?)(?=\r?\n\r?\n\[\d+\]\s+\S+\r?\n|\z)')
    return @(foreach ($match in $matches) {
        [pscustomobject]@{
            Citation = $match.Groups[2].Value
            Text     = $match.Groups[3].Value.TrimEnd()
        }
    })
}

function Select-RagBlockBudget {
    # Fit as many whole blocks as the budget allows, in order, and report what
    # did not fit. Length is estimated against the rendered form ("[n] cite\n").
    param([object[]]$Blocks, [int]$MaxChars)

    $kept = @()
    $used = 0
    $dropped = 0
    foreach ($block in $Blocks) {
        $length = $block.Citation.Length + $block.Text.Length + 8
        if ($used + $length -gt $MaxChars -and $kept.Count -gt 0) {
            $dropped = $Blocks.Count - $kept.Count
            break
        }
        $kept += $block
        $used += $length
    }
    return [pscustomobject]@{ Blocks = $kept; Chars = $used; Dropped = $dropped }
}

function Join-RagContextBlock {
    # Renumber a merged block list and stop at the character budget. Whole
    # blocks are dropped, never half of one, for the same reason the server
    # does it that way: a chunk cut mid-sentence gets finished from imagination.
    param(
        [object[]]$Blocks,
        [int]$MaxChars,
        [hashtable]$SourceIndex
    )

    $rendered = @()
    $sources = @()
    $used = 0
    $dropped = 0
    $number = 0

    foreach ($block in $Blocks) {
        $number++
        $text = "[$number] $($block.Citation)`n$($block.Text)"
        if ($used + $text.Length -gt $MaxChars -and $rendered.Count -gt 0) {
            $dropped = $Blocks.Count - $rendered.Count
            break
        }
        $rendered += $text
        $used += $text.Length + 2

        $source = [pscustomobject]@{
            n        = $number
            citation = $block.Citation
            # Chunks cite a range, grep lines cite one line; strip either.
            path     = ($block.Citation -replace ':[0-9]+(-[0-9]+)?$', '')
            score    = $null
            origin   = $block.Origin
        }
        if ($SourceIndex -and $SourceIndex.ContainsKey($block.Citation)) {
            $known = $SourceIndex[$block.Citation]
            $source.score = $known.score
        }
        $sources += $source
    }

    return [pscustomobject]@{
        Context = ($rendered -join "`n`n")
        Sources = $sources
        Used    = $rendered.Count
        Dropped = $dropped
        Chars   = $used
    }
}

function Get-RagContext {
    <#
    .SYNOPSIS
      Retrieval assembled for a generator's prompt, with project scoping,
      cached background context and a literal-search fallback.
    .DESCRIPTION
      Calls POST /context with the project's path_prefix and language filter
      and the chosen mode's top_k / max_chars. When retrieval comes back empty
      - or, with reranking off, weak - it also runs a literal search over the
      same subtree and appends those lines as extra numbered evidence.
    .EXAMPLE
      Get-RagContext 'How is CHORD NGPCN deployed to prod?' -Mode Deep
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Query,
        [string]$Project,
        [ValidateSet('Quick', 'Deep')][string]$Mode = 'Quick',
        [int]$TopK,
        [int]$MaxChars,
        [string]$Language,
        [string]$PathPrefix,
        [switch]$UseCache,
        [switch]$NoFallback,
        [switch]$NoRewrite,
        [switch]$NoProject,
        [string]$Api
    )

    $plan = Resolve-RagQueryPlan -Query $Query -Project $Project -Mode $Mode -TopK $TopK `
        -MaxChars $MaxChars -Language $Language -PathPrefix $PathPrefix `
        -NoRewrite:$NoRewrite -NoProject:$NoProject

    $body = @{
        query     = $plan.EffectiveQuery
        top_k     = [Math]::Min($plan.TopK, 50)
        max_chars = $plan.MaxChars
    }
    if ($plan.Language)   { $body.language_filter = $plan.Language }
    if ($plan.PathPrefix) { $body.path_prefix = $plan.PathPrefix }

    $response = Invoke-RagApi -Route '/context' -Body $body -Api $Api

    $sourceIndex = @{}
    foreach ($source in $response.sources) { $sourceIndex[$source.citation] = $source }

    $cacheBlocks = @()
    $cacheUsed = $null
    if ($UseCache) {
        if (-not $plan.Project) {
            Write-Warning '-UseCache needs a project; ignoring it'
        } else {
            $cached = Get-RagContextCache -Project $plan.Project
            if ($cached) {
                $cacheUsed = $cached
                foreach ($block in (Split-RagContextBlock $cached.context)) {
                    $cacheBlocks += [pscustomobject]@{
                        Citation = $block.Citation
                        Text     = $block.Text
                        Origin   = 'cache'
                    }
                }
            }
        }
    }

    $freshBlocks = @(foreach ($block in (Split-RagContextBlock $response.context)) {
        [pscustomobject]@{
            Citation = $block.Citation
            Text     = $block.Text
            Origin   = 'search'
        }
    })

    $fallback = @()
    $config = Get-RagConfig
    $fallbackConfig = Get-RagValue $config 'fallback' @{}
    $fallbackEnabled = [bool](Get-RagValue $fallbackConfig 'enabled' $true) -and -not $NoFallback

    if ($fallbackEnabled) {
        $needFallback = $response.chunks_used -eq 0
        if (-not $needFallback -and $response.sources) {
            # A score threshold is only meaningful for cosine similarity. With
            # the cross-encoder on, scores are unbounded logits and can be
            # negative for a perfectly good hit, so weakness is not judged.
            $health = $null
            try { $health = Get-RagHealthCached -Api $Api } catch { $health = $null }
            if ($health -and -not $health.rerank) {
                $best = ($response.sources | Measure-Object -Property score -Maximum).Maximum
                $floor = [double](Get-RagValue $fallbackConfig 'min_score' 0.35)
                if ($best -lt $floor) {
                    Write-Verbose "best score $best below $floor - adding a literal search"
                    $needFallback = $true
                }
            }
        }
        if ($needFallback) {
            $pattern = ConvertTo-RagLiteralPattern -Query $Query
            Write-Verbose "literal fallback pattern: $pattern"
            $fallbackArgs = @{ Pattern = $pattern; Api = $Api }
            if ($plan.PathPrefix) { $fallbackArgs.PathPrefix = $plan.PathPrefix }
            try {
                $fallback = @(Find-RagLiteral @fallbackArgs)
            } catch {
                Write-Warning "literal search failed: $($_.Exception.Message)"
            }
            foreach ($line in $fallback) {
                $freshBlocks += [pscustomobject]@{
                    Citation = "$($line.Path):$($line.Line)"
                    Text     = $line.Text
                    Origin   = 'grep'
                }
            }
        }
    }

    # Deduplicate, then budget. Fresh hits win over a cached copy of the same
    # chunk, and they get first call on the character budget: the background
    # block is context for the answer, not the answer.
    $seen = @{}
    $fresh = @(foreach ($block in $freshBlocks) {
        if (-not $seen.ContainsKey($block.Citation)) { $seen[$block.Citation] = $true; $block }
    })
    $background = @(foreach ($block in $cacheBlocks) {
        if (-not $seen.ContainsKey($block.Citation)) { $seen[$block.Citation] = $true; $block }
    })

    $freshBudget = $plan.MaxChars
    if ($background.Count -gt 0) {
        $share = [double](Get-RagValue $config 'cache_share' 0.4)
        $freshBudget = [int]($plan.MaxChars * (1.0 - $share))
    }
    $freshFitted = Select-RagBlockBudget -Blocks $fresh -MaxChars $freshBudget
    $backgroundFitted = Select-RagBlockBudget -Blocks $background -MaxChars ($plan.MaxChars - $freshFitted.Chars)

    # Background first, so the model reads the orientation before the specifics.
    $ordered = @($backgroundFitted.Blocks) + @($freshFitted.Blocks)
    $budgetDropped = $freshFitted.Dropped + $backgroundFitted.Dropped

    $assembled = Join-RagContextBlock -Blocks $ordered -MaxChars $plan.MaxChars -SourceIndex $sourceIndex

    return [pscustomobject]@{
        Question       = $Query
        EffectiveQuery = $plan.EffectiveQuery
        Project        = $plan.Project
        PathPrefix     = $plan.PathPrefix
        Language       = $plan.Language
        Mode           = $plan.Mode
        TopK           = $plan.TopK
        MaxChars       = $plan.MaxChars
        Context        = $assembled.Context
        Sources        = $assembled.Sources
        ChunksUsed     = $assembled.Used
        ChunksDropped  = $assembled.Dropped + $budgetDropped
        Chars          = $assembled.Chars
        Fallback       = $fallback
        CachedAt       = $(if ($cacheUsed) { $cacheUsed.created } else { $null })
    }
}

# -------------------------------------------------------------------- cache

function Get-RagCacheDir {
    $stats = $null
    try { $stats = Get-RagStatsCached } catch { $stats = $null }
    $collection = 'default'
    if ($stats -and $stats.collection) { $collection = $stats.collection }
    $dir = Join-Path (Join-Path (Join-Path $script:ModuleRoot '.data') 'context-cache') $collection
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    return $dir
}

function Get-RagCacheFile {
    param([string]$Project)
    $safe = ($Project -replace '[^A-Za-z0-9._-]', '_')
    Join-Path (Get-RagCacheDir) "$safe.json"
}

function Update-RagContextCache {
    <#
    .SYNOPSIS
      Pre-build the background context for a project and store it on disk.
    .DESCRIPTION
      The 20-30 chunks that answer "what is this, who owns it, which
      environments" are the same for most questions about a project. Building
      them once makes repeat questions faster and, more usefully, consistent.
    .EXAMPLE
      Update-RagContextCache -Project CHORD
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Project,
        [string]$Query,
        [int]$TopK = 30,
        [int]$MaxChars = 40000,
        [string]$Language,
        [string]$Api
    )

    $entry = Get-RagProject -Name $Project
    if (-not $Query) {
        $Query = "$($entry.Name) overview: purpose, scope, owners, environments, deployment, key documents and dependencies"
    }

    $context = Get-RagContext -Query $Query -Project $entry.Name -TopK $TopK -MaxChars $MaxChars `
        -Language $Language -Api $Api -NoFallback

    $payload = [pscustomobject]@{
        project = $entry.Name
        query   = $Query
        created = (Get-Date).ToString('o')
        top_k   = $TopK
        context = $context.Context
        sources = $context.Sources
    }
    $file = Get-RagCacheFile -Project $entry.Name
    ($payload | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $file -Encoding UTF8

    Write-Verbose "cached $($context.ChunksUsed) chunks to $file"
    return [pscustomobject]@{
        Project    = $entry.Name
        File       = $file
        ChunksUsed = $context.ChunksUsed
        Chars      = $context.Chars
        Created    = $payload.created
    }
}

function Get-RagContextCache {
    <#
    .SYNOPSIS
      Read a project's cached background context, warning when it is stale.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Project,
        [int]$MaxAgeHours = 168
    )

    $file = Get-RagCacheFile -Project $Project
    if (-not (Test-Path -LiteralPath $file)) {
        Write-Warning "no cached context for $Project - build it with Update-RagContextCache -Project $Project"
        return $null
    }

    $payload = ConvertFrom-Json (Get-Content -LiteralPath $file -Raw -Encoding UTF8)
    $age = (Get-Date) - [datetime]::Parse($payload.created)
    if ($age.TotalHours -gt $MaxAgeHours) {
        Write-Warning ("cached context for {0} is {1:N0} days old - refresh it with Update-RagContextCache" -f $Project, $age.TotalDays)
    }
    return $payload
}

function Clear-RagContextCache {
    <#
    .SYNOPSIS
      Delete cached background context for one project, or all of them.
    #>
    [CmdletBinding(SupportsShouldProcess = $true)]
    param([Parameter(Position = 0)][string]$Project)

    if ($Project) {
        $file = Get-RagCacheFile -Project $Project
        if (Test-Path -LiteralPath $file) {
            if ($PSCmdlet.ShouldProcess($file, 'delete')) { Remove-Item -LiteralPath $file -Force }
        }
        return
    }
    $dir = Get-RagCacheDir
    if ($PSCmdlet.ShouldProcess($dir, 'delete all cached context')) {
        Get-ChildItem -LiteralPath $dir -Filter '*.json' -ErrorAction SilentlyContinue |
            Remove-Item -Force
    }
}

# --------------------------------------------------------------------- ask

function Ask-Rag {
    <#
    .SYNOPSIS
      Retrieve, then assemble the standard grounded-answer prompt.
    .DESCRIPTION
      This is the seam to whatever writes the prose. It runs no language model:
      it returns an object whose .Prompt is ready to paste or pipe into one,
      with the retrieved context already numbered and citable.

      Scoping is project-aware by default - the project comes from -Project, or
      is inferred from names and aliases in the question itself.
    .EXAMPLE
      (Ask-Rag 'How is CHORD NGPCN deployed to prod?').Prompt | Set-Clipboard
    .EXAMPLE
      Ask-Rag 'Explain the NGPCN architecture.' -Project CHORD -Mode Deep -UseCache
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Question,
        [string]$Project,
        [ValidateSet('Quick', 'Deep')][string]$Mode = 'Quick',
        [int]$TopK,
        [int]$MaxChars,
        [string]$Language,
        [string]$PathPrefix,
        [switch]$UseCache,
        [switch]$NoFallback,
        [switch]$NoRewrite,
        [switch]$NoProject,
        [switch]$Raw,
        [string]$OutFile,
        [string]$Api
    )

    $context = Get-RagContext -Query $Question -Project $Project -Mode $Mode -TopK $TopK `
        -MaxChars $MaxChars -Language $Language -PathPrefix $PathPrefix `
        -UseCache:$UseCache -NoFallback:$NoFallback -NoRewrite:$NoRewrite `
        -NoProject:$NoProject -Api $Api

    if ($context.ChunksUsed -eq 0) {
        $scope = 'the whole corpus'
        if ($context.PathPrefix) { $scope = "'$($context.PathPrefix)'" }
        Write-Warning "nothing retrieved from $scope - widen the scope (-NoProject) or check the prefix with Get-RagPrefix"
    }

    $prompt = $script:PromptTemplate.Replace('{context}', $context.Context).Replace('{question}', $Question)

    if ($OutFile) {
        $prompt | Set-Content -LiteralPath $OutFile -Encoding UTF8
        Write-Verbose "prompt written to $OutFile"
    }
    if ($Raw) { return $prompt }

    return [pscustomobject]@{
        Question      = $Question
        Prompt        = $prompt
        Context       = $context.Context
        Sources       = $context.Sources
        Project       = $context.Project
        PathPrefix    = $context.PathPrefix
        Language      = $context.Language
        Mode          = $context.Mode
        ChunksUsed    = $context.ChunksUsed
        ChunksDropped = $context.ChunksDropped
        Chars         = $context.Chars
        Fallback      = $context.Fallback
        CachedAt      = $context.CachedAt
    }
}

function Ask-RagQuick {
    <#
    .SYNOPSIS
      Lookup mode: a small, precise window. "Where is X defined or documented?"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Question,
        [string]$Project,
        [string]$Language,
        [string]$PathPrefix,
        [switch]$Raw,
        [string]$Api
    )
    Ask-Rag -Question $Question -Project $Project -Mode Quick -Language $Language `
        -PathPrefix $PathPrefix -Raw:$Raw -Api $Api
}

function Ask-RagDeep {
    <#
    .SYNOPSIS
      Deep-dive mode: a wide window for overviews and "explain the whole thing".
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$Question,
        [string]$Project,
        [string]$Language,
        [string]$PathPrefix,
        [switch]$UseCache,
        [switch]$Raw,
        [string]$Api
    )
    Ask-Rag -Question $Question -Project $Project -Mode Deep -Language $Language `
        -PathPrefix $PathPrefix -UseCache:$UseCache -Raw:$Raw -Api $Api
}

Export-ModuleMember -Function @(
    'Ask-Rag', 'Ask-RagQuick', 'Ask-RagDeep',
    'Get-RagContext', 'Invoke-RagSearch', 'Find-RagLiteral',
    'Get-RagProject', 'Set-RagProject', 'Remove-RagProject', 'Get-RagPrefix',
    'ConvertTo-RagLiteralPattern',
    'Update-RagContextCache', 'Get-RagContextCache', 'Clear-RagContextCache',
    'Expand-RagQuery', 'Get-RagConfig', 'Get-RagConfigPath'
)
