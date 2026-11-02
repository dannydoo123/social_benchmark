param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $project
$env:PYTHONPATH = Join-Path $project 'src'
& $Python scripts/refresh_live_evidence.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts/collect_community_feeds.py
& $Python scripts/collect_reddit_archive.py --max-requests 30
& $Python scripts/collect_hn_comments.py
& $Python scripts/collect_forum_evidence.py
& $Python scripts/build_community_perspective.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
