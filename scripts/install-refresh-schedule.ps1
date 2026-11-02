param([ValidateSet(1,4)][int]$Hours = 4)
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$python = (Get-Command python).Source
$configPath = Join-Path $project 'config/live_sources.json'
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$config.refresh_hours = $Hours
$config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $configPath -Encoding UTF8
$runner = Join-Path $project 'scripts/run-live-refresh.ps1'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -WindowStyle Hidden -File "' + $runner + '" -Python "' + $python + '"') -WorkingDirectory $project
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval (New-TimeSpan -Hours $Hours)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName 'SocialBenchmark-FreshEvidence' -Action $action -Trigger $trigger -Settings $settings -Description 'Refresh community evidence; requires network and configured source credentials.' -Force
