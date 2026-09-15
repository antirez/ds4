$ErrorActionPreference = 'Stop'
if (-not $env:DS4_HOOK_URL) { throw 'Set DS4_HOOK_URL' }
$payload = [Console]::In.ReadToEnd()
Invoke-RestMethod -Method Post -Uri $env:DS4_HOOK_URL -ContentType 'application/json' -Body $payload | Out-Null
