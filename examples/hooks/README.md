# Hook command examples

All commands read exactly one JSON event from standard input.

```sh
export DS4_HOOK_LOG="$PWD/hooks.jsonl"
./ds4-agent --after-response-hook './examples/hooks/log-json.sh'
```

For an HTTP callback:

```sh
export DS4_HOOK_URL='https://example.test/events'
./ds4-agent --after-response-hook './examples/hooks/notify-curl.sh'
```

PowerShell can be invoked from a POSIX host when `pwsh` is available:

```sh
./ds4-agent --after-response-hook 'pwsh -File ./examples/hooks/notify-powershell.ps1'
```
