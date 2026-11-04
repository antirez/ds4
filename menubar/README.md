# ds4 menu bar status icon

A single-file macOS menu bar companion for `ds4-server`: shows live
inference state without watching logs.

| Menu bar | Meaning |
|---|---|
| `PP 45%` (blue) | prefill in progress (progress / speed / ETA in the dropdown) |
| `GEN 42.1 t/s` (green) | decoding (live speed / tokens generated / elapsed) |
| `WAIT 3` (orange) | requests queued |
| `idle` / `offline` | nothing running / server unreachable |

Data comes from the read-only endpoint `GET /v1/status`, polled once a
second. The sidecar is a plain Python process: it never imports ds4 code,
so engine upgrades cannot break it.

## Requirements

- macOS.
- `python3` with PyObjC:

  ```sh
  pip install pyobjc-framework-Cocoa
  ```

## Automatic start/stop

Start the server with `--menubar`:

```sh
./ds4-server -m ds4flash.gguf --port 8000 --menubar
```

The server spawns `menubar/ds4_menubar.py` (found in the working directory,
same convention as the Metal kernels; use `--chdir` when starting outside the
repo) and terminates it on shutdown, so the icon appears and disappears with
the server. If `python3` or PyObjC is missing the server prints a hint and
runs unchanged. Use `--no-menubar` to turn it off explicitly.

## Manual start

```sh
python3 menubar/ds4_menubar.py --port 8000 &
```

Send a request and watch the icon flip `PP xx%` → `GEN xx.x t/s` → `idle`.
The dropdown lists per-slot progress and session totals (requests, prefill /
decode tokens, average speeds), plus an "Open /v1/status" shortcut.

## Troubleshooting

- **No icon**: PyObjC is missing from the Python that ran the spawn;
  `pip install pyobjc-framework-Cocoa` in that environment.
- **Stuck on `offline`**: the server is not running, or the port is wrong;
  pass `--port` explicitly.
- **Two icons**: `pkill -f ds4_menubar.py`, remove
  `~/.ds4/menubar/ds4_menubar.pid`, then start again.
- **Debug**: `DS4_MENUBAR_DEBUG=1 python3 menubar/ds4_menubar.py` prints
  every poll and title change to stderr.
