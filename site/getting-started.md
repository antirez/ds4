# Getting Started

Choose your platform and build:

| Platform | Build command |
| --- | --- |
| [Apple Silicon (Metal)](/metal) | `make` |
| [DGX Spark](/dgx-spark) | `make cuda-spark` |
| [Strix Halo / ROCm](/strix-halo) | `make strix-halo` |
| [CUDA Multi-GPU](/cuda-multi-gpu) | `make cuda-generic` |

## Quick Start

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
make
./download_model.sh ds4f-q2
./ds4
```

Downloads go in `gguf/`. Repeat to resume interrupted downloads.

## First Run

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`. Pass `-m FILE` to choose explicitly.

## Interactive CLI

| Command | Action |
| --- | --- |
| `/help` | Show available commands |
| `/read FILE` | Read a file into context |
| `/ctx N` | Change context size |
| `/quit` | Exit |
| `Ctrl+C` | Interrupt generation |

## Native Coding Agent

`ds4-agent` runs inference directly. Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |

## Thinking Controls

Thinking is enabled by default:

- `--nothink` or `/nothink` — direct answers
- `--think` or `/think` — enable thinking
- `--think-level 25` or `/think 25` — set reasoning effort (1-100)
- `--think-max` — maximum reasoning effort

Default sampling: temperature 1, top-p 1, min-p 0.05. Use `--temp 0` for greedy output.

## Server

```sh
./ds4-server --ctx 32768
```

The server listens at `http://127.0.0.1:8000` by default. See the [Serving guide](/server).