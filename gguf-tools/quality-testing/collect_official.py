#!/usr/bin/env python3
"""Collect hosted-model continuations for local quant scoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "deepseek-v4-flash"
ENDPOINT = "https://api.deepseek.com/chat/completions"


PROMPTS = [
    "Explain B-tree insertion, including splits and the root special case.",
    "Write a concise design for a TCP echo server that handles slow clients.",
    "Compare mmaped model weights with copying all weights into private buffers on macOS.",
    "Derive why RMSNorm needs a sum of squares and a scaling pass.",
    "Explain why KV cache checkpointing helps agentic sessions.",
    "Give pseudocode for a binary heap push operation.",
    "What are the tradeoffs between top-p sampling and temperature zero decoding?",
    "Summarize how speculative decoding preserves exact greedy output.",
    "Explain why a GPU prefill path can be faster than single-token decode.",
    "Write three invariants for a tokenizer special-token table.",
    "Spiega come funziona l'inserimento in un B-tree, inclusi split e radice.",
    "Scrivi un progetto conciso per un server TCP echo con client lenti.",
    "Confronta mmap dei pesi e copia completa dei pesi su macOS.",
    "Deriva perche RMSNorm richiede somma dei quadrati e riscalamento.",
    "Spiega perche la cache KV su disco aiuta nelle sessioni agentiche.",
    "Scrivi pseudocodice per inserire un elemento in un heap binario.",
    "Quali sono i pro e contro di top-p rispetto a temperatura zero?",
    "Riassumi perche la decodifica speculativa puo mantenere l'output esatto.",
    "Spiega perche il prefill GPU puo essere piu veloce del decode token singolo.",
    "Scrivi tre invarianti per una tabella di token speciali.",
    "Given a sorted array, describe how binary search finds the insertion point.",
    "Write a short C function that clamps an integer to a range.",
    "Explain the difference between a mutex and an atomic counter.",
    "What does backpressure mean in a network server?",
    "Describe a safe format for storing a model checkpoint header.",
    "Explain how a ring buffer wraps and how to avoid overwriting unread data.",
    "Write a brief plan for testing long-context prompt chunking.",
    "What is an importance matrix in low-bit quantization?",
    "Explain why grouped MoE expert execution can improve prefill.",
    "Describe how to compare two model quantizations without relying on one answer.",
    "Data una lista ordinata, descrivi la ricerca binaria del punto di inserimento.",
    "Scrivi una breve funzione C che limita un intero a un intervallo.",
    "Spiega la differenza tra mutex e contatore atomico.",
    "Che cosa significa backpressure in un server di rete?",
    "Descrivi un formato sicuro per salvare l'header di un checkpoint modello.",
    "Spiega come funziona un ring buffer e come evitare sovrascritture.",
    "Scrivi un piano breve per testare il chunking di prompt lunghi.",
    "Che cos'e una matrice di importanza nella quantizzazione low-bit?",
    "Spiega perche raggruppare gli esperti MoE puo accelerare il prefill.",
    "Descrivi come confrontare due quantizzazioni senza fidarsi di una sola risposta.",
    "A user reports generation slows at long context. List the first five measurements to take.",
    "Why can small logit differences change a greedy continuation?",
    "Explain online softmax in attention using simple variables.",
    "Give a minimal JSON schema for a tool call with name and arguments.",
    "How would you test that a file-backed mmap is not repeatedly remapped?",
    "Explain why one large Metal buffer can be worse than multiple overlapping views.",
    "What is the role of a router in a mixture-of-experts layer?",
    "Describe the difference between raw KV rows and compressed KV rows.",
    "Write a checklist for validating that a quantized model still follows instructions.",
    "Explain why evaluating 50 prompts is more informative than one favorite prompt.",
    "Un utente segnala decode lento a contesto lungo. Elenca cinque misure iniziali.",
    "Perche piccole differenze nei logit possono cambiare una continuazione greedy?",
    "Spiega online softmax nell'attenzione con variabili semplici.",
    "Dai uno schema JSON minimo per una tool call con nome e argomenti.",
    "Come testeresti che un mmap su file non venga rimappato ripetutamente?",
    "Spiega perche un grande buffer Metal puo essere peggiore di viste sovrapposte.",
    "Qual e il ruolo del router in un layer mixture-of-experts?",
    "Descrivi la differenza tra righe KV raw e righe KV compresse.",
    "Scrivi una checklist per validare che un modello quantizzato segua ancora le istruzioni.",
    "Perche valutare 50 prompt e piu utile di un singolo prompt preferito?",
    "Write a tiny Python function that returns the median of three numbers.",
    "Explain why sorted arrays make membership tests faster with binary search.",
    "What does eventual consistency mean in a distributed database?",
    "Give a short example of a race condition in C.",
    "Explain the difference between latency and throughput.",
    "How does a trie represent a set of strings?",
    "Describe how to test a command-line parser with edge cases.",
    "Why can mmap page residency differ from virtual address space size?",
    "Explain why quantization error can affect rare experts more than common experts.",
    "Write a short checklist for reviewing a pull request that touches memory lifetimes.",
    "Scrivi una piccola funzione Python che restituisce la mediana di tre numeri.",
    "Spiega perche array ordinati rendono piu veloce la ricerca di appartenenza.",
    "Che cosa significa consistenza eventuale in un database distribuito?",
    "Fai un breve esempio di race condition in C.",
    "Spiega la differenza tra latenza e throughput.",
    "Come rappresenta un trie un insieme di stringhe?",
    "Descrivi come testare un parser da riga di comando con casi limite.",
    "Perche la residenza delle pagine mmap puo differire dalla dimensione virtuale?",
    "Spiega perche l'errore di quantizzazione puo colpire piu gli esperti rari.",
    "Scrivi una breve checklist per revisionare lifetime di memoria in una PR.",
    "Complete this sentence: A good benchmark should measure",
    "Complete this C comment: /* This lock protects",
    "Complete this Italian sentence: Il vantaggio principale della cache e",
    "Translate to Italian: The model should answer only after reading the whole prompt.",
    "Translate to English: La quantizzazione riduce memoria ma puo alterare i logit.",
    "In one paragraph, explain how a compiler uses an abstract syntax tree.",
    "In one paragraph, explain why checksums catch accidental corruption.",
    "Give three examples of useful server metrics.",
    "Why should a tokenizer treat special tags carefully?",
    "Explain how a hash table handles collisions.",
    "Describe the role of calibration data when quantizing a neural network.",
    "What is a confidence interval, in plain language?",
    "Write a simple SQL query that counts rows per category.",
    "Explain why a page cache can make a second file read faster.",
    "Give a short answer: what is the capital of Japan?",
    "Give a short answer: what is the derivative of x squared?",
    "Give a short answer: who wrote The Divine Comedy?",
    "Rispondi brevemente: qual e la capitale del Giappone?",
    "Rispondi brevemente: quanto fa 17 per 23?",
    "Rispondi brevemente: chi ha scritto la Divina Commedia?",
]


def request_one(
    api_key: str,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    top_logprobs: int,
    thinking: str,
    reasoning_effort: str,
    token_limit_field: str,
    provider_order: list[str],
    provider_allow_fallbacks: bool,
    provider_require_parameters: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "stream": False,
    }
    payload[token_limit_field] = max_tokens
    if thinking != "omit":
        payload["thinking"] = {"type": thinking}
    if reasoning_effort != "omit":
        payload["reasoning"] = {"effort": reasoning_effort}
    if provider_order or provider_require_parameters:
        provider = {"require_parameters": provider_require_parameters}
        if provider_order:
            provider["order"] = provider_order
            provider["allow_fallbacks"] = provider_allow_fallbacks
        payload["provider"] = provider
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as fp:
        return json.loads(fp.read().decode("utf-8"))


def fetch_with_retry(
    api_key: str,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    top_logprobs: int,
    thinking: str,
    reasoning_effort: str,
    token_limit_field: str,
    provider_order: list[str],
    provider_allow_fallbacks: bool,
    provider_require_parameters: bool,
) -> dict:
    delay = 1.0
    for attempt in range(6):
        try:
            return request_one(
                api_key,
                endpoint,
                model,
                prompt,
                max_tokens,
                top_logprobs,
                thinking,
                reasoning_effort,
                token_limit_field,
                provider_order,
                provider_allow_fallbacks,
                provider_require_parameters,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code < 500 and e.code != 429:
                raise RuntimeError(f"HTTP {e.code}: {body}") from e
            last = RuntimeError(f"HTTP {e.code}: {body}")
        except Exception as e:  # noqa: BLE001 - command-line retry wrapper.
            last = e
        if attempt == 5:
            raise last
        time.sleep(delay)
        delay *= 1.7
    raise AssertionError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gguf-tools/quality-testing/data")
    ap.add_argument("--prompts", default="gguf-tools/quality-testing/prompts.jsonl")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument("--thinking", choices=("disabled", "enabled", "omit"), default="disabled")
    ap.add_argument("--reasoning-effort",
                    choices=("xhigh", "high", "medium", "low", "minimal", "none", "omit"),
                    default=None)
    ap.add_argument("--token-limit-field",
                    choices=("auto", "max_tokens", "max_completion_tokens"),
                    default="auto")
    ap.add_argument("--provider-order",
                    help="comma-separated OpenRouter provider slugs, for example ambient/fp8")
    ap.add_argument("--allow-provider-fallbacks", action="store_true")
    ap.add_argument("--require-parameters", action="store_true",
                    help="for OpenRouter, route only to endpoints advertising the requested parameters")
    args = ap.parse_args()
    if args.top_logprobs < 0 or args.top_logprobs > 20:
        raise SystemExit("--top-logprobs must be between 0 and 20")

    openrouter = "openrouter.ai" in args.endpoint
    api_key_env = args.api_key_env or ("OPENROUTER_API_KEY" if openrouter else "DEEPSEEK_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"{api_key_env} is not set")
    thinking = args.thinking
    reasoning_effort = args.reasoning_effort
    if reasoning_effort is None:
        reasoning_effort = "none" if openrouter else "omit"
    if openrouter and args.thinking == "disabled":
        thinking = "omit"
    token_limit_field = args.token_limit_field
    if token_limit_field == "auto":
        token_limit_field = "max_tokens"
    provider_order = []
    if args.provider_order:
        provider_order = [item.strip() for item in args.provider_order.split(",") if item.strip()]
    provider_require_parameters = args.require_parameters
    prompts = load_prompts(Path(args.prompts))

    out = Path(args.out)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "continuations").mkdir(parents=True, exist_ok=True)
    (out / "responses").mkdir(parents=True, exist_ok=True)

    manifest = out / "manifest.tsv"
    rows = []
    total = min(args.count, len(prompts))
    print(f"model={args.model} endpoint={args.endpoint}", file=sys.stderr)
    print(f"key_env={api_key_env} token_field={token_limit_field} thinking={thinking} reasoning={reasoning_effort}",
          file=sys.stderr)
    if provider_order or provider_require_parameters:
        print(f"provider_order={','.join(provider_order) or '-'} "
              f"provider_fallbacks={args.allow_provider_fallbacks} "
              f"require_parameters={provider_require_parameters}", file=sys.stderr)
    for i, prompt in enumerate(prompts[: args.count]):
        case_id = f"case_{i:03d}"
        print(f"official {i + 1}/{total}: {case_id}", file=sys.stderr, flush=True)
        response = fetch_with_retry(
            api_key,
            args.endpoint,
            args.model,
            prompt,
            args.max_tokens,
            args.top_logprobs,
            thinking,
            reasoning_effort,
            token_limit_field,
            provider_order,
            args.allow_provider_fallbacks,
            provider_require_parameters,
        )
        choice = response["choices"][0]
        content = choice.get("message", {}).get("content", "")
        if not content:
            print(f"warning: empty continuation for {case_id}", file=sys.stderr)
        logprob_items = (choice.get("logprobs") or {}).get("content", []) or []
        if args.top_logprobs > 0 and not logprob_items:
            raise RuntimeError(f"{case_id}: response did not include output-token logprobs")

        prompt_path = out / "prompts" / f"{case_id}.txt"
        cont_path = out / "continuations" / f"{case_id}.txt"
        resp_path = out / "responses" / f"{case_id}.json"
        prompt_path.write_text(prompt, encoding="utf-8")
        cont_path.write_text(content, encoding="utf-8")
        resp_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append((case_id, prompt_path, cont_path, resp_path))
        time.sleep(0.05)

    with manifest.open("w", encoding="utf-8") as fp:
        fp.write("# id\tprompt_file\tcontinuation_file\tresponse_file\n")
        for row in rows:
            fp.write("\t".join([row[0], str(row[1]), str(row[2]), str(row[3])]) + "\n")
    print(f"wrote {manifest}", file=sys.stderr)
    return 0


def load_prompts(path: Path) -> list[str]:
    if not path.exists():
        return PROMPTS
    prompts = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") if isinstance(obj, dict) else None
            if not prompt:
                raise SystemExit(f"bad prompt row in {path}: {line[:120]}")
            prompts.append(prompt)
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


if __name__ == "__main__":
    raise SystemExit(main())
