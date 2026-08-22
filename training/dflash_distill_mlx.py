#!/usr/bin/env python3
"""Distill a classic Qwen3.5 DFlash draft against Ornith with MLX.

The pipeline creates target-generated responses, caches frozen Ornith hidden
features, then fine-tunes the draft with the block objective from the DFlash
paper. Target embeddings and the target LM head remain frozen.
"""

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import mlx.core as mx
from mlx import nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from datasets import load_dataset
from huggingface_hub import snapshot_download
from mlx_lm import generate, load as load_target
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from dflash.model_mlx import _patch_model, load_draft


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--target", default="ornith-ai/Ornith-1.5-9B-MLX-4bit")
    prepare.add_argument("--draft", default="z-lab/Qwen3.5-9B-DFlash")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--samples", type=int, default=128)
    prepare.add_argument("--eval-samples", type=int, default=16)
    prepare.add_argument("--max-new-tokens", type=int, default=192)
    prepare.add_argument("--max-sequence-tokens", type=int, default=384)
    prepare.add_argument("--dataset", default="tatsu-lab/alpaca")
    prepare.add_argument("--seed", type=int, default=20260821)
    recache = sub.add_parser("recache")
    recache.add_argument("--target", default="ornith-ai/Ornith-1.5-9B")
    recache.add_argument("--draft", default="z-lab/Qwen3.5-9B-DFlash")
    recache.add_argument("--data", type=Path, required=True)


    train = sub.add_parser("train")
    train.add_argument("--target", default="ornith-ai/Ornith-1.5-9B-MLX-4bit")
    train.add_argument("--draft", default="z-lab/Qwen3.5-9B-DFlash")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=768)
    train.add_argument("--block-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-ratio", type=float, default=0.04)
    train.add_argument("--clip-grad", type=float, default=1.0)
    train.add_argument("--loss-gamma", type=float, default=4.0)
    train.add_argument("--eval-every", type=int, default=96)
    train.add_argument("--save-every", type=int, default=192)
    train.add_argument(
        "--train-scope", choices=("all", "projection"), default="all"
    )
    train.add_argument("--seed", type=int, default=20260821)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--target", default="ornith-ai/Ornith-1.5-9B-MLX-4bit")
    evaluate.add_argument("--draft", required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--block-size", type=int, default=8)
    evaluate.add_argument("--anchors", type=int, default=64)
    evaluate.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def dataset_prompts(name, limit=None):
    dataset = load_dataset(name, split="train", streaming=True)
    seen = set()
    for row in dataset:
        instruction = str(row.get("instruction") or row.get("prompt") or "").strip()
        extra = str(row.get("input") or "").strip()
        if not instruction or instruction in seen:
            continue
        seen.add(instruction)
        if extra:
            instruction = f"{instruction}\n\nInput:\n{extra}"
        yield instruction
        if limit is not None and len(seen) >= limit:
            return


def token_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def target_features(model, tokens, layer_ids):
    cache = make_prompt_cache(model)
    model(mx.array(tokens, dtype=mx.int32)[None], cache)
    hidden = mx.concatenate(model._hidden_states, axis=-1)[0]
    hidden = mx.stop_gradient(hidden.astype(mx.bfloat16))
    mx.eval(hidden)
    return hidden


def prepare_data(args):
    random.seed(args.seed)
    mx.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output / "samples"
    samples_dir.mkdir(exist_ok=True)

    model, tokenizer = load_target(args.target)
    draft = load_draft(args.draft)
    _patch_model(model, draft.config.target_layer_ids)
    model.eval()
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
    total = args.samples + args.eval_samples
    manifest = []
    started = time.perf_counter()

    # Consume rows until the manifest fills. Short generations are discarded
    # below without taking a sample slot, so capping the stream at `total`
    # would guarantee the shortfall check at the end of this function fires.
    for instruction in dataset_prompts(args.dataset):
        messages = [{"role": "user", "content": instruction}]
        prompt_ids = token_list(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ))
        response = generate(
            model,
            tokenizer,
            prompt=prompt_ids,
            max_tokens=args.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        response_ids = token_list(tokenizer.encode(response, add_special_tokens=False))
        tokens = (prompt_ids + response_ids)[:args.max_sequence_tokens]
        if len(tokens) < len(prompt_ids) + 9:
            continue
        hidden = target_features(model, tokens, draft.config.target_layer_ids)
        sample_name = f"{len(manifest):05d}.safetensors"
        mx.save_safetensors(str(samples_dir / sample_name), {
            "tokens": mx.array(tokens, dtype=mx.int32),
            "hidden": hidden,
            "prompt_length": mx.array([len(prompt_ids)], dtype=mx.int32),
        })
        split = "eval" if len(manifest) < args.eval_samples else "train"
        manifest.append({
            "file": sample_name,
            "split": split,
            "instruction": instruction,
            "response": response,
            "tokens": len(tokens),
            "prompt_tokens": len(prompt_ids),
        })
        elapsed = time.perf_counter() - started
        print(f"[{len(manifest):3d}/{total}] {split:5s} tokens={len(tokens):3d} "
              f"elapsed={elapsed / 60:.1f}m", flush=True)
        mx.clear_cache()
        if len(manifest) >= total:
            break

    if len(manifest) < total:
        raise RuntimeError(f"prepared {len(manifest)} samples, expected {total}")
    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as out:
        for row in manifest:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "target": args.target,
        "draft": args.draft,
        "dataset": args.dataset,
        "seed": args.seed,
        "samples": args.samples,
        "eval_samples": args.eval_samples,
    }
    (args.output / "config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

def recache_features(args):
    model, _ = load_target(args.target)
    draft = load_draft(args.draft)
    _patch_model(model, draft.config.target_layer_ids)
    model.eval()
    rows = read_manifest(args.data, "eval") + read_manifest(args.data, "train")
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        path = args.data / "samples" / row["file"]
        sample = mx.load(str(path))
        tokens = mx.array(sample["tokens"])
        prompt_length = mx.array(sample["prompt_length"])
        mx.eval(tokens, prompt_length)
        hidden = target_features(
            model, tokens.tolist(), draft.config.target_layer_ids
        )
        temporary = path.with_suffix(".tmp.safetensors")
        mx.save_safetensors(str(temporary), {
            "tokens": tokens,
            "hidden": hidden,
            "prompt_length": prompt_length,
        })
        temporary.replace(path)
        print(f"[{index:3d}/{len(rows)}] {row['file']} "
              f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
              flush=True)
        mx.clear_cache()



def read_manifest(data_dir, split):
    rows = []
    with (data_dir / "manifest.jsonl").open(encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            if row["split"] == split:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no {split} samples in {data_dir}")
    return rows


def load_sample(data_dir, row):
    return mx.load(str(data_dir / "samples" / row["file"]))


def choose_anchor(sample, block_size, rng):
    tokens = sample["tokens"]
    prompt_length = int(sample["prompt_length"][0].item())
    last = int(tokens.shape[0]) - block_size
    if last < prompt_length:
        raise RuntimeError("sample response is shorter than the draft block")
    return rng.randint(prompt_length, last)


def bind_and_freeze(draft, target, scope="all"):
    draft.bind(target)
    draft.freeze()
    draft.fc.unfreeze()
    draft.hidden_norm.unfreeze()
    if scope == "all":
        for layer in draft.layers:
            layer.unfreeze()
        draft.norm.unfreeze()
    draft.train()
    target.eval()


def loss_fn(draft, sample, anchor, block_size, loss_gamma):
    tokens = sample["tokens"]
    target_hidden = sample["hidden"][None, :anchor]
    draft_input = mx.concatenate((
        tokens[anchor:anchor + 1],
        mx.full((block_size - 1,), draft.config.mask_token_id, dtype=mx.int32),
    ))[None]
    labels = tokens[anchor + 1:anchor + block_size][None]
    logits = draft(draft_input, target_hidden, draft.make_cache(), logits_start=1)
    losses = nn.losses.cross_entropy(logits, labels, reduction="none")
    positions = mx.arange(block_size - 1, dtype=mx.float32)
    weights = mx.exp(-positions / loss_gamma)
    return mx.sum(losses * weights[None]) / mx.sum(weights)


def acceptance_for_sample(draft, sample, anchor, block_size):
    tokens = sample["tokens"]
    target_hidden = sample["hidden"][None, :anchor]
    draft_input = mx.concatenate((
        tokens[anchor:anchor + 1],
        mx.full((block_size - 1,), draft.config.mask_token_id, dtype=mx.int32),
    ))[None]
    logits = draft(draft_input, target_hidden, draft.make_cache(), logits_start=1)
    predictions = mx.argmax(logits, axis=-1)[0]
    expected = tokens[anchor + 1:anchor + block_size]
    matches = mx.equal(predictions, expected).tolist()
    accepted = 0
    for match in matches:
        if not match:
            break
        accepted += 1
    return accepted


def evaluate_draft(draft, data_dir, rows, block_size, anchors, seed):
    rng = random.Random(seed)
    draft.eval()
    accepted = []
    exact = 0
    for index in range(anchors):
        row = rows[index % len(rows)]
        sample = load_sample(data_dir, row)
        anchor = choose_anchor(sample, block_size, rng)
        count = acceptance_for_sample(draft, sample, anchor, block_size)
        accepted.append(count)
        exact += count == block_size - 1
        if index % 8 == 7:
            mx.clear_cache()
    draft.train()
    mean = sum(accepted) / len(accepted)
    return {
        "anchors": len(accepted),
        "mean_accepted_drafts": mean,
        "mean_cycle_tokens": mean + 1.0,
        "full_blocks": exact,
        "full_block_rate": exact / len(accepted),
    }


def draft_source_path(draft_id):
    return Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))


def save_draft(draft, draft_id, output, metrics, training):
    output.mkdir(parents=True, exist_ok=True)
    source = draft_source_path(draft_id)
    original = {
        key
        for file in source.glob("*.safetensors")
        for key in mx.load(str(file)).keys()
    }
    parameters = dict(tree_flatten(draft.parameters()))
    missing = sorted(original - parameters.keys())
    if missing:
        raise RuntimeError(f"trained model is missing original weights: {missing}")
    weights = {key: parameters[key] for key in sorted(original)}
    mx.eval(weights)
    mx.save_safetensors(str(output / "model.safetensors"), weights)
    shutil.copy2(source / "config.json", output / "config.json")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "training.json").write_text(json.dumps(training, indent=2) + "\n")


def learning_rate(step, total, peak, warmup_ratio):
    warmup = max(1, round(total * warmup_ratio))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_draft(args):
    rng = random.Random(args.seed)
    mx.random.seed(args.seed)
    train_rows = read_manifest(args.data, "train")
    eval_rows = read_manifest(args.data, "eval")
    target, _ = load_target(args.target)
    draft = load_draft(args.draft)
    bind_and_freeze(draft, target, args.train_scope)

    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_and_grad = nn.value_and_grad(draft, loss_fn)
    baseline = evaluate_draft(
        draft, args.data, eval_rows, args.block_size,
        min(64, len(eval_rows) * 4), args.seed,
    )
    print(f"baseline {json.dumps(baseline, sort_keys=True)}", flush=True)
    best = baseline
    metrics = baseline
    started = time.perf_counter()

    for step in range(args.steps):
        row = train_rows[step % len(train_rows)]
        if step and step % len(train_rows) == 0:
            rng.shuffle(train_rows)
        sample = load_sample(args.data, row)
        anchor = choose_anchor(sample, args.block_size, rng)
        rate = learning_rate(step, args.steps, args.learning_rate, args.warmup_ratio)
        optimizer.learning_rate = mx.array(rate)
        loss, grads = loss_and_grad(
            draft, sample, anchor, args.block_size, args.loss_gamma
        )
        grads, grad_norm = optim.clip_grad_norm(grads, args.clip_grad)
        optimizer.update(draft, grads)
        mx.eval(loss, grad_norm, draft.parameters(), optimizer.state)

        if step == 0 or (step + 1) % 8 == 0:
            elapsed = time.perf_counter() - started
            print(f"step={step + 1}/{args.steps} loss={loss.item():.5f} "
                  f"grad={grad_norm.item():.3f} lr={rate:.3e} "
                  f"steps_s={(step + 1) / elapsed:.3f}", flush=True)
        eval_due = (step + 1) % args.eval_every == 0 or step + 1 == args.steps
        save_due = (step + 1) % args.save_every == 0
        if eval_due or save_due:
            metrics = evaluate_draft(
                draft, args.data, eval_rows, args.block_size,
                min(64, len(eval_rows) * 4), args.seed,
            )
            if eval_due:
                print(f"eval step={step + 1} {json.dumps(metrics, sort_keys=True)}", flush=True)
            if metrics["mean_accepted_drafts"] > best["mean_accepted_drafts"]:
                best = metrics
                save_draft(draft, args.draft, args.output / "best", metrics, json_safe_args(args))
        if save_due:
            save_draft(
                draft, args.draft, args.output / f"step-{step + 1:06d}",
                metrics, json_safe_args(args),
            )
        mx.clear_cache()

    final_metrics = evaluate_draft(
        draft, args.data, eval_rows, args.block_size,
        min(64, len(eval_rows) * 4), args.seed,
    )
    save_draft(draft, args.draft, args.output / "final", final_metrics, json_safe_args(args))
    summary = {"baseline": baseline, "best": best, "final": final_metrics}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def evaluate_command(args):
    rows = read_manifest(args.data, "eval")
    target, _ = load_target(args.target)
    draft = load_draft(args.draft)
    bind_and_freeze(draft, target)
    metrics = evaluate_draft(
        draft, args.data, rows, args.block_size, args.anchors, args.seed
    )
    print(json.dumps(metrics, indent=2))


def json_safe_args(args):
    values = vars(args).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def main():
    args = parse_args()
    if args.command == "prepare":
        prepare_data(args)
    elif args.command == "train":
        train_draft(args)
    elif args.command == "recache":
        recache_features(args)
    else:
        evaluate_command(args)


if __name__ == "__main__":
    main()
