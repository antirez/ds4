"""Batched local BF16 reference-fixture generator for Qwen3.8-Flash-Next.

Same output schema as /tmp/bf16_gen_fixture.py, but all prompts are
left-padded into one batch so the per-token MoE expert loop runs once per
step for the whole batch (the dominant CPU cost). Fidelity is verified
against the single-sequence cases already generated case-by-case.
"""
import argparse
import json
import os
import time

import torch

SNAP = ('/Users/ifioravanti/.cache/huggingface/hub/models--Qwen--'
        'Qwen3.8-Flash-Next/snapshots/'
        'de4b8e4d43b917e7706784d8bb445c9af86a3540')
PROMPTS = 'gguf-tools/quality-testing/prompts.jsonl'


def ds4_chat_ids(tok, prompt_text):
    im_start = tok.convert_tokens_to_ids('<|im_start|>')
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    think_end = tok.convert_tokens_to_ids('</think>')
    assert im_start >= 0 and im_end >= 0 and think_end >= 0
    ids = []
    ids.append(im_start)
    ids += tok('user\n', add_special_tokens=False)['input_ids']
    ids += tok(prompt_text, add_special_tokens=False)['input_ids']
    ids.append(im_end)
    ids += tok('\n', add_special_tokens=False)['input_ids']
    ids.append(im_start)
    ids += tok('assistant\n', add_special_tokens=False)['input_ids']
    ids.append(think_end)
    ids += tok('\n\n', add_special_tokens=False)['input_ids']
    return ids


def token_bytes(tok, tid):
    text = tok.decode([tid], skip_special_tokens=False)
    return list(text.encode('utf-8', errors='replace'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/qwen38-bf16-local-100')
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--max-tokens', type=int, default=24)
    ap.add_argument('--top-logprobs', type=int, default=20)
    ap.add_argument('--verify-only', action='store_true',
                    help='verify batched vs existing single-sequence cases '
                         'and exit without writing')
    args = ap.parse_args()

    torch.set_num_threads(max(os.cpu_count() or 8, 8))
    from transformers import AutoTokenizer, Qwen4ExpForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(SNAP)

    t0 = time.time()
    model = Qwen4ExpForConditionalGeneration.from_pretrained(
        SNAP, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        low_cpu_mem_usage=True, device_map='cpu')
    model.eval()
    print(f'model loaded on cpu in {time.time()-t0:.1f}s', flush=True)

    prompts = []
    with open(PROMPTS) as f:
        for line in f:
            prompts.append(json.loads(line)['prompt'])
    prompts = prompts[:args.count]

    all_ids = [ds4_chat_ids(tok, p) for p in prompts]
    pad_id = tok.convert_tokens_to_ids('<|endoftext|>')
    assert pad_id >= 0
    max_len = max(len(x) for x in all_ids)
    batch = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(all_ids), max_len), dtype=torch.long)
    for i, ids in enumerate(all_ids):
        batch[i, max_len - len(ids):] = torch.tensor(ids)
        attn[i, max_len - len(ids):] = 1

    generated = [[] for _ in all_ids]
    positions = [[] for _ in all_ids]
    t0 = time.time()
    with torch.inference_mode():
        past = None
        cur = batch
        cur_attn = attn
        for step in range(args.max_tokens):
            out = model(cur, attention_mask=cur_attn,
                        past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            top = torch.topk(torch.log_softmax(logits.float(), dim=-1),
                             args.top_logprobs, dim=-1)
            nxt = top.indices[:, 0]
            for i in range(len(all_ids)):
                generated[i].append(int(nxt[i]))
                positions[i].append({
                    'token': tok.decode([int(nxt[i])]),
                    'bytes': token_bytes(tok, int(nxt[i])),
                    'logprob': float(top.values[i, 0]),
                    'top_logprobs': [
                        {
                            'token': tok.decode([int(t_)]),
                            'bytes': token_bytes(tok, int(t_)),
                            'logprob': float(v_),
                        }
                        for t_, v_ in zip(top.indices[i], top.values[i])
                    ],
                })
            past = out.past_key_values
            cur = nxt.unsqueeze(1)
            cur_attn = torch.cat(
                [cur_attn, torch.ones((len(all_ids), 1), dtype=torch.long)],
                dim=1)
            print(f'step {step+1}/{args.max_tokens} '
                  f'({time.time()-t0:.1f}s)', flush=True)

    if args.verify_only:
        match = 0
        total = 0
        for i, gen in enumerate(generated):
            cid = f'case_{i:03d}'
            path = os.path.join(args.out, 'continuations', f'{cid}.txt')
            if not os.path.exists(path):
                continue
            single = tok(path.read_text(),
                         add_special_tokens=False)['input_ids']
            total += 1
            same = single == gen
            match += same
            if not same:
                first = next((j for j, (a, b) in
                              enumerate(zip(single, gen)) if a != b),
                             min(len(single), len(gen)))
                print(f'{cid}: MISMATCH at token {first}: '
                      f'single={tok.decode(single[max(0,first-2):first+3])!r} '
                      f'batch={tok.decode(gen[max(0,first-2):first+3])!r}',
                      flush=True)
        print(f'batched-vs-single verification: {match}/{total} identical')
        return

    for sub in ('prompts', 'continuations', 'responses'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    rows = []
    roundtrip_ok = 0
    for i, gen in enumerate(generated):
        cid = f'case_{i:03d}'
        text = tok.decode(gen, skip_special_tokens=False)
        re_ids = tok(text, add_special_tokens=False)['input_ids']
        roundtrip_ok += re_ids == gen
        response = {
            'id': f'bf16-{cid}',
            'object': 'chat.completion',
            'model': 'local-bf16-qwen3.8-flash-next',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': text},
                'logprobs': {'content': positions[i]},
                'finish_reason': 'length',
            }],
            'usage': {'completion_tokens': len(gen)},
        }
        (open(os.path.join(args.out, 'prompts', f'{cid}.txt'), 'w')
         .write(prompts[i]))
        (open(os.path.join(args.out, 'continuations', f'{cid}.txt'), 'w')
         .write(text))
        with open(os.path.join(args.out, 'responses', f'{cid}.json'),
                  'w') as f:
            json.dump(response, f, ensure_ascii=False, indent=1)
        rows.append(cid)
    with open(os.path.join(args.out, 'manifest.tsv'), 'w') as f:
        f.write('# id\tprompt_file\tcontinuation_file\tresponse_file\n')
        for cid in rows:
            f.write('\t'.join([
                cid,
                os.path.join(args.out, 'prompts', f'{cid}.txt'),
                os.path.join(args.out, 'continuations', f'{cid}.txt'),
                os.path.join(args.out, 'responses', f'{cid}.json'),
            ]) + '\n')
    self_nll = 0.0
    n_pos = 0
    for i in range(len(rows)):
        for pos in positions[i]:
            self_nll += -pos['logprob']
            n_pos += 1
    print(f'wrote {len(rows)} cases to {args.out}; round-trip clean '
          f'{roundtrip_ok}/{len(rows)}; BF16 self-NLL '
          f'{self_nll/n_pos:.6f} over {n_pos} tokens', flush=True)


if __name__ == '__main__':
    main()
