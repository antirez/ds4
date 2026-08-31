"""Local BF16 reference-fixture generator for Qwen3.8-Flash-Next.

Generates 24-token greedy continuations from the official BF16 checkpoint on
CPU (MoE touches only activated experts; PLE shards are mmapped), recording
chosen-token logprobs and top-20 alternatives per position in the same JSON
schema collect_official.py writes for hosted APIs, so score_official can
score any pack against this exact-checkpoint reference with its full api_*
drift metrics.

Prompt rendering replicates DS4's segmented chat tokenization
(DS4_THINK_NONE, no system prompt) exactly as validated 60/60 against the
pack scorer in /tmp/bf16_nll_qwen4.py.
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
    ap.add_argument('--limit', type=int, default=0)
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
    if args.limit:
        prompts = prompts[:args.limit]

    for sub in ('prompts', 'continuations', 'responses'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    rows = []
    roundtrip_ok = 0
    for i, prompt in enumerate(prompts):
        case_id = f'case_{i:03d}'
        t0 = time.time()
        ids = ds4_chat_ids(tok, prompt)
        input_ids = torch.tensor([ids], dtype=torch.long)
        generated = []
        positions = []
        with torch.inference_mode():
            past = None
            cur = input_ids
            for step in range(args.max_tokens):
                out = model(cur, past_key_values=past, use_cache=True)
                logits = out.logits[0, -1]
                lp = torch.log_softmax(logits.float(), dim=-1)
                top = torch.topk(lp, args.top_logprobs)
                nxt = int(top.indices[0])
                positions.append({
                    'token': tok.decode([nxt]),
                    'bytes': token_bytes(tok, nxt),
                    'logprob': float(top.values[0]),
                    'top_logprobs': [
                        {
                            'token': tok.decode([int(t_)]),
                            'bytes': token_bytes(tok, int(t_)),
                            'logprob': float(v_),
                        }
                        for t_, v_ in zip(top.indices, top.values)
                    ],
                })
                generated.append(nxt)
                past = out.past_key_values
                cur = torch.tensor([[nxt]], dtype=torch.long)
        text = tok.decode(generated, skip_special_tokens=False)
        re_ids = tok(text, add_special_tokens=False)['input_ids']
        if re_ids == generated:
            roundtrip_ok += 1
        else:
            print(f'{case_id}: decode/encode round-trip mismatch '
                  f'({len(generated)} -> {len(re_ids)} tokens); api metrics '
                  f'for this case will be skipped by the scorer',
                  flush=True)
        response = {
            'id': f'bf16-{case_id}',
            'object': 'chat.completion',
            'model': 'local-bf16-qwen3.8-flash-next',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': text},
                'logprobs': {'content': positions},
                'finish_reason': 'length',
            }],
            'usage': {'completion_tokens': len(generated)},
        }
        (open(os.path.join(args.out, 'prompts', f'{case_id}.txt'), 'w')
         .write(prompt))
        (open(os.path.join(args.out, 'continuations', f'{case_id}.txt'), 'w')
         .write(text))
        with open(os.path.join(args.out, 'responses', f'{case_id}.json'),
                  'w') as f:
            json.dump(response, f, ensure_ascii=False, indent=1)
        rows.append(case_id)
        print(f'{case_id}: {text[:60]!r} ({time.time()-t0:.1f}s)', flush=True)

    with open(os.path.join(args.out, 'manifest.tsv'), 'w') as f:
        f.write('# id\tprompt_file\tcontinuation_file\tresponse_file\n')
        for cid in rows:
            f.write('\t'.join([
                cid,
                os.path.join(args.out, 'prompts', f'{cid}.txt'),
                os.path.join(args.out, 'continuations', f'{cid}.txt'),
                os.path.join(args.out, 'responses', f'{cid}.json'),
            ]) + '\n')
    self_nll = sum(-p['logprob'] for p in
                   [pos for r in rows
                    for pos in json.load(open(os.path.join(
                        args.out, 'responses', f'{r}.json'
                    )))['choices'][0]['logprobs']['content']])
    n_pos = sum(len(json.load(open(os.path.join(
        args.out, 'responses', f'{r}.json'
    )))['choices'][0]['logprobs']['content']) for r in rows)
    print(f'wrote {len(rows)} cases to {args.out}; round-trip clean '
          f'{roundtrip_ok}/{len(rows)}; BF16 self-NLL {self_nll/n_pos:.6f} '
          f'over {n_pos} tokens', flush=True)


if __name__ == '__main__':
    main()
