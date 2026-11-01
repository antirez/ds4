#!/usr/bin/env python3
"""Batch images and videos through ds4-server's OpenAI-compatible endpoint.

Throughput on ds4 comes from long prompts: prefill amortizes streamed MoE
experts, so its tokens/s grows with prompt length. This tool therefore packs
several images into one request, asking for a JSON object with one answer per
image, and keeps --jobs requests in flight (match the server's resident
sessions). Videos go one per request as video_url parts and need --video on
the server. Images whose answer is missing from a packed reply are retried
alone. Results stream to JSONL, and re-running with the same --out skips
inputs already done. Run with --image-cache-dir on the server and re-runs
skip the vision encoder too.

  tools/ds4-vision-batch.py -p "Transcribe all visible text." shots/ -o out.jsonl
  tools/ds4-vision-batch.py -p "Summarize what happens." --fps 0.5 clips/*.mp4
"""
import argparse, base64, concurrent.futures as cf, json, math, os, struct, sys
import threading, time, urllib.request

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}


def image_size(path):
    """(w, h) from PNG/JPEG/GIF headers, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:6] in (b"GIF87a", b"GIF89a"):
                return struct.unpack("<HH", head[6:10])
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff":
                        b = f.read(1)
                    while b == b"\xff":
                        b = f.read(1)
                    if not b:
                        return None
                    m = b[0]
                    seglen = struct.unpack(">H", f.read(2))[0]
                    if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        h, w = struct.unpack(">xHH", f.read(5))
                        return w, h
                    f.seek(seglen - 2, 1)
    except (OSError, struct.error):
        pass
    return None


def token_cost(path, budget):
    """Approximate LLM tokens for one image: 42 px cells, h*(w+1)+2, capped by budget."""
    wh = image_size(path)
    if not wh:
        return budget or 1024
    gw, gh = math.ceil(wh[0] / 42), math.ceil(wh[1] / 42)
    native = gh * (gw + 1) + 2
    return min(native, budget) if budget else native


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                out += [os.path.join(root, f) for f in sorted(files)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXT | VIDEO_EXT]
        else:
            out.append(p)
    return [os.path.abspath(p) for p in out]


def media_url(path, file_urls):
    if file_urls:
        return "file://" + path
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{MIME.get(ext, 'application/octet-stream')};base64,{data}"


def extract_json(text):
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


class Client:
    def __init__(self, a):
        self.a = a
        self.url = a.server.rstrip("/") + "/v1/chat/completions"

    def chat(self, content, max_tokens):
        body = {"model": "ds4", "temperature": self.a.temperature, "max_tokens": max_tokens,
                "think": self.a.think, "messages": [{"role": "user", "content": content}]}
        req = urllib.request.Request(self.url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=self.a.timeout) as r:
            resp = json.load(r)
        msg = resp["choices"][0]["message"]
        return (msg.get("content") or "").strip(), resp.get("usage", {}), time.time() - t

    def image_part(self, path):
        part = {"type": "image_url", "image_url": {"url": media_url(path, self.a.file_urls)}}
        if self.a.image_tokens:
            part["image_url"]["max_tokens"] = self.a.image_tokens
        else:
            part["image_url"]["detail"] = "native"
        return part

    def run_pack(self, paths):
        if len(paths) == 1:
            content = [self.image_part(paths[0]), {"type": "text", "text": self.a.prompt}]
            text, usage, dt = self.chat(content, self.a.answer_tokens)
            return {paths[0]: text}, usage, dt
        content = []
        for k, p in enumerate(paths, 1):
            content.append({"type": "text", "text": f"IMG{k}:"})
            content.append(self.image_part(p))
        keys = ", ".join(f'"IMG{k}"' for k in range(1, len(paths) + 1))
        content.append({"type": "text", "text":
            f"There are {len(paths)} independent images above, labeled IMG1..IMG{len(paths)}. "
            f"Treat each one separately. For each image: {self.a.prompt}\n"
            f"Reply with only a JSON object with keys {keys}, each mapped to that image's answer as a string."})
        text, usage, dt = self.chat(content, self.a.answer_tokens * len(paths) + 64)
        obj = extract_json(text) or {}
        return {p: obj.get(f"IMG{k}") for k, p in enumerate(paths, 1)}, usage, dt

    def run_video(self, path):
        vu = {"url": media_url(path, self.a.file_urls), "fps": self.a.fps}
        if self.a.video_tokens:
            vu["total_tokens"] = self.a.video_tokens
        if self.a.max_frames:
            vu["max_frames"] = self.a.max_frames
        content = [{"type": "video_url", "video_url": vu}, {"type": "text", "text": self.a.prompt}]
        text, usage, dt = self.chat(content, self.a.answer_tokens * 4)
        return {path: text}, usage, dt


def make_packs(images, a):
    packs, cur, cur_tok = [], [], 0
    for p in images:
        c = token_cost(p, a.image_tokens)
        if cur and (len(cur) >= a.pack or cur_tok + c > a.pack_tokens):
            packs.append(cur)
            cur, cur_tok = [], 0
        cur.append(p)
        cur_tok += c
    if cur:
        packs.append(cur)
    return packs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image/video files or directories")
    ap.add_argument("-p", "--prompt", required=True, help="question asked of every image/video")
    ap.add_argument("-o", "--out", help="JSONL output (appends, resumes); default stdout")
    ap.add_argument("--server", default="http://localhost:8031")
    ap.add_argument("-j", "--jobs", type=int, default=2, help="requests in flight (server resident sessions)")
    ap.add_argument("--pack", type=int, default=32, help="max images per request (1 = no packing)")
    ap.add_argument("--pack-tokens", type=int, default=24576, help="max image tokens per request")
    ap.add_argument("--image-tokens", type=int, default=1024, help="per-image budget; 0 = native resolution")
    ap.add_argument("--answer-tokens", type=int, default=256, help="output tokens allowed per image")
    ap.add_argument("--fps", type=float, default=1.0, help="video sampling rate")
    ap.add_argument("--video-tokens", type=int, default=0, help="token budget per video (server default if 0)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--file-urls", action="store_true",
                    help="send file:// paths instead of uploading (server must share the filesystem; paths under --media-root)")
    ap.add_argument("--think", action="store_true", help="allow reasoning (much slower)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()

    files = collect(a.inputs)
    done = set()
    if a.out and os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("answer") is not None:
                        done.add(r["file"])
                except (json.JSONDecodeError, KeyError):
                    pass
    files = [p for p in files if p not in done]
    images = [p for p in files if os.path.splitext(p)[1].lower() in IMAGE_EXT]
    videos = [p for p in files if os.path.splitext(p)[1].lower() in VIDEO_EXT]
    unknown = [p for p in files if p not in images and p not in videos]
    for p in unknown:
        print(f"skip (unknown type): {p}", file=sys.stderr)
    work = [("pack", pk) for pk in make_packs(images, a)] + [("video", [v]) for v in videos]
    print(f"{len(images)} images in {len(work) - len(videos)} requests, {len(videos)} videos, "
          f"{len(done)} already done, jobs={a.jobs}", file=sys.stderr)

    cl = Client(a)
    out = open(a.out, "a") if a.out else sys.stdout
    lock = threading.Lock()
    stats = {"n": 0, "prompt": 0, "completion": 0, "retried": 0, "failed": 0}
    t0 = time.time()

    def emit(path, answer, usage, dt, n_in_req):
        with lock:
            out.write(json.dumps({"file": path, "answer": answer, "latency_s": round(dt, 2),
                                  "request_images": n_in_req, "usage": usage}) + "\n")
            out.flush()
            stats["n"] += 1
            if answer is None:
                stats["failed"] += 1

    def task(kind, paths):
        try:
            res, usage, dt = cl.run_video(paths[0]) if kind == "video" else cl.run_pack(paths)
        except Exception as e:  # network or server error: retry singly if packed
            res, usage, dt = {p: None for p in paths}, {"error": str(e)}, 0.0
        with lock:
            stats["prompt"] += usage.get("prompt_tokens", 0)
            stats["completion"] += usage.get("completion_tokens", 0)
        for p, ans in res.items():
            if ans is None and len(paths) > 1:
                with lock:
                    stats["retried"] += 1
                try:
                    r1, u1, d1 = cl.run_pack([p])
                    ans, usage, dt = r1[p], u1, d1
                except Exception as e:
                    usage = {"error": str(e)}
                emit(p, ans, usage, dt, 1)
            else:
                emit(p, ans if isinstance(ans, str) or ans is None else json.dumps(ans), usage, dt, len(paths))
        with lock:
            el = time.time() - t0
            print(f"\r{stats['n']}/{len(files)} done  {stats['n'] / el:.2f} items/s  "
                  f"prompt {stats['prompt'] / el:.0f} tok/s", end="", file=sys.stderr)

    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        list(ex.map(lambda w: task(*w), work))
    el = time.time() - t0
    print(f"\n{stats['n']} items in {el:.1f}s ({stats['n'] / max(el, 1e-9):.2f}/s), "
          f"prompt {stats['prompt']} tok, completion {stats['completion']} tok, "
          f"retried {stats['retried']}, failed {stats['failed']}", file=sys.stderr)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
