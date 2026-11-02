#!/usr/bin/env python3
"""Opt-in real native-agent vision session acceptance (one model at a time).

The real ~/.ds4/kvcache is used without changing HOME. Existing user KV files
are only stat'ed; sysprompt.kv and editor history are moved aside and restored.
Only checkpoints whose saved title contains this run's nonce are collected.
"""
from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
import uuid

import pexpect
from PIL import Image

ANSI = re.compile(rb'\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[78]')
STATUS = re.compile(r'ctx\s+[0-9.k]+/[0-9.k]+\s*\|\s*([^\r\n]*?)(?: {2,}|[\r\n]|$)')
KV_NAME = re.compile(r'^[0-9a-fA-F]{40}\.kv$')
MARKER = 'DS4-VISION-INTEGRATION:'


def plain_terminal(data: bytes) -> str:
    return ANSI.sub(b'', data).decode('utf-8', 'replace')


def visible_text(data: bytes) -> str:
    return STATUS.sub('', plain_terminal(data))


def file_stat(path: Path) -> dict:
    s = path.lstat()
    return dict(dev=s.st_dev, ino=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns,
                mode=s.st_mode, uid=s.st_uid)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def no_other_model() -> None:
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,comm='], capture_output=True,
                            text=True, check=True, timeout=10)
    found = []
    for line in result.stdout.splitlines():
        values = line.strip().split(None, 1)
        if len(values) != 2:
            continue
        name = Path(values[1]).name
        if name == 'ds4' or name.startswith(('ds4-agent', 'ds4-server')):
            found.append(line.strip())
    require(not found, 'Another native DS4 process exists; no model was started: '+repr(found))


def session_metadata(path: Path, section_copy: Path | None = None) -> dict:
    """Skip KV tensors; stream only the bounded saved DS4V input for CRC/hash."""
    with path.open('rb') as f:
        size = os.fstat(f.fileno()).st_size
        h = f.read(52)
        require(len(h) == 52 and h[:3] == b'KVC', 'Invalid checkpoint header')
        version, quant, reason, flags, model_id = h[3:8]
        require(version in (1, 2) and h[20] == 2 and reason == 6, 'Unexpected native agent cache format')
        tokens, hits, context = struct.unpack_from('<III', h, 8)
        created, updated, payload = struct.unpack_from('<QQQ', h, 24)
        text_bytes = struct.unpack_from('<I', h, 48)[0]
        require(52+text_bytes+payload <= size and text_bytes <= 32*1024**2, 'Invalid text/KV bounds')
        text = f.read(text_bytes).decode('utf-8', 'replace')
        f.seek(52+text_bytes+payload)
        require(flags & 8, 'Integration checkpoints must have a title')
        raw_length = f.read(4)
        require(len(raw_length) == 4, 'Missing title length')
        length = struct.unpack('<I', raw_length)[0]
        require(length <= 1024**2 and f.tell()+length <= size, 'Invalid title length')
        title_raw = f.read(length)
        title = title_raw.decode('utf-8', 'replace')
        identity = hashlib.sha1(title_raw+struct.pack('<Q', created)).hexdigest()
        require(path.stem.lower() == identity, 'Checkpoint identity mismatch')
        row = dict(id=identity, title=title, format_version=version, quant_bits=quant,
                   model_id=model_id, flags=flags, tokens=tokens, saved_context=context,
                   payload_bytes=payload, file_bytes=size, image_count=0, hits=hits,
                   created=created, updated=updated, text=text)
        start = f.tell()
        if version == 1:
            require(not flags & 16 and start == size, 'Unexpected input in text-only cache')
            return row
        require(flags == 24, 'Vision cache must use exactly TITLE|INPUT flags')
        vh = f.read(32)
        require(len(vh) == 32 and vh[:4] == b'DS4V', 'Missing DS4V section')
        v, total, exact, count, n_embd, reserved = struct.unpack_from('<IQIIII', vh, 4)
        header_bytes = 96 if v == 1 else 112
        require(v in (1, 2) and reserved == 0 and exact == tokens and count > 0 and n_embd > 0,
                'Invalid visual input header')
        require(start+total == size and header_bytes+4+exact*4 <= total, 'Invalid visual section bounds')
        f.seek(start+header_bytes+exact*4)
        spans = []
        previous_end = 0
        for _ in range(count):
            raw = f.read(68)
            require(len(raw) == 68, 'Truncated visual metadata')
            values = struct.unpack_from('<9I', raw)
            token_start, token_count = values[:2]
            require(token_count > 0 and token_start >= previous_end and token_start+token_count <= exact,
                    'Invalid visual token span')
            require(f.tell()+token_count*n_embd*4 <= start+total-4, 'Truncated visual embedding')
            spans.append(dict(token_start=token_start, token_count=token_count,
                              layout=values[2], grid_w=values[3], grid_h=values[4],
                              width=values[5], height=values[6], content_w=values[7],
                              content_h=values[8], fingerprint=raw[36:68].hex()))
            f.seek(token_count*n_embd*4, os.SEEK_CUR)
            previous_end = token_start+token_count
        require(f.tell() == start+total-4, 'Extra bytes inside visual section')
        crc_raw = f.read(4)
        require(len(crc_raw) == 4, 'Missing visual CRC')
        expected_crc = struct.unpack('<I', crc_raw)[0]
        f.seek(start)
        digest = hashlib.sha256()
        crc = 0
        remaining = total-4
        destination = section_copy.open('wb') if section_copy else None
        try:
            while remaining:
                block = f.read(min(1024**2, remaining))
                require(block, 'Truncated input while computing CRC')
                digest.update(block)
                crc = binascii.crc32(block, crc)
                if destination: destination.write(block)
                remaining -= len(block)
            digest.update(crc_raw)
            if destination: destination.write(crc_raw)
        finally:
            if destination: destination.close()
        require(crc & 0xffffffff == expected_crc, 'Visual input CRC mismatch')
        row.update(image_count=count, n_embd=n_embd, spans=spans, input_version=v,
                   input_section_bytes=total, input_sha256=digest.hexdigest(),
                   input_crc32=f'{expected_crc:08x}')
        return row


def public_metadata(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != 'text'}


def owned_title(path: Path) -> str:
    """Identify a test file even if its input/CRC is the failed test outcome."""
    fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        st = os.fstat(f.fileno())
        require(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid(), 'Not an owned regular checkpoint')
        h = f.read(52)
        require(len(h) == 52 and h[:3] == b'KVC' and h[5] == 6 and h[6] & 8, 'No agent title')
        payload = struct.unpack_from('<Q', h, 40)[0]
        text_bytes = struct.unpack_from('<I', h, 48)[0]
        f.seek(52+text_bytes+payload)
        raw = f.read(4)
        require(len(raw) == 4, 'No title length')
        length = struct.unpack('<I', raw)[0]
        require(length <= 1024**2 and f.tell()+length <= st.st_size, 'Invalid owned title bounds')
        title = f.read(length)
        created = struct.unpack_from('<Q', h, 24)[0]
        require(hashlib.sha1(title+struct.pack('<Q',created)).hexdigest() == path.stem.lower(), 'Owned title identity mismatch')
        return title.decode('utf-8', 'replace')


class CacheOwnership:
    def __init__(self, output: Path, nonce: str):
        self.output, self.nonce = output, nonce
        self.cache = Path.home()/'.ds4/kvcache'
        self.cache.mkdir(parents=True, exist_ok=True)
        require(self.cache.stat().st_dev == output.stat().st_dev,
                'Output and real cache must share a filesystem for inode-preserving moves')
        self.before = {p.name: file_stat(p) for p in self.cache.iterdir() if p.suffix == '.kv'}
        self.originals = {}
        self.claimed = set()
        self.managed = set()
        self.active = False

    def prepare(self):
        write_json(self.output/'preexisting-kv.json', self.before)
        self.active = True  # Restore earlier moves even if a later move fails.
        for name, path in [('sysprompt.kv', self.cache/'sysprompt.kv'),
                           ('editor-history', Path.home()/'.ds4_agent_history')]:
            if path.exists() or path.is_symlink():
                saved = self.output/'backup'/name
                require(not saved.exists(), 'Backup already exists: '+str(saved))
                snapshot = file_stat(path)
                os.rename(path, saved)
                self.originals[name] = (path, saved, snapshot)
            self.managed.add(name)

    def recognize(self, prefix: str) -> Path:
        matches = [p for p in self.cache.iterdir() if KV_NAME.fullmatch(p.name) and p.stem.lower().startswith(prefix.lower())]
        require(len(matches) == 1, 'Reported SHA does not identify one checkpoint: '+prefix)
        path = matches[0]
        require(path.name not in self.before, 'Refusing to claim a pre-existing user checkpoint')
        require(MARKER+self.nonce in owned_title(path), 'New checkpoint is not owned by this run')
        self.claimed.add(path.name)
        return path

    def cleanup(self) -> dict:
        if not self.active: return {'prepared': False}
        errors = []
        moved = []
        # Recover only this nonce's checkpoints if a command failed before SHA delivery.
        for p in self.cache.iterdir():
            if p.name in self.before or not KV_NAME.fullmatch(p.name): continue
            try:
                if MARKER+self.nonce in owned_title(p):
                    self.claimed.add(p.name)
            except Exception:
                pass  # Unrecognized files remain in place, never broadly deleted.
        for name in sorted(self.claimed):
            p = self.cache/name
            if not p.exists(): continue
            try:
                require(name not in self.before, 'Refusing to move an original checkpoint')
                require(MARKER+self.nonce in owned_title(p), 'Claimed checkpoint was replaced with another file')
                destination = self.output/'artifacts'/name
                require(not destination.exists(), 'Artifact already exists: '+name)
                os.rename(p, destination)
                moved.append(name)
            except Exception as e: errors.append(str(e))
        for name, path in [('sysprompt.kv', self.cache/'sysprompt.kv'),
                           ('editor-history', Path.home()/'.ds4_agent_history')]:
            if name not in self.managed: continue
            try:
                if path.exists() or path.is_symlink():
                    # These paths were absent during preparation or moved aside.
                    destination = self.output/'artifacts'/('generated-'+name)
                    require(not destination.exists(), 'Generated artifact collision')
                    os.rename(path, destination)
                if name in self.originals:
                    original_path, saved, snapshot = self.originals[name]
                    os.rename(saved, original_path)
                    require(file_stat(original_path) == snapshot, 'Original inode/mtime was not restored: '+name)
            except Exception as e: errors.append(str(e))
        unchanged = {}
        for name, expected in self.before.items():
            try:
                unchanged[name] = file_stat(self.cache/name) == expected
            except OSError:
                unchanged[name] = False
        if not all(unchanged.values()): errors.append('Pre-existing user KV stat changed')
        unclaimed = [p.name for p in self.cache.iterdir()
                     if KV_NAME.fullmatch(p.name) and p.name not in self.before]
        if unclaimed: errors.append('New checkpoint ownership could not be proven; left in place')
        self.active = False
        return dict(moved_test_files=moved, original_stats_unchanged=unchanged,
                    restored_backups=list(self.originals), unclaimed_new_files=unclaimed, errors=errors)


class NativeController:
    def __init__(self, args, project: Path, output: Path, phase: str):
        self.args, self.project, self.output, self.phase = args, project, output, phase
        self.trace_path = output/(phase+'-trace.log')
        self.raw = bytearray()
        self.trace_data = ''
        self.trace_offset = 0
        self.last_cpr = -1
        self.closed = False
        self.raw_file = (output/(phase+'-terminal.log')).open('wb')
        command = ['--model', str(args.model), '--vision', str(args.vision),
                   '--ctx', str(args.ctx), '--prefill-chunk', '1024', '--mtp',
                   '--mtp-draft', '1', '--power', '100', '--temp', '0', '--nothink',
                   '--trace', str(self.trace_path), '--chdir', str(project)]
        self.command = [str(args.binary), *command]
        self.child = pexpect.spawn(str(args.binary), command, cwd=str(args.binary.parent),
                                  encoding=None, timeout=1, dimensions=(24, 200), echo=False,
                                  maxread=65536)
        self.child.logfile_read = self.raw_file
        self.child.delaybeforesend = 0.03

    def wait_for_prompt(self):
        # Caller has already retained this controller for its failure/finally path.
        self.wait(lambda: 'tokens label=initial_system_prompt' in self.trace_data and self.is_idle(),
                  self.args.startup_timeout, 'initial native prompt')

    def update_trace(self):
        if not self.trace_path.exists(): return
        with self.trace_path.open('rb') as f:
            f.seek(self.trace_offset)
            data = f.read()
            self.trace_offset += len(data)
        self.trace_data += data.decode('utf-8', 'replace')

    def pump(self):
        try:
            data = self.child.read_nonblocking(65536, timeout=0.15)
        except pexpect.TIMEOUT:
            data = b''
        except pexpect.EOF:
            self.update_trace()
            raise RuntimeError('Native agent exited during '+self.phase)
        old_size = len(self.raw)
        before = bytes(self.raw[-8:])
        self.raw.extend(data)
        for match in re.finditer(rb'\x1b\[6n', before+data):
            offset = old_size-len(before)+match.start()
            if offset > self.last_cpr:
                self.child.send(b'\x1b[22;1R')
                self.last_cpr = offset
        self.update_trace()
        return bool(data)

    def is_idle(self):
        states = STATUS.findall(plain_terminal(bytes(self.raw[-65536:])))
        return bool(states and states[-1].strip().startswith('idle'))

    def wait(self, predicate, timeout, label):
        deadline = time.monotonic()+timeout
        quiet_since = None
        while time.monotonic() < deadline:
            had_output = self.pump()
            if predicate():
                if quiet_since is None or had_output: quiet_since = time.monotonic()
                if time.monotonic()-quiet_since >= 0.35: return
            else: quiet_since = None
        raise TimeoutError('Timed out waiting for '+label)

    def execute(self, command: str, expected: str | None = None, *, model_turn=False) -> dict:
        require('\n' not in command and '\r' not in command, 'Commands must be single-line')
        self.update_trace()
        raw_start, trace_start = len(self.raw), len(self.trace_data)
        started = time.monotonic()
        self.child.send(command.encode()+b'\r')
        def ready():
            trace = self.trace_data[trace_start:]
            plain = visible_text(bytes(self.raw[raw_start:]))
            if model_turn:
                return 'generation finished' in trace and self.is_idle()
            return bool(expected and re.search(expected, plain, re.S)) and self.is_idle()
        self.wait(ready, self.args.timeout if model_turn else self.args.startup_timeout, command[:70])
        return dict(command=command, seconds=round(time.monotonic()-started, 3),
                    plain=visible_text(bytes(self.raw[raw_start:])), trace=self.trace_data[trace_start:])

    def close(self) -> dict:
        if self.closed: return {'already_closed': True}
        try:
            if self.child.isalive():
                self.child.sendcontrol('c')
                time.sleep(0.15)
                self.child.send(b'/quit\r')
                deadline = time.monotonic()+30
                answered = 0
                while self.child.isalive() and time.monotonic() < deadline:
                    try: self.pump()
                    except RuntimeError: break
                    plain = visible_text(bytes(self.raw))
                    prompts = len(re.findall(r'Save current session\? \(y/n\)', plain))
                    if prompts > answered:
                        self.child.send(b'n\r')
                        answered = prompts
                if self.child.isalive(): self.child.terminate(force=True)
            self.child.close(force=True)
            require(not self.child.isalive(), 'Owned native child could not be stopped')
            return dict(exitstatus=self.child.exitstatus, signalstatus=self.child.signalstatus)
        finally:
            try:
                if self.child.isalive(): self.child.terminate(force=True)
                self.child.close(force=True)
                require(not self.child.isalive(), 'Owned child still resident after forced teardown')
            finally:
                self.raw_file.close()
                self.closed = True


def save_session(controller: NativeController, ownership: CacheOwnership) -> tuple[Path, dict]:
    event = controller.execute('/save', r'saved session ([0-9a-f]{8}) \(\d+ tokens\)')
    match = re.search(r'saved session ([0-9a-f]{8}) \(\d+ tokens\)', event['plain'])
    require(match is not None, 'No saved SHA in actual native output')
    path = ownership.recognize(match[1])
    row = session_metadata(path)
    return path, dict(seconds=event['seconds'], metadata=public_metadata(row),
                     native_save_trace=re.findall(r'[^\n]*session save[^\n]*', event['trace']))


def switch_session(controller: NativeController, path: Path) -> dict:
    return controller.execute('/switch '+path.stem, r'switched to session '+path.stem[:8]+r' \(')


def tool_oracle(event: dict):
    # Native render labels are observed only after this prompt, never history replay.
    for name in ('Reading', 'edit', r'\$'):
        require(re.search(r'🛠(?:️)?\s*'+name, event['plain']), 'Required native tool missing: '+name)
    generated = b''.join(bytes.fromhex(value) for value in
                        re.findall(r' token index=\d+ .* hex=([0-9a-f]+)', event['trace'])).decode('utf-8', 'replace')
    require('<function=view_image>' not in generated, 'Restore test reopened an image')


def last_assistant(text: str) -> str:
    for prefix in ('<|im_start|>assistant', '<｜Assistant｜>', '<|assistant|>'):
        if prefix in text:
            body = text.rsplit(prefix, 1)[1]
            if prefix == '<|im_start|>assistant' and '\n' in body:
                body = body.split('\n', 1)[1]
            body = re.split(r'<\|im_end\|>|<｜end▁of▁sentence｜>|<\|user\|>', body)[0].strip()
            # Native --nothink prefixes an empty think block for Qwen/GLM.
            body = re.sub(r'^<think>\s*</think>\s*|^</think>\s*', '', body)
            return body.strip()
    raise AssertionError('Cannot identify final assistant text in saved native transcript')


def check_hot_answer(project: Path):
    answer = json.loads((project/'hot-note.json').read_text())
    require(str(answer.get('train')).strip() == '482', 'Hot restore train OCR failed')
    require(str(answer.get('access')).strip().upper() == 'MINT-731', 'Hot restore access OCR failed')
    return answer


def check_cold_answer(project: Path):
    answer = json.loads((project/'cold-note.json').read_text())
    for color, shape, positions in [('blue', 'circle', ('upper left', 'top left')),
                                     ('red', 'triangle', ('upper right', 'top right')),
                                     ('green', 'square', ('lower center', 'bottom center', 'below both'))]:
        value = answer.get(color, {})
        require(isinstance(value, dict), 'Cold shape object missing: '+color)
        require(str(value.get('shape', '')).strip().lower() == shape, 'Cold shape failed: '+color)
        position = str(value.get('position', '')).strip().lower().replace('-', ' ')
        require(any(x in position for x in positions), 'Cold spatial relation failed: '+color)
    return answer


def run(args):
    no_other_model()
    output = args.output
    require(not output.exists() or not any(output.iterdir()), 'Output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('project', 'backup', 'artifacts', 'hidden-images'):
        (output/name).mkdir()
    project = output/'project'
    nonce = uuid.uuid4().hex
    report = dict(nonce=nonce, status='running', arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, steps=[])
    ownership = CacheOwnership(output, nonce)
    controller = None
    def record(name, data):
        report['steps'].append(dict(name=name, **data))
        write_json(output/'result.json', report)
        print(json.dumps({'step':name,'status':'passed'}, ensure_ascii=False), flush=True)
    try:
        fixtures = Path(__file__).resolve().parent/'vision-fixtures/glm53'
        for name in ('text.png', 'spatial.png'):
            source = fixtures/name
            with Image.open(source) as image: image.verify()
            shutil.copy2(source, project/name)
        ownership.prepare()
        controller = NativeController(args, project, output, 'initial')
        controller.wait_for_prompt()
        record('initial_start', {'pid':controller.child.pid, 'argv':controller.command})
        initial = MARKER+nonce+':vision. Use only view_image to view text.png and spatial.png, one observation for each image. Then answer exactly READY-'+nonce+'. Do not transcribe, describe, summarize, or mention any contents of either image. Do not use any other tools.'
        controller.execute(initial, model_turn=True)
        visual, event = save_session(controller, ownership)
        row = session_metadata(visual)
        require(row['format_version'] == 2 and row['image_count'] == 2 and row['payload_bytes'] > 0, 'Expected hot v2 session containing exactly two images')
        require(last_assistant(row['text']) == 'READY-'+nonce, 'Initial assistant output was not only READY')
        for hidden_fact in ('mint-731', 'north harbor', '19:45'):
            require(hidden_fact not in row['text'].lower(), 'Image fact leaked into saved text before restore: '+hidden_fact)
        record('save_visual_v2', event)
        controller.execute('/new', r'DwarfStar.*?Agent')
        controller.execute(MARKER+nonce+':plain. No tools. Answer only PLAIN-'+nonce+'.', model_turn=True)
        text_session, text_event = save_session(controller, ownership)
        require(session_metadata(text_session)['format_version'] == 1, 'Plain session unexpectedly kept visual input')
        record('save_plain_v1', text_event)
        switch_session(controller, visual)
        visual_roundtrip, ve = save_session(controller, ownership)
        require(visual_roundtrip == visual and ve['metadata']['image_count'] == 2, 'Visual roundtrip lost images or identity')
        switch_session(controller, text_session)
        _, pe = save_session(controller, ownership)
        require(pe['metadata']['format_version'] == 1 and pe['metadata']['image_count'] == 0, 'Plain switch leaked visual input')
        switch_session(controller, visual)
        record('visual_plain_visual_switch', {'visual':ve, 'plain':pe})
        record('initial_exit', controller.close())
        controller = None
        for name in ('text.png', 'spatial.png'):
            os.rename(project/name, output/'hidden-images'/name)
        write_json(project/'hot-note.json', {'train':'TRAIN_PENDING', 'access':'ACCESS_PENDING'})
        write_json(project/'cold-note.json', {color:{'shape':'SHAPE_PENDING','position':'POSITION_PENDING'} for color in ('blue','red','green')})
        no_other_model()
        controller = NativeController(args, project, output, 'hot')
        controller.wait_for_prompt()
        restored = switch_session(controller, visual)
        require(re.search(r'vision restore tokens=\d+ images=2 path=hot cached=\d+', restored['trace']), 'Native trace did not confirm hot visual restore')
        saved_tokens = session_metadata(visual)['tokens']
        hot_prompt = 'Use the previously viewed ticket image from this restored session. Do not view or reopen images and do not read files outside this project. Use native read on hot-note.json, then native edit to replace its pending train and access values with the exact ticket train number and access code. Finally use native bash to parse hot-note.json as JSON and print it. Do not write or use other tools. Reply HOT_DONE.'
        hot_event = controller.execute(hot_prompt, model_turn=True)
        tool_oracle(hot_event)
        prefill = re.search(r'prefill tool_round=0 transcript=(\d+) prompt=(\d+) cached=(\d+) suffix=(\d+)', hot_event['trace'])
        require(prefill and int(prefill[3]) >= saved_tokens, 'Hot continuation did not reuse saved KV prefix')
        record('hot_restore_and_tools', {'restore_seconds':restored['seconds'], 'turn_seconds':hot_event['seconds'],
                                         'restore_trace':re.findall(r'[^\n]*vision restore[^\n]*', restored['trace']),
                                         'prefill':dict(zip(('transcript','prompt','cached','suffix'), map(int,prefill.groups()))),
                                         'answer':check_hot_answer(project)})
        before = session_metadata(visual, output/'artifacts/input-before-strip.bin')
        strip_event = controller.execute('/strip '+visual.stem, r'stripped session '+visual.stem[:8]+r' \(')
        after = session_metadata(visual, output/'artifacts/input-after-strip.bin')
        require(after['payload_bytes'] == 0 and after['format_version'] == 2, 'Strip did not produce cold v2')
        require(before['input_sha256'] == after['input_sha256'] and before['input_crc32'] == after['input_crc32'], 'Strip changed visual input bytes or CRC')
        record('strip_preserves_input', {'seconds':strip_event['seconds'], 'before':public_metadata(before), 'after':public_metadata(after)})
        record('hot_exit_without_resaving', controller.close())
        controller = None
        require(session_metadata(visual)['payload_bytes'] == 0, 'Exit unexpectedly overwrote the stripped checkpoint')
        no_other_model()
        controller = NativeController(args, project, output, 'cold')
        controller.wait_for_prompt()
        restored = switch_session(controller, visual)
        require(re.search(r'vision restore tokens=\d+ images=2 path=cold cached=0', restored['trace']), 'Native trace did not confirm cold visual rebuild')
        cold_prompt = 'Use the previously viewed colored-shapes image from this restored session. Do not view or reopen images and do not read files outside this project. Use native read on cold-note.json, then native edit to replace each color\'s pending shape and position with its shape type and relative position in that image. Use English positions. Finally use native bash to parse cold-note.json as JSON and print it. Do not write or use other tools. Reply COLD_DONE.'
        cold_event = controller.execute(cold_prompt, model_turn=True)
        tool_oracle(cold_event)
        cold_prefill = re.search(r'prefill tool_round=0 transcript=(\d+) prompt=(\d+) cached=(\d+) suffix=(\d+)', cold_event['trace'])
        require(cold_prefill and int(cold_prefill[3]) >= after['tokens'], 'Cold continuation did not reuse the rebuilt KV prefix')
        record('cold_restore_and_tools', {'restore_seconds':restored['seconds'], 'turn_seconds':cold_event['seconds'],
                                          'restore_trace':re.findall(r'[^\n]*vision restore[^\n]*', restored['trace']),
                                          'prefill':dict(zip(('transcript','prompt','cached','suffix'), map(int,cold_prefill.groups()))),
                                          'answer':check_cold_answer(project)})
        record('cold_exit_without_resaving', controller.close())
        controller = None
        report['status'] = 'passed'
    except BaseException as e:
        report.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        stopped = True
        if controller is not None:
            try: report['failed_process_exit'] = controller.close()
            except Exception as e: report['stop_error'] = str(e)
            try: stopped = not controller.child.isalive()
            except Exception: stopped = False
        # An unkillable owned process must never write into the restored user cache.
        report['cleanup'] = ownership.cleanup() if stopped else {'errors':['Owned model is still alive; backups retained in output/backup']}
        if report['cleanup'].get('errors'): report['status'] = 'failed'
        write_json(output/'result.json', report)
    require(report['status'] == 'passed', 'Acceptance or exact cleanup failed; see result.json')
    print(json.dumps({'status':'passed','result':str(output/'result.json')}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--vision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ctx', type=int, default=32768)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--startup-timeout', type=float, default=300)
    args = parser.parse_args()
    for name in ('binary','model','vision'):
        path = getattr(args, name).expanduser().resolve(strict=True)
        require(path.is_file(), 'Not a file: '+str(path))
        setattr(args, name, path)
    args.output = args.output.expanduser().absolute()
    cache = (Path.home()/'.ds4/kvcache').resolve()
    require(args.output.resolve() != cache and cache not in args.output.resolve().parents,
            'Output must be outside the real user cache')
    require(args.ctx > 0 and args.timeout > 0 and args.startup_timeout > 0, 'Limits must be positive')
    run(args)


if __name__ == '__main__':
    main()
