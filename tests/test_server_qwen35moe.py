#!/usr/bin/env python3
"""Bounded qwen35moe HTTP compatibility checks against an already-running server.

The parent owns server/model lifecycle. This suite executes no tools: it supplies
fixed clock and file-content results. All requests run sequentially.
"""
import argparse
import copy
import json
import http.client
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

APIS = {
    'chat': '/v1/chat/completions', 'messages': '/v1/messages',
    'responses': '/v1/responses', 'completions': '/v1/completions',
}
MODELS = ['qwen3.6-35b-a3b', 'ornith-1.5-35b-a3b', 'qwen35moe-35b-a3b']


def body_for(api, model, effort='none', limit=16):
    body = {'model': model, 'temperature': 0}
    if api in ('chat', 'messages'):
        body.update(messages=[{'role': 'user', 'content': 'Give one short greeting.'}], max_tokens=limit)
    elif api == 'responses':
        body.update(input='Give one short greeting.', max_output_tokens=limit)
    else:
        body.update(prompt='Give one short greeting.', max_tokens=limit)
    if effort is not None:
        if api == 'responses':
            body['reasoning'] = {'effort': effort}
        else:
            body['reasoning_effort'] = effort
    return body


def visible(api, data):
    if api == 'chat':
        return data['choices'][0]['message'].get('content') or ''
    if api == 'completions':
        return data['choices'][0].get('text') or ''
    if api == 'messages':
        return ''.join(block.get('text', '') for block in data['content'] if block.get('type') == 'text')
    return ''.join(block.get('text', '') for item in data['output'] if item.get('type') == 'message'
                   for block in item.get('content', []) if block.get('type') == 'output_text')


def validate_response(api, data, require_text):
    if api == 'messages':
        assert data['type'] == 'message' and isinstance(data['content'], list), data
    elif api == 'responses':
        assert data['object'] == 'response' and isinstance(data['output'], list), data
    else:
        assert len(data['choices']) >= 1, data
    text = visible(api, data)
    assert isinstance(text, str), data
    assert '<think>' not in text and '</think>' not in text, text
    if require_text:
        assert text.strip(), data
    return text


def sse_events(raw):
    events = []
    for block in raw.decode('utf-8').replace('\r\n', '\n').split('\n\n'):
        fields, event = [], None
        for line in block.splitlines():
            if line.startswith('event:'):
                event = line[6:].lstrip(' ')
            elif line.startswith('data:'):
                fields.append(line[5:].lstrip(' '))
        if fields:
            data = '\n'.join(fields)
            events.append((event, data if data == '[DONE]' else json.loads(data)))
    return events


def validate_stream(api, raw):
    events = sse_events(raw)
    text = []
    if api == 'chat':
        assert events and events[-1][1] == '[DONE]', events
        finished = False
        for _, data in events:
            if data == '[DONE]':
                continue
            for choice in data.get('choices', []):
                text.append(choice.get('delta', {}).get('content') or '')
                finished |= choice.get('finish_reason') is not None
        assert finished, events
    elif api == 'messages':
        kinds = [data.get('type', event) for event, data in events]
        assert kinds and kinds[-1] == 'message_stop' and 'message_delta' in kinds, events
        for _, data in events:
            delta = data.get('delta', {})
            if data.get('type') == 'content_block_delta' and delta.get('type') == 'text_delta':
                text.append(delta['text'])
    else:
        kinds = [data.get('type', event) for event, data in events]
        assert kinds and kinds[-1] == 'response.completed', events
        for _, data in events:
            if data.get('type') == 'response.output_text.delta':
                text.append(data['delta'])
        completed = events[-1][1]['response']
        assert ''.join(text) == visible('responses', completed), completed
    value = ''.join(text)
    assert value.strip(), events
    assert '<think>' not in value and '</think>' not in value, value
    return {'visible_text': value, 'event_count': len(events)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Server base URL, e.g. http://127.0.0.1:8080')
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--output', required=True, type=Path, help='Fresh artifact directory')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--case', choices=['all', 'protocol', 'tools'], default='all',
                        help='Select HTTP protocol checks, model tool probes, or both')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'invocation': sys.argv, 'url': args.url.rstrip('/'), 'model': args.model,
              'case': args.case, 'requests': [], 'checks': [], 'status': 'running'}
    print(f'Artifacts: {out}; selected scope: {args.case}', flush=True)

    def save():
        (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    def request(name, path, body=None, expected=200):
        payload = b'' if body is None else json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        (out / (name + '.request.bin')).write_bytes(payload)
        headers = {'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'}
        req = urllib.request.Request(report['url'] + path, data=None if body is None else payload, headers=headers)
        record = {'name': name, 'method': req.get_method(), 'url': req.full_url,
                  'request_headers': list(req.header_items()), 'expected_status': expected,
                  'status': None, 'response_file': name + '.response.bin'}
        report['requests'].append(record)
        (out / record['response_file']).write_bytes(b'')
        started = time.monotonic()
        try:
            try:
                response = urllib.request.urlopen(req, timeout=args.timeout)
            except urllib.error.HTTPError as error:
                response = error
            with response, (out / record['response_file']).open('wb') as fp:
                record['status'] = response.status
                record['response_headers'] = list(response.headers.raw_items())
                while True:
                    try:
                        chunk = response.read(65536)
                    except http.client.IncompleteRead as exc:
                        fp.write(exc.partial)
                        fp.flush()
                        raise
                    if not chunk:
                        break
                    fp.write(chunk)
                    fp.flush()
            raw = (out / record['response_file']).read_bytes()
            assert record['status'] == expected, (name, record['status'], raw.decode('utf-8', errors='replace'))
            return raw
        except Exception as exc:
            record['error'] = str(exc)
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            save()

    def check(name, fn):
        started = time.monotonic()
        result = {'name': name, 'passed': False}
        report['checks'].append(result)
        try:
            value = fn()
            result['passed'] = True
            result['result'] = value
            print(f'PASS {name}', flush=True)
            return value
        except Exception as exc:
            result['error'] = str(exc)
            print(f'FAIL {name}: {exc}', flush=True)
            return None
        finally:
            result['elapsed_seconds'] = time.monotonic() - started
            save()

    def models():
        data = json.loads(request('models', '/v1/models'))
        ids = [model['id'] for model in data['data']]
        assert args.model in ids, ids
        return {'advertised_ids': ids}

    def normal(api):
        data = json.loads(request(api + '-nonthinking', APIS[api], body_for(api, args.model)))
        validate_response(api, data, True)
        return data

    def numeric(api):
        # A string exercises parse_level plus the family guard, not JSON type rejection.
        data = json.loads(request(api + '-numeric-50', APIS[api], body_for(api, args.model, '50'), expected=400))
        assert 'error' in data, data
        return data

    def named(api):
        data = json.loads(request(api + '-named-low', APIS[api], body_for(api, args.model, 'low', 1)))
        validate_response(api, data, False)
        return data

    def stream(api):
        body = body_for(api, args.model + '-chat', effort=None)
        body['stream'] = True
        raw = request(api + '-stream', APIS[api], body)
        return validate_stream(api, raw)

    save()
    try:
        if args.case in ('all', 'protocol'):
            check('models', models)
            normal_results = {api: check(api + '-nonthinking', lambda api=api: normal(api)) for api in APIS}
            for api in APIS:
                check(api + '-numeric-50', lambda api=api: numeric(api))
                check(api + '-named-low', lambda api=api: named(api))
            for api in ['chat', 'messages', 'responses']:
                check(api + '-stream', lambda api=api: stream(api))

            def continuation():
                first = normal_results['chat']
                assert first is not None, 'Initial chat request failed'
                body = body_for('chat', args.model)
                body['messages'] += [first['choices'][0]['message'], {'role': 'user', 'content': 'Give another short greeting.'}]
                data = json.loads(request('chat-continuation', APIS['chat'], body))
                validate_response('chat', data, True)
                return data

            check('chat-continuation', continuation)
        if args.case in ('all', 'tools'):
            clock_body = body_for('chat', args.model, limit=96)
            clock_body['messages'] = [{'role': 'user', 'content':
                'Call clock exactly once with no arguments to read the test clock. Do not invent a time or answer directly.'}]
            clock_body['tools'] = [{'type': 'function', 'function': {
                'name': 'clock', 'description': 'Read-only test clock. Returns a fixed UTC time supplied by the client. Takes no arguments.',
                'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}]

            def clock_call():
                data = json.loads(request('clock-call', APIS['chat'], clock_body))
                validate_response('chat', data, False)
                choice = data['choices'][0]
                calls = choice['message'].get('tool_calls') or []
                assert choice['finish_reason'] == 'tool_calls' and len(calls) == 1, data
                assert calls[0]['type'] == 'function' and calls[0]['function']['name'] == 'clock', calls
                assert calls[0]['id'] and json.loads(calls[0]['function']['arguments']) == {}, calls
                return data

            clock = check('clock-call', clock_call)

            def clock_result():
                assert clock is not None, 'Clock call request failed'
                body = copy.deepcopy(clock_body)
                message = clock['choices'][0]['message']
                body['messages'] += [message, {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'],
                    'content': '{"utc":"2000-01-01T00:00:00Z"}'},
                    {'role': 'user', 'content': 'Report the returned UTC time briefly.'}]
                body['tool_choice'] = 'none'
                body['max_tokens'] = 16
                data = json.loads(request('clock-result-continuation', APIS['chat'], body))
                validate_response('chat', data, True)
                assert not data['choices'][0]['message'].get('tool_calls'), data
                return data

            check('clock-result-continuation', clock_result)
            read_body = body_for('chat', args.model, effort=None, limit=512)
            read_body['messages'] = [{'role': 'user', 'content':
                'Call the appropriate tool to read src/tokenizer.cpp. Do not list the directory and do not answer directly.'}]
            read_body['tools'] = [{'type': 'function', 'function': {
                'name': name, 'description': description,
                'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                               'required': ['path'], 'additionalProperties': False}}}
                for name, description in [('read_file', 'Read the contents of one file.'),
                                          ('list_directory', 'List the entries of a directory.')]]
            read_body['tool_choice'] = 'auto'
            read_body['parallel_tool_calls'] = False

            def read_call():
                data = json.loads(request('read-file-call', APIS['chat'], read_body))
                validate_response('chat', data, False)
                choice = data['choices'][0]
                calls = choice['message'].get('tool_calls') or []
                assert choice['finish_reason'] == 'tool_calls' and len(calls) == 1, data
                assert calls[0]['type'] == 'function' and calls[0]['function']['name'] == 'read_file', calls
                assert calls[0]['id'] and json.loads(calls[0]['function']['arguments']) == {'path': 'src/tokenizer.cpp'}, calls
                return data

            read = check('read-file-call', read_call)

            def read_result():
                assert read is not None, 'Read-file call request failed'
                sentinel = 'QWEN35MOE_READ_RESULT_7E19C4'
                body = copy.deepcopy(read_body)
                message = read['choices'][0]['message']
                body['messages'] += [message, {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'],
                    'content': 'const char *TEST_FILE_CONSTANT = "' + sentinel + '";\n'},
                    {'role': 'user', 'content':
                     'Report the unique constant string value shown in the tool result. Do not call any more tools.'}]
                body['tool_choice'] = 'none'
                data = json.loads(request('read-file-result-continuation', APIS['chat'], body))
                text = validate_response('chat', data, True)
                assert not data['choices'][0]['message'].get('tool_calls'), data
                assert sentinel in text, data
                return data

            check('read-file-result-continuation', read_result)
        report['status'] = 'passed' if all(item['passed'] for item in report['checks']) else 'failed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        raise
    finally:
        save()
    if report['status'] != 'passed':
        raise SystemExit('FAIL: see retained request/response bytes and results.json')
    print(f'PASS: qwen35moe {args.case} scope; {len(report["checks"])} selected checks')


if __name__ == '__main__':
    main()
