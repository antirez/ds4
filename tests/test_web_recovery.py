"""Local CDP fixture: last-tab retention, HTTP recovery, and failure reporting."""
import base64
import hashlib
import http.server
import json
import pathlib
import struct
import subprocess
import sys
import threading


def main():
    state = {"creates": 0, "puts": 0, "lists": 0, "closes": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/json/version":
                self.reply({"webSocketDebuggerUrl": f"ws://127.0.0.1:{self.server.server_port}/browser"})
            elif self.path == "/json/list":
                state["lists"] += 1
                self.reply([{"type": "page"}] * state["lists"] + [{"type": "service_worker"}])
            elif self.path.startswith("/json/close/"):
                state["closes"].append(self.path)
                self.reply({})
            elif self.path == "/browser":
                key = self.headers["Sec-WebSocket-Key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", base64.b64encode(hashlib.sha1(key.encode()).digest()).decode())
                self.end_headers()
                header = self.rfile.read(2)
                length = header[1] & 127
                if length == 126:
                    length = struct.unpack("!H", self.rfile.read(2))[0]
                elif length == 127:
                    length = struct.unpack("!Q", self.rfile.read(8))[0]
                mask = self.rfile.read(4)
                payload = self.rfile.read(length)
                request = json.loads(bytes(v ^ mask[i % 4] for i, v in enumerate(payload)))
                assert request["method"] == "Target.createTarget", request
                state["creates"] += 1
                response = {"id": request["id"]}
                if state["creates"] == 2:
                    response["result"] = {"targetId": "cdp-target"}
                else:
                    response["error"] = {"code": -32000, "message": "No browser window"}
                data = json.dumps(response).encode()
                assert len(data) < 126
                self.wfile.write(bytes([0x81, len(data)]) + data)
                self.wfile.flush()
            else:
                self.send_error(404)

        def do_PUT(self):
            assert self.path == "/json/new?about%3Ablank", self.path
            state["puts"] += 1
            self.reply({"id": "http-target"} if state["puts"] == 1 else {})

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        binary = pathlib.Path(sys.argv[1]).resolve()
        subprocess.run([str(binary), str(server.server_port)], check=True, timeout=30)
        assert state == {"creates": 3, "puts": 2, "lists": 2, "closes": ["/json/close/cdp-target"]}, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
