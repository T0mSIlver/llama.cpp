import base64
import json
import os
import socket
import struct
import pytest
from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinylfm3omni()


class WebSocket:
    def __init__(self, host: str, port: int, path: str):
        self.sock = socket.create_connection((host, port), timeout=60)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        self.stream = self.sock.makefile("rb")
        assert b" 101 " in self.stream.readline()
        while self.stream.readline().strip():
            pass

    def send(self, event: dict):
        payload = json.dumps(event).encode()
        n = len(payload)
        if n < 126:
            header = bytes([0x81, 0x80 | n])
        else:
            header = bytes([0x81, 0x80 | 126]) + struct.pack(">H", n)
        mask = os.urandom(4)
        self.sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def recv(self) -> dict:
        while True:
            header = self.stream.read(2)
            assert len(header) == 2, "the server closed the connection"
            b0, b1 = header
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self.stream.read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self.stream.read(8))[0]
            payload = self.stream.read(n)
            if b0 & 0x0F == 1:
                return json.loads(payload)
            assert b0 & 0x0F != 8, "the server closed the session"
            assert b0 & 0x0F in (9, 10), "expected a text frame" # ping, pong

    def recv_until(self, event_type: str) -> list[dict]:
        events = []
        while not events or events[-1]["type"] != event_type:
            events.append(self.recv())
        return events

    def close(self):
        self.stream.close()
        self.sock.close()


def test_realtime_context_full():
    global server
    server.start()
    ws = WebSocket(server.server_host, server.server_port, "/v1/realtime")
    ws.recv_until("session.created")

    # audio past the end of the context ends the session with an error, the server keeps running
    second = {"type": "input_audio_buffer.append", "audio": base64.b64encode(bytes(16000 * 2)).decode()} # 1 s at 16 kHz
    for _ in range(150):
        try:
            ws.send(second)
        except OSError:
            break # the server ended the session
    error = ws.recv_until("error")[-1]["error"]
    assert "does not fit in the context" in error["message"]
    ws.close()
    assert server.make_request("GET", "/health").status_code == 200

    ws = WebSocket(server.server_host, server.server_port, "/v1/realtime")
    ws.recv_until("session.created")
    ws.close()


def test_realtime_error_code():
    global server
    server.start()
    ws = WebSocket(server.server_host, server.server_port, "/v1/realtime")
    ws.recv_until("session.created")

    # clients match on error.code, error.type is the class of the error
    ws.send({"type": "input_audio_buffer.commit"})
    error = ws.recv_until("error")[-1]["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "input_audio_buffer_commit_empty"

    ws.send({"type": "response.create"})
    error = ws.recv_until("error")[-1]["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "conversation_empty"
    ws.close()
