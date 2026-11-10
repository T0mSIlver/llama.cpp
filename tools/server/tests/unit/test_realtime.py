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


def user_text(text: str) -> dict:
    return {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}}


def test_realtime_response_create_while_responding():
    global server
    # the first response runs to its 256 tokens, so it is still in progress when the second response.create is read
    server.ignore_eos = True
    server.start()
    ws = WebSocket(server.server_host, server.server_port, "/v1/realtime")
    ws.recv_until("session.created")
    ws.send({"type": "session.update", "session": {"type": "realtime", "output_modalities": ["text"]}})
    ws.recv_until("session.updated")

    # a second response.create while the first one is generating
    ws.send(user_text("Count to one hundred."))
    ws.send({"type": "response.create", "response": {"max_output_tokens": 256}})
    ws.send(user_text("And now?"))
    ws.send({"type": "response.create", "response": {"max_output_tokens": 8}})
    events = ws.recv_until("response.done")
    first = next(e for e in events if e["type"] == "response.created")["response"]["id"]
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["error"]["type"] == "invalid_request_error"
    assert errors[0]["error"]["code"] == "conversation_already_has_active_response"
    assert [e["response"]["id"] for e in events if e["type"] == "response.created"] == [first]
    assert all(e["response_id"] == first for e in events if e["type"] == "response.output_text.delta")
    assert events[-1]["response"]["id"] == first

    # once it is done, the next response answers the second message
    ws.send({"type": "response.create", "response": {"max_output_tokens": 8}})
    events = ws.recv_until("response.done")
    second = next(e for e in events if e["type"] == "response.created")["response"]["id"]
    assert second != first
    assert events[-1]["response"]["id"] == second
    ws.close()


def test_realtime_clear_transcription():
    global server
    server.start()
    ws = WebSocket(server.server_host, server.server_port, "/v1/realtime")
    ws.recv_until("session.created")
    # 24 kHz goes through the resampler
    ws.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}}})
    ws.recv_until("session.updated")
    second = {"type": "input_audio_buffer.append", "audio": base64.b64encode(bytes(24000 * 2)).decode()} # 1 s of silence

    def append_10s():
        for _ in range(10):
            ws.send(second)

    # nothing left to commit after a clear
    append_10s()
    ws.send({"type": "input_audio_buffer.clear"})
    ws.recv_until("input_audio_buffer.cleared")
    ws.send({"type": "input_audio_buffer.commit"})
    ev = ws.recv()
    assert ev["type"] == "error"
    assert ev["error"]["code"] == "input_audio_buffer_commit_empty"

    # if the cleared audio stayed in the slot, 100 s of it would not fit in the context
    for _ in range(10):
        append_10s()
        ws.send({"type": "input_audio_buffer.clear"})
        assert not any(e["type"] == "error" for e in ws.recv_until("input_audio_buffer.cleared"))
    append_10s()
    ws.send({"type": "input_audio_buffer.commit"})
    events = ws.recv_until("conversation.item.input_audio_transcription.completed")
    assert not any(e["type"] == "error" for e in events)
    ws.close()
