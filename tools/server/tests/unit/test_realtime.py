import asyncio
import base64
import io
import json
import os
import wave

import aiohttp
import pytest
import requests
from utils import *

# Voxtral Realtime is the smallest model with a streaming audio encoder, too large for the default run
pytestmark = pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")

AUDIO_URL = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"  # 16 kHz mono PCM16, 11 s

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerProcess()
    local_model = os.environ.get("VOXTRAL_RT_LOCAL_MODEL")
    if local_model:
        server.model_hf_repo = None
        server.model_file = local_model
        server.mmproj_file = os.environ.get("VOXTRAL_RT_LOCAL_MMPROJ")
    else:
        server.model_hf_repo = "ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF:Q4_K_M"
    server.model_hf_file = None
    server.n_ctx = 2048
    server.n_slots = 1
    server.temperature = 0.0  # greedy, the asserts check words


@pytest.fixture(scope="module")
def pcm() -> bytes:
    with wave.open(io.BytesIO(requests.get(AUDIO_URL).content)) as w:
        return w.readframes(w.getnframes())


async def run_session(events: list[dict], pcm: bytes) -> list[dict]:
    """Send events ("audio" stands for the whole clip), return every server event until the socket closes."""
    url = f"ws://{server.server_host}:{server.server_port}/v1/realtime"
    received = []
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(url) as ws:
            received.append(json.loads((await ws.receive()).data))
            for ev in events:
                if ev == "audio":
                    for i in range(0, len(pcm), 3200):
                        await ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + 3200]).decode()})
                else:
                    await ws.send_json(ev)
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                received.append(json.loads(msg.data))
    return received


def test_realtime_append_odd_byte_counts(pcm):
    # chunks that split a sample in two must not shift the samples that follow
    server.start()
    # a 1-byte first chunk: if the byte were dropped, every later sample would be read one byte off
    bounds = [0, 1] + list(range(3201, len(pcm), 3200)) + [len(pcm)]
    chunks = [{"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[a:b]).decode()} for a, b in zip(bounds, bounds[1:])]
    events = asyncio.run(run_session(chunks + [{"type": "input_audio_buffer.commit", "final": True}], pcm))
    completed = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed"]
    assert len(completed) == 1
    assert "country" in completed[0]["transcript"].lower()
