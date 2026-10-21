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


def item_text(events: list[dict], item_id: str) -> str:
    return "".join(e["delta"] for e in events if e["type"].endswith(".delta") and e["item_id"] == item_id)


def test_realtime_final_commit(pcm):
    server.start()
    events = asyncio.run(run_session(["audio", {"type": "input_audio_buffer.commit", "final": True}], pcm))
    assert events[0]["type"] == "session.created"
    completed = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed"]
    assert len(completed) == 1
    assert completed[0]["transcript"] == item_text(events, completed[0]["item_id"])
    assert "country" in completed[0]["transcript"].lower()


def test_realtime_commit_completes_after_drain(pcm):
    # the words held back by the model's delay at the commit belong to the committed item, not to the next one
    server.start()
    events = asyncio.run(run_session([
        "audio",
        {"type": "input_audio_buffer.commit"},
        {"type": "input_audio_buffer.commit", "final": True},
    ], pcm))
    committed = [e for e in events if e["type"] == "input_audio_buffer.committed"]
    completed = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed"]
    assert len(committed) == 1 and len(completed) == 2
    assert committed[0]["previous_item_id"] is None
    first = committed[0]["item_id"]
    assert completed[0]["item_id"] == first
    assert events.index(committed[0]) < events.index(completed[0])
    assert completed[0]["transcript"] == item_text(events, first)
    assert completed[0]["transcript"].lower().rstrip(" .").endswith("country")
    assert item_text(events, completed[1]["item_id"]).strip() == ""

