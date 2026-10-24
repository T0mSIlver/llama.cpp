import asyncio
import base64
import wave
import aiohttp
import pytest
from utils import *

server: ServerProcess

# not a ServerPreset, load_all() would download it for every test session
VOXTRAL_RT_HF_REPO = "ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF:Q4_K_M"

# 16 kHz mono PCM16: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
AUDIO_URL = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"
AUDIO_RE = r"^\W*(and so\W+)?my fellow americans\W.*\Wyour country\W*$"

EV_DELTA     = "conversation.item.input_audio_transcription.delta"
EV_COMPLETED = "conversation.item.input_audio_transcription.completed"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()


def start_voxtral_rt():
    global server
    server = ServerProcess()
    server.model_hf_repo = VOXTRAL_RT_HF_REPO
    server.model_hf_file = None
    server.n_ctx = 4096
    server.n_slots = 1
    server.temperature = 0.0
    server.start(timeout_seconds=600)


def load_pcm16() -> bytes:
    with wave.open(download_file(AUDIO_URL), "rb") as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (16000, 1, 2)
        return f.readframes(f.getnframes())


def append_events(pcm: bytes) -> list[dict]:
    # 100 ms chunks, like a microphone would send them
    chunk = 3200
    return [
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + chunk]).decode()}
        for i in range(0, len(pcm), chunk)
    ]


def run_realtime(client_events: list[dict]) -> list[dict]:
    """ Send client_events once the session is created, return every server event until the server closes the socket. """
    async def session() -> list[dict]:
        events = []
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(server.make_url("/v1/realtime")) as ws:
                async for msg in ws:
                    assert msg.type == aiohttp.WSMsgType.TEXT
                    events.append(json.loads(msg.data))
                    if events[-1]["type"] == "session.created":
                        for ev in client_events:
                            await ws.send_json(ev)
        return events
    events = asyncio.run(asyncio.wait_for(session(), timeout=DEFAULT_REQUEST_TIMEOUT))
    print("Events from server", json.dumps([ev for ev in events if ev["type"] != EV_DELTA], indent=2))
    return events


def test_realtime_unsupported_model():
    global server
    server.start()
    events = run_realtime([])
    assert [ev["type"] for ev in events] == ["error"]
    assert "does not support realtime" in events[0]["error"]["message"]


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_realtime_control_events():
    start_voxtral_rt()
    events = run_realtime([
        {"type": "session.update", "session": {}},
        {"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}}},
        {"type": "session.update", "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}},
        {"type": "input_audio_buffer.clear"},
        {"type": "not.an.event"},
        {"type": "input_audio_buffer.commit", "final": True},
    ])
    assert [ev["type"] for ev in events if ev["type"] != EV_DELTA] == [
        "session.created",
        "session.updated",
        "error",
        "session.updated",
        "input_audio_buffer.cleared",
        "error",
        EV_COMPLETED,
    ]
    assert events[1]["session"] == events[0]["session"]
    assert "must be 16000" in events[2]["error"]["message"]


# ref: https://github.com/ggml-org/llama.cpp/issues/30968
@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_realtime_commit():
    start_voxtral_rt()
    pcm = load_pcm16()
    events = run_realtime([
        *append_events(pcm),
        {"type": "input_audio_buffer.commit"},
        *append_events(pcm),
        {"type": "input_audio_buffer.commit", "final": True},
    ])
    assert events[0]["type"] == "session.created"
    assert events[0]["session"]["audio"]["input"]["format"]["rate"] == 16000

    committed = [ev for ev in events if ev["type"] == "input_audio_buffer.committed"]
    completed = [ev for ev in events if ev["type"] == EV_COMPLETED]
    assert len(committed) == 1
    assert len(completed) == 2
    assert committed[0]["previous_item_id"] is None
    assert committed[0]["item_id"] == completed[0]["item_id"]
    assert completed[1]["item_id"] != completed[0]["item_id"]
    assert events.index(committed[0]) < events.index(completed[0])
    assert events[-1] == completed[1]

    for item in completed:
        deltas = [ev["delta"] for ev in events if ev["type"] == EV_DELTA and ev["item_id"] == item["item_id"]]
        assert item["transcript"] == "".join(deltas)
        # the words spoken right before a commit belong to the committed item, not to the next one
        assert match_regex(AUDIO_RE, item["transcript"])
