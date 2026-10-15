"""VoiceForge: local voice cloning + TTS with inline tags (Chatterbox Turbo engine)."""
import json, re, threading, uuid, webbrowser
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chunker import chunk_text

ROOT = Path(__file__).parent
VOICES, OUT = ROOT / "voices", ROOT / "outputs"
VOICES.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REF_SR, MIN_SEC, MAX_SEC = 24000, 6, 30

app = FastAPI(title="VoiceForge")
_state = {"model": None, "status": "loading", "error": ""}
_lock = threading.Lock()  # one generation at a time keeps VRAM use predictable


def _load_model():
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        _state["model"] = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
        _state["status"] = "ready"
    except Exception as e:  # shown in the UI so you can see what went wrong
        _state["status"], _state["error"] = "error", str(e)


@app.on_event("startup")
def startup():
    threading.Thread(target=_load_model, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    return {"status": _state["status"], "error": _state["error"], "device": DEVICE}


def _voice(vid: str) -> dict:
    p = VOICES / re.sub(r"[^a-f0-9]", "", vid) / "meta.json"
    if not p.exists():
        raise HTTPException(404, "Voice not found")
    return json.loads(p.read_text())


@app.get("/api/voices")
def list_voices():
    metas = [json.loads(p.read_text()) for p in VOICES.glob("*/meta.json")]
    return sorted(metas, key=lambda m: m["created"], reverse=True)


@app.post("/api/voices")
async def add_voice(name: str = Form(...), file: UploadFile = ...):
    vid = uuid.uuid4().hex[:8]
    vdir = VOICES / vid
    vdir.mkdir()
    tmp = vdir / ("upload_" + Path(file.filename or "x").suffix)
    tmp.write_bytes(await file.read())
    try:
        y, _ = librosa.load(str(tmp), sr=REF_SR, mono=True)
    except Exception:
        _rm(vdir)
        raise HTTPException(400, "Could not read that audio. Try a WAV or MP3 file.")
    tmp.unlink()
    if len(y) / REF_SR < MIN_SEC:
        _rm(vdir)
        raise HTTPException(400, f"Clip is too short. Use at least {MIN_SEC} seconds, ideally 10-30.")
    y = y[: REF_SR * MAX_SEC]
    y = y / max(np.abs(y).max(), 1e-6) * 0.95  # gentle peak normalise
    sf.write(vdir / "ref.wav", y, REF_SR)
    meta = {"id": vid, "name": name.strip() or "Untitled", "seed": int(np.random.randint(1, 2**31 - 1)),
            "created": __import__("time").time(), "seconds": round(len(y) / REF_SR, 1)}
    (vdir / "meta.json").write_text(json.dumps(meta))
    return meta


def _rm(d: Path):
    for f in d.glob("*"):
        f.unlink()
    d.rmdir()


@app.delete("/api/voices/{vid}")
def delete_voice(vid: str):
    _voice(vid)
    _rm(VOICES / vid)
    return {"ok": True}


class GenReq(BaseModel):
    voice_id: str
    text: str


@app.post("/api/generate")
def generate(req: GenReq):
    if _state["status"] != "ready":
        raise HTTPException(503, "Model is still loading" if _state["status"] == "loading"
                            else f"Model failed to load: {_state['error']}")
    meta = _voice(req.voice_id)
    ref = str(VOICES / meta["id"] / "ref.wav")
    chunks = chunk_text(req.text)
    if not chunks:
        raise HTTPException(400, "Write some text first.")
    model = _state["model"]
    gap = np.zeros(int(model.sr * 0.18), dtype=np.float32)
    parts = []
    with _lock:
        for c in chunks:
            # same reference + same seed for every chunk = same voice from start to end
            torch.manual_seed(meta["seed"])
            if DEVICE == "cuda":
                torch.cuda.manual_seed_all(meta["seed"])
            wav = model.generate(c, audio_prompt_path=ref)
            parts += [wav.squeeze().detach().cpu().numpy().astype(np.float32), gap]
    audio = np.concatenate(parts[:-1])
    name = f"{uuid.uuid4().hex[:10]}.wav"
    sf.write(OUT / name, audio, model.sr)
    return {"url": f"/api/audio/{name}", "seconds": round(len(audio) / model.sr, 1), "chunks": len(chunks)}


@app.get("/api/audio/{name}")
def audio(name: str):
    if not re.fullmatch(r"[a-f0-9]{10}\.wav", name) or not (OUT / name).exists():
        raise HTTPException(404)
    return FileResponse(OUT / name, media_type="audio/wav", filename=name)


if __name__ == "__main__":
    threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    uvicorn.run(app, host="127.0.0.1", port=7860)
