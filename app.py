from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List
import os
import uuid
import time

app = FastAPI(title="VocalForge Audio Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RENDERS = {}

class ChainRequest(BaseModel):
    analysis: Dict
    warmth: int = 50
    brightness: int = 50
    compression: int = 50
    space: int = 50
    character: int = 50

class RenderRequest(BaseModel):
    file_id: str
    chain: List[Dict]
    mode: str = "wet"

@app.get("/health")
def health():
    return {"status": "ok", "message": "VocalForge audio service is alive"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        tmp_path = f"/tmp/{file_id}{suffix}"
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        try:
            import librosa
            import numpy as np
            y, sr = librosa.load(tmp_path, sr=48000, mono=True)

            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key_index = int(np.argmax(np.sum(chroma, axis=1)))
            keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            key = keys[key_index]

            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(np.atleast_1d(tempo)[0])

            rms = librosa.feature.rms(y=y)[0]
            dynamic_range = float(20 * np.log10(np.max(rms) / (np.mean(rms) + 1e-9)))
            peak_db = float(20 * np.log10(np.max(np.abs(y)) + 1e-9))

            spec = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            sib_band = (freqs >= 5000) & (freqs <= 9000)
            sib_energy = float(np.mean(spec[sib_band, :]))
            mud_band = (freqs >= 200) & (freqs <= 400)
            mud_energy = float(np.mean(spec[mud_band, :]))

            issues = []
            if mud_energy > np.mean(spec) * 1.3:
                issues.append("muddy_200_400hz")
            if sib_energy > np.mean(spec) * 1.4:
                issues.append("sibilance_5_9khz")
            if dynamic_range > 15:
                issues.append("wide_dynamic_range")
            if peak_db > -1:
                issues.append("near_clipping")
            if peak_db < -12:
                issues.append("low_recording_level")

            f0, _, _ = librosa.pyin(y, fmin=80, fmax=500, sr=sr)
            f0_clean = f0[~np.isnan(f0)]
            median_f0 = float(np.median(f0_clean)) if len(f0_clean) > 0 else 200

            if median_f0 < 130:
                vocal_type = "male_sung"
            elif median_f0 < 200:
                vocal_type = "female_low_or_male_high"
            else:
                vocal_type = "female_sung"

            RENDERS[file_id] = tmp_path

            return {
                "file_id": file_id,
                "key": key,
                "tempo": round(tempo, 1),
                "vocalType": vocal_type,
                "genre_hint": "pop",
                "issues": issues,
                "dynamic_range_lu": round(dynamic_range, 2),
                "peak_db": round(peak_db, 2),
                "recommended_profile": "Pop Modern"
            }
        except ImportError:
            RENDERS[file_id] = tmp_path
            return {
                "file_id": file_id,
                "key": "C",
                "tempo": 120.0,
                "vocalType": "unknown",
                "genre_hint": "pop",
                "issues": [],
                "dynamic_range_lu": 12.0,
                "peak_db": -6.0,
                "recommended_profile": "Generic"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-chain")
async def generate_chain(req: ChainRequest):
    try:
        analysis = req.analysis
        issues = analysis.get("issues", [])
        warmth = req.warmth / 100.0
        brightness = req.brightness / 100.0
        comp_amount = req.compression / 100.0
        space = req.space / 100.0
        character = req.character / 100.0

        chain = []
        hp_freq = 80 + int(warmth * 20)
        chain.append({"type": "highpass", "freq": hp_freq, "q": 0.7})

        if "muddy_200_400hz" in issues:
            chain.append({"type": "eq_cut", "freq": 280, "gain": round(-2 - warmth * 2, 1), "q": 1.4})

        chain.append({
            "type": "compressor",
            "threshold": round(-14 - comp_amount * 8, 1),
            "ratio": round(2 + comp_amount * 2, 1),
            "attack": 8,
            "release": 80,
            "makeup": round(3 + comp_amount * 3, 1)
        })

        if "sibilance_5_9khz" in issues:
            chain.append({"type": "deesser", "freq": 7200, "threshold": -22, "ratio": 4})

        if character > 0.2:
            chain.append({
                "type": "saturation",
                "style": "tape",
                "drive": round(character * 0.5, 2),
                "mix": round(0.3 + character * 0.3, 2)
            })

        if brightness > 0.3:
            chain.append({
                "type": "eq_boost",
                "freq": 10000,
                "gain": round(brightness * 4, 1),
                "q": 0.8
            })

        if space > 0.1:
            chain.append({
                "type": "reverb",
                "style": "plate",
                "predelay": 20 + int(space * 10),
                "decay": round(1.0 + space * 1.5, 2),
                "wet": round(space * 0.35, 2)
            })

        chain.append({"type": "limiter", "ceiling": -1.0, "target_lufs": -14})

        return {
            "chain": chain,
            "notes": "Applied " + str(len(chain)) + "-stage chain. Detected issues: " + (", ".join(issues) if issues else "none") + "."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def build_pedalboard(chain, include_time_effects=True):
    from pedalboard import (
        Pedalboard, HighpassFilter, Compressor, Reverb,
        Limiter, Gain, PeakFilter, Distortion
    )
    plugins = []
    for step in chain:
        t = step["type"]
        if t == "highpass":
            plugins.append(HighpassFilter(cutoff_frequency_hz=step["freq"]))
        elif t == "eq_cut":
            plugins.append(PeakFilter(cutoff_frequency_hz=step["freq"], gain_db=step["gain"], q=step["q"]))
        elif t == "eq_boost":
            plugins.append(PeakFilter(cutoff_frequency_hz=step["freq"], gain_db=step["gain"], q=step["q"]))
        elif t == "compressor":
            plugins.append(Compressor(threshold_db=step["threshold"], ratio=step["ratio"], attack_ms=step["attack"], release_ms=step["release"]))
            plugins.append(Gain(gain_db=step["makeup"]))
        elif t == "deesser":
            plugins.append(PeakFilter(cutoff_frequency_hz=step["freq"], gain_db=-3, q=2.0))
        elif t == "saturation":
            plugins.append(Distortion(drive_db=step["drive"] * 12))
        elif t == "reverb" and include_time_effects:
            plugins.append(Reverb(room_size=step["decay"] / 3.0, damping=0.5, wet_level=step["wet"], dry_level=1.0 - step["wet"]))
        elif t == "limiter":
            plugins.append(Limiter(threshold_db=step["ceiling"]))
    return Pedalboard(plugins)

@app.post("/render")
async def render(req: RenderRequest):
    try:
        if req.file_id not in RENDERS:
            raise HTTPException(status_code=404, detail="File not found. Analyze first.")

        from pedalboard.io import AudioFile
        src_path = RENDERS[req.file_id]
        include_time = (req.mode == "wet")
        board = build_pedalboard(req.chain, include_time_effects=include_time)

        out_id = str(uuid.uuid4())
        out_path = "/tmp/" + out_id + ".wav"

        with AudioFile(src_path) as f:
            audio = f.read(f.frames)
            sr = f.samplerate

        processed = board(audio, sr)

        with AudioFile(out_path, "w", sr, processed.shape[0]) as f:
            f.write(processed)

        RENDERS[out_id] = out_path
        return {"render_id": out_id, "url": "/download/" + out_id, "mode": req.mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{render_id}")
async def download(render_id: str):
    if render_id not in RENDERS:
        raise HTTPException(status_code=404, detail="Render not found")
    return FileResponse(RENDERS[render_id], media_type="audio/wav")
