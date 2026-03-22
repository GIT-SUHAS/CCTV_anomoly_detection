"""
CCTV Anomaly Detection — Web Dashboard
=======================================
Flask-based frontend showcasing all project modules:
  • Phase 5: Digital Twin "Dreaming" Engine
  • Phase 6: Audio-Visual Cross-Modal Verification
  • Phase 7: "Common Sense" Reasoning Engine

Usage:
    python web_app.py
    → Open http://localhost:5000
"""

import os
import sys
import time
import json
import base64
import traceback
import warnings

import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

FRAME_H, FRAME_W = 224, 224

# ─────────────────────────────────────────────────────────────────────────────
# LAZY-LOADED MODULES
# ─────────────────────────────────────────────────────────────────────────────
_modules = {
    "dreaming": {"loaded": False, "engine": None, "error": None},
    "audio_visual": {"loaded": False, "detector": None, "audio_gen": None, "error": None},
    "reasoning": {"loaded": False, "manager": None, "error": None},
}


def _init_dreaming():
    """Initialize the Dreaming Engine."""
    if _modules["dreaming"]["loaded"]:
        return
    try:
        from dreaming_engine import DreamingEngine
        _modules["dreaming"]["engine"] = DreamingEngine(use_diffusion=False, seed=42)
        _modules["dreaming"]["loaded"] = True
        print("[WebApp] ✓ Dreaming Engine loaded.")
    except Exception as e:
        _modules["dreaming"]["error"] = str(e)
        print(f"[WebApp] ✗ Dreaming Engine failed: {e}")


def _init_audio_visual():
    """Initialize Audio-Visual cross-modal detector."""
    if _modules["audio_visual"]["loaded"]:
        return
    try:
        import torch
        from audio_visual_fusion import CrossModalAnomalyDetector, SyntheticAudioGenerator

        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                              else "cpu")

        audio_gen = SyntheticAudioGenerator(seed=42)
        detector = CrossModalAnomalyDetector(device=device)

        # Quick train on synthetic normal data
        print("[WebApp] Training Audio-Visual model on synthetic data…")
        n = 20
        normal_audio = [audio_gen.generate("ambient", 1.0) for _ in range(n)]
        normal_frames = []
        for _ in range(n):
            bg = np.random.uniform(0.4, 0.6, (FRAME_H, FRAME_W)).astype(np.float32)
            bg += np.random.randn(FRAME_H, FRAME_W).astype(np.float32) * 0.02
            normal_frames.append(np.clip(bg, 0, 1))
        detector.train_on_normal(normal_audio, normal_frames, epochs=10)

        _modules["audio_visual"]["detector"] = detector
        _modules["audio_visual"]["audio_gen"] = audio_gen
        _modules["audio_visual"]["loaded"] = True
        print("[WebApp] ✓ Audio-Visual module loaded and trained.")
    except Exception as e:
        _modules["audio_visual"]["error"] = str(e)
        print(f"[WebApp] ✗ Audio-Visual module failed: {e}")


def _init_reasoning():
    """Initialize the Reasoning Engine."""
    if _modules["reasoning"]["loaded"]:
        return
    try:
        from reasoning_engine import AlertManager
        _modules["reasoning"]["manager"] = AlertManager(use_llm=True)
        _modules["reasoning"]["loaded"] = True
        print("[WebApp] ✓ Reasoning Engine loaded.")
    except Exception as e:
        _modules["reasoning"]["error"] = str(e)
        print(f"[WebApp] ✗ Reasoning Engine failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def numpy_to_base64(frame: np.ndarray) -> str:
    """Convert a numpy grayscale frame [0,1] to base64 PNG."""
    frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    # Apply a color map for visual appeal
    colored = cv2.applyColorMap(frame_u8, cv2.COLORMAP_BONE)
    _, buf = cv2.imencode(".png", colored)
    return base64.b64encode(buf).decode("utf-8")


def numpy_to_base64_gray(frame: np.ndarray) -> str:
    """Convert to base64 without color map."""
    frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", frame_u8)
    return base64.b64encode(buf).decode("utf-8")


def create_hallway_bg(brightness: float = 0.55) -> np.ndarray:
    """Create a synthetic hallway background."""
    bg = np.ones((FRAME_H, FRAME_W), dtype=np.float32) * brightness
    bg[FRAME_H // 2:, :] = brightness - 0.15
    for y in range(FRAME_H):
        vx = FRAME_W // 2
        left = int(vx - (vx * (y / FRAME_H)))
        right = int(vx + (vx * (y / FRAME_H)))
        if left > 0:
            bg[y, :left] = brightness - 0.25
        if right < FRAME_W:
            bg[y, right:] = brightness - 0.25
    bg += np.random.randn(FRAME_H, FRAME_W).astype(np.float32) * 0.015
    return np.clip(bg, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Pages
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────────────────────────────────────
# API — System
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/system/status")
def system_status():
    return jsonify({
        "dreaming": {
            "loaded": _modules["dreaming"]["loaded"],
            "error": _modules["dreaming"]["error"],
        },
        "audio_visual": {
            "loaded": _modules["audio_visual"]["loaded"],
            "error": _modules["audio_visual"]["error"],
        },
        "reasoning": {
            "loaded": _modules["reasoning"]["loaded"],
            "error": _modules["reasoning"]["error"],
        },
    })


@app.route("/api/system/init", methods=["POST"])
def system_init():
    """Initialize all modules."""
    _init_dreaming()
    _init_reasoning()
    _init_audio_visual()
    return system_status()


@app.route("/api/system/init/<module_name>", methods=["POST"])
def system_init_module(module_name):
    """Initialize a specific module."""
    if module_name == "dreaming":
        _init_dreaming()
    elif module_name == "audio_visual":
        _init_audio_visual()
    elif module_name == "reasoning":
        _init_reasoning()
    return system_status()


# ─────────────────────────────────────────────────────────────────────────────
# API — Dreaming Engine (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/dreaming/generate", methods=["POST"])
def dreaming_generate():
    if not _modules["dreaming"]["loaded"]:
        return jsonify({"error": "Dreaming Engine not loaded"}), 503

    data = request.json or {}
    anomaly_type = data.get("anomaly_type", "fire")
    num_frames = min(int(data.get("num_frames", 5)), 10)
    brightness = float(data.get("brightness", 0.55))

    engine = _modules["dreaming"]["engine"]

    # Create background
    bg = create_hallway_bg(brightness)
    bg_b64 = numpy_to_base64(bg)

    # Generate synthetic anomalies
    frames = engine.simple_augmentor.generate(bg, anomaly_type=anomaly_type,
                                               num_frames=num_frames)
    frames_b64 = [numpy_to_base64(f) for f in frames]

    return jsonify({
        "background": bg_b64,
        "anomaly_type": anomaly_type,
        "frames": frames_b64,
        "count": len(frames),
    })


@app.route("/api/dreaming/types")
def dreaming_types():
    return jsonify({
        "types": [
            {"id": "fire", "name": "Fire / Glow", "icon": "🔥",
             "desc": "Procedural fire with flickering glow effect"},
            {"id": "fallen_person", "name": "Fallen Person", "icon": "🧑",
             "desc": "Person collapsed on the ground"},
            {"id": "intruder", "name": "Intruder", "icon": "🕵️",
             "desc": "Standing silhouette with motion blur"},
            {"id": "abandoned_object", "name": "Abandoned Object", "icon": "🎒",
             "desc": "Suspicious bag or box left unattended"},
        ]
    })


# ─────────────────────────────────────────────────────────────────────────────
# API — Audio-Visual Fusion (Phase 6)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/audio-visual/detect", methods=["POST"])
def av_detect():
    if not _modules["audio_visual"]["loaded"]:
        return jsonify({"error": "Audio-Visual module not loaded"}), 503

    data = request.json or {}
    audio_type = data.get("audio_type", "ambient")
    brightness = float(data.get("brightness", 0.5))

    detector = _modules["audio_visual"]["detector"]
    audio_gen = _modules["audio_visual"]["audio_gen"]

    # Generate audio and frame
    audio = audio_gen.generate(audio_type, 1.0)
    frame = np.full((FRAME_H, FRAME_W), brightness, dtype=np.float32)
    frame += np.random.randn(FRAME_H, FRAME_W).astype(np.float32) * 0.02
    frame = np.clip(frame, 0, 1)

    result = detector.detect(audio, frame)
    frame_b64 = numpy_to_base64(frame)

    # Generate simple waveform visualization
    waveform_points = audio[::max(1, len(audio) // 200)].tolist()

    return jsonify({
        "audio_type": audio_type,
        "similarity": round(result["similarity"], 4),
        "z_score": round(result["z_score"], 4),
        "is_anomaly": result["is_anomaly"],
        "severity": result["severity"],
        "frame": frame_b64,
        "waveform": waveform_points,
    })


@app.route("/api/audio-visual/types")
def av_types():
    return jsonify({
        "types": [
            {"id": "ambient", "name": "Ambient Noise", "icon": "🔈",
             "desc": "Normal background hum", "expected": "Normal"},
            {"id": "scream", "name": "Scream", "icon": "😱",
             "desc": "High-pitch vocal distress", "expected": "Anomaly"},
            {"id": "glass_breaking", "name": "Glass Breaking", "icon": "💥",
             "desc": "Shattering glass impact", "expected": "Anomaly"},
            {"id": "gunshot", "name": "Gunshot", "icon": "🔫",
             "desc": "Sharp impulse with reverb", "expected": "Anomaly"},
        ]
    })


# ─────────────────────────────────────────────────────────────────────────────
# API — Reasoning Engine (Phase 7)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/reasoning/process", methods=["POST"])
def reasoning_process():
    if not _modules["reasoning"]["loaded"]:
        return jsonify({"error": "Reasoning Engine not loaded"}), 503

    from reasoning_engine import AnomalyAlert, AlertDecision

    data = request.json or {}
    alert_type = data.get("alert_type", "running_person")
    anomaly_score = float(data.get("anomaly_score", 0.65))
    location = data.get("location", "hallway")
    brightness = float(data.get("brightness", 0.5))
    av_score = float(data.get("audio_visual_score", 0.1))

    manager = _modules["reasoning"]["manager"]

    alert = AnomalyAlert(
        timestamp=time.time(),
        clip_index=1,
        frame_index=50,
        anomaly_score=anomaly_score,
        alert_type=alert_type,
        reconstruction_error=anomaly_score * 0.1,
        audio_visual_score=av_score,
    )

    frame = np.full((FRAME_H, FRAME_W), brightness, dtype=np.float32)
    frame += np.random.randn(FRAME_H, FRAME_W).astype(np.float32) * 0.02
    frame = np.clip(frame, 0, 1)

    result = manager.process_alert(alert, frame, location_hint=location)

    stats = manager.get_stats()

    return jsonify({
        "decision": result.decision.value,
        "confidence": round(result.confidence, 2),
        "reasoning": result.reasoning,
        "llm_used": result.llm_used,
        "scene_context": {
            "location": result.scene_context.location_type,
            "time_of_day": result.scene_context.time_of_day,
            "crowd_density": result.scene_context.crowd_density,
            "lighting": result.scene_context.lighting,
            "actions": result.scene_context.detected_actions,
            "objects": result.scene_context.detected_objects,
        },
        "stats": stats,
    })


@app.route("/api/reasoning/presets")
def reasoning_presets():
    return jsonify({
        "presets": [
            {
                "name": "Running at Bus Stop",
                "desc": "Person sprinting to catch a departing bus — should suppress",
                "expected": "suppress",
                "params": {"alert_type": "running_person", "anomaly_score": 0.65,
                           "location": "bus_stop", "brightness": 0.6, "audio_visual_score": 0.1},
            },
            {
                "name": "Fall in Dark Hallway",
                "desc": "Person collapses in a poorly lit, empty corridor — should escalate",
                "expected": "escalate",
                "params": {"alert_type": "falling_person", "anomaly_score": 0.85,
                           "location": "hallway", "brightness": 0.12, "audio_visual_score": 0.6},
            },
            {
                "name": "Waiting in Lobby",
                "desc": "Person standing in a daytime lobby — should suppress",
                "expected": "suppress",
                "params": {"alert_type": "loitering", "anomaly_score": 0.45,
                           "location": "lobby", "brightness": 0.55, "audio_visual_score": 0.05},
            },
            {
                "name": "Fight in Parking Lot",
                "desc": "Violent altercation in a parking lot — should escalate",
                "expected": "escalate",
                "params": {"alert_type": "fighting", "anomaly_score": 0.92,
                           "location": "parking_lot", "brightness": 0.3, "audio_visual_score": 0.85},
            },
            {
                "name": "Night Corridor Runner",
                "desc": "Person running through empty corridor at night — should confirm",
                "expected": "confirm",
                "params": {"alert_type": "running_person", "anomaly_score": 0.7,
                           "location": "hallway", "brightness": 0.18, "audio_visual_score": 0.3},
            },
        ],
        "alert_types": [
            "running_person", "falling_person", "fighting",
            "loitering", "crowd_anomaly", "intrusion",
        ],
        "locations": [
            "hallway", "parking_lot", "bus_stop", "lobby",
            "entrance", "stairwell", "elevator", "outdoor_path",
        ],
    })


# ─────────────────────────────────────────────────────────────────────────────
# API — Full Pipeline Demo
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/pipeline/run", methods=["POST"])
def pipeline_run():
    """Run all three modules in sequence on a scenario."""
    results = {"steps": [], "success": True}

    data = request.json or {}
    anomaly_type = data.get("anomaly_type", "intruder")
    audio_type = data.get("audio_type", "glass_breaking")
    location = data.get("location", "hallway")
    brightness = float(data.get("brightness", 0.55))

    # Step 1: Dreaming Engine
    if _modules["dreaming"]["loaded"]:
        engine = _modules["dreaming"]["engine"]
        bg = create_hallway_bg(brightness)
        frames = engine.simple_augmentor.generate(bg, anomaly_type=anomaly_type, num_frames=3)
        results["steps"].append({
            "phase": 5,
            "name": "Dreaming Engine",
            "status": "success",
            "data": {
                "anomaly_type": anomaly_type,
                "frames_generated": len(frames),
                "background": numpy_to_base64(bg),
                "sample_frame": numpy_to_base64(frames[0]) if frames else None,
            }
        })
    else:
        results["steps"].append({
            "phase": 5, "name": "Dreaming Engine",
            "status": "unavailable", "data": {}
        })

    # Step 2: Audio-Visual
    if _modules["audio_visual"]["loaded"]:
        detector = _modules["audio_visual"]["detector"]
        audio_gen = _modules["audio_visual"]["audio_gen"]
        audio = audio_gen.generate(audio_type, 1.0)
        frame = np.full((FRAME_H, FRAME_W), brightness, dtype=np.float32)
        frame += np.random.randn(FRAME_H, FRAME_W).astype(np.float32) * 0.02
        frame = np.clip(frame, 0, 1)
        av_result = detector.detect(audio, frame)
        results["steps"].append({
            "phase": 6,
            "name": "Audio-Visual Fusion",
            "status": "success",
            "data": {
                "audio_type": audio_type,
                "similarity": round(av_result["similarity"], 4),
                "z_score": round(av_result["z_score"], 4),
                "is_anomaly": av_result["is_anomaly"],
                "severity": av_result["severity"],
            }
        })
        av_z = av_result["z_score"]
    else:
        results["steps"].append({
            "phase": 6, "name": "Audio-Visual Fusion",
            "status": "unavailable", "data": {}
        })
        av_z = 0

    # Step 3: Reasoning Engine
    if _modules["reasoning"]["loaded"]:
        from reasoning_engine import AnomalyAlert
        manager = _modules["reasoning"]["manager"]
        alert = AnomalyAlert(
            timestamp=time.time(), clip_index=1, frame_index=50,
            anomaly_score=0.75, alert_type=anomaly_type,
            reconstruction_error=0.06,
            audio_visual_score=max(0, av_z / 5.0),
        )
        frame = np.full((FRAME_H, FRAME_W), brightness, dtype=np.float32)
        frame += np.random.randn(FRAME_H, FRAME_W).astype(np.float32) * 0.02
        frame = np.clip(frame, 0, 1)
        r = manager.process_alert(alert, frame, location_hint=location)
        results["steps"].append({
            "phase": 7,
            "name": "Reasoning Engine",
            "status": "success",
            "data": {
                "decision": r.decision.value,
                "confidence": round(r.confidence, 2),
                "reasoning": r.reasoning,
                "llm_used": r.llm_used,
            }
        })
    else:
        results["steps"].append({
            "phase": 7, "name": "Reasoning Engine",
            "status": "unavailable", "data": {}
        })

    return jsonify(results)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CCTV Anomaly Detection — Web Dashboard")
    print("=" * 60)
    print("\nInitializing modules (this may take a moment)…\n")

    _init_dreaming()
    _init_reasoning()
    _init_audio_visual()

    print("\n" + "=" * 60)
    print("  Dashboard ready → http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5050, debug=False)
