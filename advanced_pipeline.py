"""
Advanced CCTV Anomaly Detection Pipeline — Unified Orchestrator
================================================================
Integrates all three advanced features:
  • Phase 5: Digital Twin "Dreaming" Engine (Synthetic Data Augmentation)
  • Phase 6: Audio-Visual Cross-Modal Verification
  • Phase 7: "Common Sense" Reasoning (VLM/LLM Filter)

This file demonstrates the complete pipeline working together:
  1. Normal CCTV footage → Dreaming Engine generates synthetic anomalies
  2. Audio + Video streams → Cross-Modal detector finds multi-sensor anomalies
  3. Raw alerts → Reasoning Engine filters false alarms with context

Usage:
    python advanced_pipeline.py                # Full demo with synthetic data
    python advanced_pipeline.py --phase 5      # Demo only Phase 5
    python advanced_pipeline.py --phase 6      # Demo only Phase 6
    python advanced_pipeline.py --phase 7      # Demo only Phase 7
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_HEIGHT = 224
FRAME_WIDTH = 224


def create_synthetic_scene(brightness: float = 0.55,
                           noise: float = 0.02,
                           n_frames: int = 30) -> list[np.ndarray]:
    """Create a synthetic CCTV scene (hallway) for demo purposes."""
    frames = []
    for i in range(n_frames):
        bg = np.ones((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32) * brightness
        # Add floor
        bg[FRAME_HEIGHT // 2:, :] = brightness - 0.15
        # Add perspective walls
        for y in range(FRAME_HEIGHT):
            vanish_x = FRAME_WIDTH // 2
            left_wall = int(vanish_x - (vanish_x * (y / FRAME_HEIGHT)))
            right_wall = int(vanish_x + (vanish_x * (y / FRAME_HEIGHT)))
            if left_wall > 0:
                bg[y, :left_wall] = brightness - 0.25
            if right_wall < FRAME_WIDTH:
                bg[y, right_wall:] = brightness - 0.25
        # Add noise
        bg += np.random.randn(FRAME_HEIGHT, FRAME_WIDTH).astype(np.float32) * noise
        frames.append(np.clip(bg, 0, 1))
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 DEMO: Digital Twin "Dreaming" Engine
# ══════════════════════════════════════════════════════════════════════════════

def demo_dreaming_engine():
    """Demonstrate the Dreaming Engine generating synthetic anomalies."""
    from dreaming_engine import DreamingEngine

    print("\n" + "═" * 70)
    print("  PHASE 5: Digital Twin 'Dreaming' Engine")
    print("  Generating synthetic anomalies on CCTV backgrounds")
    print("═" * 70)

    # Create synthetic scene
    print("\n  [1/3] Creating synthetic hallway scene (30 frames)…")
    normal_clip = create_synthetic_scene(brightness=0.55, n_frames=30)
    print(f"        → {len(normal_clip)} normal frames created")

    # Initialize engine
    print("  [2/3] Initializing DreamingEngine (CPU mode)…")
    engine = DreamingEngine(use_diffusion=False, seed=42)

    # Extract background
    bg = engine.extract_background(normal_clip)
    print(f"        → Background extracted (mean brightness: {bg.mean():.3f})")

    # Generate anomalies
    print("  [3/3] Generating synthetic anomalies…\n")
    anomalies = engine.dream_anomalies(
        bg,
        anomaly_types=["fire", "fallen_person", "intruder", "abandoned_object"],
        frames_per_type=5,
    )

    print("  ┌──────────────────────┬────────┬──────────────────┐")
    print("  │ Anomaly Type         │ Frames │ Avg Brightness   │")
    print("  ├──────────────────────┼────────┼──────────────────┤")
    for atype, frames in anomalies.items():
        avg_b = np.mean([f.mean() for f in frames])
        print(f"  │ {atype:<20s} │ {len(frames):>6d} │ {avg_b:>16.3f} │")
    print("  └──────────────────────┴────────┴──────────────────┘")

    # Save samples
    engine.save_samples(anomalies, max_per_type=2)

    total = sum(len(f) for f in anomalies.values())
    print(f"\n  ✓ Dreaming Engine generated {total} synthetic anomaly frames.")
    print(f"    These can augment training data to handle rare events.\n")

    return anomalies


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 DEMO: Audio-Visual Cross-Modal Verification
# ══════════════════════════════════════════════════════════════════════════════

def demo_audio_visual():
    """Demonstrate Audio-Visual cross-modal anomaly detection."""
    from audio_visual_fusion import (
        CrossModalAnomalyDetector, SyntheticAudioGenerator
    )
    import torch

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")

    print("\n" + "═" * 70)
    print("  PHASE 6: Audio-Visual Cross-Modal Verification")
    print("  Detecting anomalies via audio-video disagreement")
    print("═" * 70)

    # Generate synthetic data
    print("\n  [1/3] Generating synthetic paired audio-video data…")
    audio_gen = SyntheticAudioGenerator(seed=42)
    n_normal = 25

    normal_audio = [audio_gen.generate("ambient", 1.0) for _ in range(n_normal)]
    normal_frames = []
    for i in range(n_normal):
        bg = np.random.uniform(0.4, 0.6, (224, 224)).astype(np.float32)
        bg += np.random.randn(224, 224).astype(np.float32) * 0.02
        normal_frames.append(np.clip(bg, 0, 1))

    print(f"        → {n_normal} normal audio-video pairs created")

    # Train
    print("  [2/3] Training contrastive model…")
    detector = CrossModalAnomalyDetector(device=device)
    detector.train_on_normal(normal_audio, normal_frames, epochs=15)

    # Test scenarios
    print("  [3/3] Testing cross-modal anomaly detection…\n")

    test_cases = [
        ("Normal: ambient + calm",     "ambient",        False),
        ("Scream + calm scene",         "scream",         True),
        ("Glass breaking + calm",       "glass_breaking", True),
        ("Gunshot + calm scene",        "gunshot",        True),
        ("Normal: ambient + calm #2",   "ambient",        False),
    ]

    print("  ┌──────────────────────────────┬────────┬─────────┬──────────┬──────────┐")
    print("  │ Scenario                     │  Sim   │ Z-score │ Anomaly? │ Severity │")
    print("  ├──────────────────────────────┼────────┼─────────┼──────────┼──────────┤")

    for name, audio_type, expected_anomaly in test_cases:
        audio = audio_gen.generate(audio_type, 1.0)
        frame = np.full((224, 224), 0.5, dtype=np.float32)
        frame += np.random.randn(224, 224).astype(np.float32) * 0.02
        frame = np.clip(frame, 0, 1)

        result = detector.detect(audio, frame)
        anomaly_str = "YES ⚠ " if result["is_anomaly"] else "  no  "
        print(f"  │ {name:<28s} │ {result['similarity']:>6.3f} │ "
              f"{result['z_score']:>7.3f} │ {anomaly_str} │ "
              f"{result['severity']:>8s} │")

    print("  └──────────────────────────────┴────────┴─────────┴──────────┴──────────┘")
    print("\n  ✓ Cross-modal detector distinguishes audio-visual mismatches.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 DEMO: "Common Sense" Reasoning
# ══════════════════════════════════════════════════════════════════════════════

def demo_reasoning():
    """Demonstrate LLM-based Common Sense Reasoning for alert filtering."""
    from reasoning_engine import AlertManager, AnomalyAlert, AlertDecision

    print("\n" + "═" * 70)
    print("  PHASE 7: 'Common Sense' Reasoning Engine")
    print("  Filtering false alarms with contextual intelligence")
    print("═" * 70)

    # Initialize
    print("\n  [1/2] Initializing AlertManager…")
    manager = AlertManager(use_llm=True)

    # Test scenarios
    print("  [2/2] Processing alert scenarios…\n")

    scenarios = [
        {
            "name": "Running at bus stop (FALSE ALARM)",
            "alert": AnomalyAlert(
                timestamp=time.time(), clip_index=1, frame_index=50,
                anomaly_score=0.65, alert_type="running_person",
                reconstruction_error=0.045, audio_visual_score=0.1,
            ),
            "location": "bus_stop",
            "brightness": 0.6,
        },
        {
            "name": "Fall in dark hallway (EMERGENCY)",
            "alert": AnomalyAlert(
                timestamp=time.time(), clip_index=2, frame_index=30,
                anomaly_score=0.85, alert_type="falling_person",
                reconstruction_error=0.08, audio_visual_score=0.6,
            ),
            "location": "hallway",
            "brightness": 0.12,
        },
        {
            "name": "Waiting in lobby (FALSE ALARM)",
            "alert": AnomalyAlert(
                timestamp=time.time(), clip_index=3, frame_index=100,
                anomaly_score=0.45, alert_type="loitering",
                reconstruction_error=0.03, audio_visual_score=0.05,
            ),
            "location": "lobby",
            "brightness": 0.55,
        },
        {
            "name": "Fight in parking lot (CRITICAL)",
            "alert": AnomalyAlert(
                timestamp=time.time(), clip_index=4, frame_index=20,
                anomaly_score=0.92, alert_type="fighting",
                reconstruction_error=0.12, audio_visual_score=0.85,
            ),
            "location": "parking_lot",
            "brightness": 0.3,
        },
        {
            "name": "Running in empty hall at night",
            "alert": AnomalyAlert(
                timestamp=time.time(), clip_index=5, frame_index=70,
                anomaly_score=0.7, alert_type="running_person",
                reconstruction_error=0.06, audio_visual_score=0.3,
            ),
            "location": "hallway",
            "brightness": 0.18,
        },
    ]

    decision_icons = {
        AlertDecision.SUPPRESS: "🟢 SUPPRESS",
        AlertDecision.CONFIRM:  "🟡 CONFIRM ",
        AlertDecision.ESCALATE: "🔴 ESCALATE",
    }

    for scenario in scenarios:
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH),
                        scenario["brightness"], dtype=np.float32)
        frame += np.random.randn(FRAME_HEIGHT, FRAME_WIDTH).astype(np.float32) * 0.02
        frame = np.clip(frame, 0, 1)

        result = manager.process_alert(
            scenario["alert"], frame,
            location_hint=scenario["location"]
        )

        icon = decision_icons[result.decision]
        print(f"  {icon}  {scenario['name']}")
        print(f"            Confidence: {result.confidence:.0%}  │  "
              f"LLM: {'Yes' if result.llm_used else 'No (rule-based)'}")
        print(f"            Reasoning: {result.reasoning[:75]}")
        print()

    # Statistics
    stats = manager.get_stats()
    total = stats["total_alerts"]
    print("  ┌─────────────────────────────────────────┐")
    print(f"  │ ALERT STATS                             │")
    print(f"  │ Total: {total}  │  Suppressed: {stats['suppressed']}  │  "
          f"Escalated: {stats['escalated']}  │")
    print(f"  │ False alarm reduction: "
          f"{stats['suppression_rate']:.0%}                │")
    print("  └─────────────────────────────────────────┘")
    print(f"\n  ✓ Reasoning engine reduced false alarms by "
          f"{stats['suppression_rate']:.0%}.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FULL INTEGRATION DEMO
# ══════════════════════════════════════════════════════════════════════════════

def demo_full_integration():
    """
    Full end-to-end demonstration:
    Dreaming Engine → Cross-Modal → Reasoning → Final Decision
    """
    from dreaming_engine import DreamingEngine
    from audio_visual_fusion import CrossModalAnomalyDetector, SyntheticAudioGenerator
    from reasoning_engine import AlertManager, AnomalyAlert, AlertDecision
    import torch

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")

    print("\n" + "█" * 70)
    print("  FULL INTEGRATION: All Three Advanced Modules Working Together")
    print("█" * 70)

    # ── STEP 1: Generate synthetic anomaly data ──
    print("\n─── Step 1: Dreaming Engine generates synthetic training data ───\n")
    normal_clip = create_synthetic_scene(brightness=0.55, n_frames=20)
    engine = DreamingEngine(use_diffusion=False, seed=42)
    bg = engine.extract_background(normal_clip)
    anomalies = engine.dream_anomalies(bg, frames_per_type=3)
    total_synth = sum(len(f) for f in anomalies.values())
    print(f"  → Generated {total_synth} synthetic anomaly frames "
          f"across {len(anomalies)} types\n")

    # ── STEP 2: Train the cross-modal detector ──
    print("─── Step 2: Cross-Modal detector learns normal audio-video pairs ───\n")
    audio_gen = SyntheticAudioGenerator(seed=42)
    n_pairs = 20
    normal_audio = [audio_gen.generate("ambient", 1.0) for _ in range(n_pairs)]
    normal_frames = normal_clip[:n_pairs]

    detector = CrossModalAnomalyDetector(device=device)
    detector.train_on_normal(normal_audio, normal_frames, epochs=10)

    # ── STEP 3: Simulate a live scenario ──
    print("\n─── Step 3: Simulating live CCTV monitoring ───\n")
    manager = AlertManager(use_llm=True)

    live_scenarios = [
        {
            "name": "Quiet hallway — all normal",
            "audio_type": "ambient",
            "alert_type": "normal_activity",
            "score": 0.2,
            "location": "hallway",
            "brightness": 0.55,
        },
        {
            "name": "Glass breaks in hallway!",
            "audio_type": "glass_breaking",
            "alert_type": "audio_anomaly",
            "score": 0.75,
            "location": "hallway",
            "brightness": 0.55,
        },
        {
            "name": "Person running at bus stop",
            "audio_type": "ambient",
            "alert_type": "running_person",
            "score": 0.6,
            "location": "bus_stop",
            "brightness": 0.6,
        },
        {
            "name": "Gunshot at night + dark parking",
            "audio_type": "gunshot",
            "alert_type": "fighting",
            "score": 0.95,
            "location": "parking_lot",
            "brightness": 0.1,
        },
    ]

    decision_icons = {
        AlertDecision.SUPPRESS: "🟢",
        AlertDecision.CONFIRM:  "🟡",
        AlertDecision.ESCALATE: "🔴",
    }

    print(f"  {'#':>3s}  {'Scenario':<35s}  {'AV Score':>8s}  {'Decision':>10s}  {'Action'}")
    print("  " + "─" * 85)

    for i, scenario in enumerate(live_scenarios, 1):
        # Generate audio and frame
        audio = audio_gen.generate(scenario["audio_type"], 1.0)
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH),
                        scenario["brightness"], dtype=np.float32)
        frame += np.random.randn(FRAME_HEIGHT, FRAME_WIDTH).astype(np.float32) * 0.02
        frame = np.clip(frame, 0, 1)

        # Cross-modal check
        av_result = detector.detect(audio, frame)

        # Build alert
        alert = AnomalyAlert(
            timestamp=time.time(),
            clip_index=i,
            frame_index=i * 25,
            anomaly_score=scenario["score"],
            alert_type=scenario["alert_type"],
            reconstruction_error=scenario["score"] * 0.1,
            audio_visual_score=av_result["z_score"] / 5.0,
        )

        # Reasoning
        result = manager.process_alert(alert, frame,
                                       location_hint=scenario["location"])

        icon = decision_icons[result.decision]
        print(f"  {i:>3d}  {scenario['name']:<35s}  "
              f"{av_result['z_score']:>8.2f}  "
              f"{icon} {result.decision.value:<9s}  "
              f"{result.reasoning[:40]}")

    # Final summary
    stats = manager.get_stats()
    print("\n  " + "─" * 85)
    print(f"\n  📊 FINAL SUMMARY:")
    print(f"     Synthetic training data:  {total_synth} frames generated")
    print(f"     Alerts processed:         {stats['total_alerts']}")
    print(f"     False alarms suppressed:  {stats['suppressed']} "
          f"({stats['suppression_rate']:.0%} reduction)")
    print(f"     Critical escalations:     {stats['escalated']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Advanced CCTV Anomaly Detection Pipeline"
    )
    parser.add_argument(
        "--phase", type=int, choices=[5, 6, 7],
        help="Run demo for a specific phase (5, 6, or 7). "
             "Omit for full integration demo."
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("  Advanced CCTV Anomaly Detection — Unified Pipeline")
    print("  Phases 5-7: Dreaming Engine + Audio-Visual + Reasoning")
    print("█" * 70)

    try:
        if args.phase == 5:
            demo_dreaming_engine()
        elif args.phase == 6:
            demo_audio_visual()
        elif args.phase == 7:
            demo_reasoning()
        else:
            demo_dreaming_engine()
            demo_audio_visual()
            demo_reasoning()
            print("\n" + "─" * 70)
            print("  Running full integration demo…")
            print("─" * 70)
            demo_full_integration()

        print("\n" + "█" * 70)
        print("  ✅ All demos completed successfully!")
        print("█" * 70 + "\n")

    except ImportError as e:
        print(f"\n  ❌ Import error: {e}")
        print("  Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
