"""
"Common Sense" Reasoning Engine — Phase 7
==========================================
VLM/LLM-based Logic for contextual alert filtering.

Current AIs are "dumb" — they might flag a person running to catch a bus 
as a criminal. This module adds a Reasoning Layer that applies contextual
understanding to filter false alarms.

Workflow:
    1. Vision Model flags an event (e.g., "Running Person").
    2. SceneDescriber provides structured context.
    3. ReasoningEngine checks context using LLM.
    4. AlertManager makes final decision: suppress, confirm, or escalate.

Usage:
    python reasoning_engine.py         # Run self-contained demo
    from reasoning_engine import AlertManager
"""

import os
import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np
import cv2
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"
FRAME_HEIGHT = 224
FRAME_WIDTH = 224

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class AlertDecision(Enum):
    SUPPRESS = "suppress"
    CONFIRM = "confirm"
    ESCALATE = "escalate"


@dataclass
class AnomalyAlert:
    """Raw anomaly detection result from the vision model."""
    timestamp: float
    clip_index: int
    frame_index: int
    anomaly_score: float
    alert_type: str                  # e.g., "running_person", "crowd_anomaly"
    reconstruction_error: float = 0.0
    audio_visual_score: float = 0.0  # From cross-modal module (if available)


@dataclass
class SceneContext:
    """Structured scene description from the vision model."""
    location_type: str          # e.g., "hallway", "parking_lot", "bus_stop"
    time_of_day: str            # "day", "night", "dusk"
    detected_objects: list[str] = field(default_factory=list)
    detected_actions: list[str] = field(default_factory=list)
    crowd_density: str = "low"  # "empty", "low", "moderate", "high"
    lighting: str = "normal"    # "normal", "dim", "dark", "bright"
    weather: str = "unknown"    # "clear", "rain", "fog", "unknown"
    additional_context: str = ""


@dataclass
class AlertResult:
    """Final alert decision after reasoning."""
    original_alert: AnomalyAlert
    scene_context: SceneContext
    decision: AlertDecision
    confidence: float
    reasoning: str
    llm_used: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# SCENE DESCRIBER (Vision → Context)
# ══════════════════════════════════════════════════════════════════════════════

class SceneDescriber:
    """
    Analyzes video frames to produce structured scene descriptions.
    Uses ResNet features + heuristic rules for scene classification.
    
    In a production system, this would use a full VLM (e.g., LLaVA, GPT-4V),
    but we implement a lightweight version using CNN features + rules.
    """

    # Scene classification labels (simplified)
    SCENE_TYPES = [
        "hallway", "parking_lot", "entrance", "stairwell",
        "elevator", "office", "lobby", "outdoor_path",
        "bus_stop", "intersection", "warehouse", "retail_store",
    ]

    # Action detection based on motion patterns
    ACTION_LABELS = [
        "walking", "running", "standing", "sitting",
        "falling", "fighting", "loitering", "carrying_object",
    ]

    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load pre-trained ResNet for scene feature extraction."""
        import torchvision.models as models
        import torchvision.transforms as transforms

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.eval()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract 512-d features from a frame."""
        frame_uint8 = (frame * 255).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2RGB)
        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        feat = self.backbone(tensor).squeeze().cpu().numpy()
        return feat

    def _estimate_crowd_density(self, frame: np.ndarray) -> str:
        """Estimate crowd density from foreground pixel ratio."""
        frame_uint8 = (frame * 255).astype(np.uint8)
        # Simple threshold-based foreground detection
        _, binary = cv2.threshold(frame_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_ratio = binary.sum() / (255.0 * binary.size)

        if fg_ratio < 0.02:
            return "empty"
        elif fg_ratio < 0.1:
            return "low"
        elif fg_ratio < 0.3:
            return "moderate"
        else:
            return "high"

    def _estimate_lighting(self, frame: np.ndarray) -> str:
        """Estimate lighting condition from frame brightness."""
        mean_brightness = frame.mean()
        if mean_brightness < 0.15:
            return "dark"
        elif mean_brightness < 0.35:
            return "dim"
        elif mean_brightness > 0.8:
            return "bright"
        else:
            return "normal"

    def _detect_motion(self, prev_frame: np.ndarray | None,
                       curr_frame: np.ndarray) -> list[str]:
        """Detect actions based on optical flow motion patterns."""
        if prev_frame is None:
            return ["standing"]

        prev_u8 = (prev_frame * 255).astype(np.uint8)
        curr_u8 = (curr_frame * 255).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(
            prev_u8, curr_u8, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_motion = mag.mean()
        max_motion = mag.max()

        actions = []
        if mean_motion < 0.5:
            actions.append("standing")
        elif mean_motion < 2.0:
            actions.append("walking")
        elif mean_motion < 5.0:
            actions.append("running")
        else:
            actions.append("fast_movement")

        # Detect sudden downward motion (potential fall)
        bottom_half_motion = mag[mag.shape[0] // 2:, :].mean()
        top_half_motion = mag[:mag.shape[0] // 2, :].mean()
        if bottom_half_motion > top_half_motion * 2 and mean_motion > 2:
            actions.append("falling")

        return actions

    def describe(self, frame: np.ndarray,
                 prev_frame: np.ndarray | None = None,
                 location_hint: str = "hallway") -> SceneContext:
        """
        Generate a structured scene description from a video frame.

        Args:
            frame: Current grayscale frame (H, W), float32 [0,1].
            prev_frame: Previous frame for motion detection (optional).
            location_hint: Hint about location type (from camera metadata).

        Returns:
            SceneContext with all extracted information.
        """
        features = self._extract_features(frame)
        crowd = self._estimate_crowd_density(frame)
        lighting = self._estimate_lighting(frame)
        actions = self._detect_motion(prev_frame, frame)

        # Detected objects (simplified — based on feature activations)
        objects = []
        if crowd != "empty":
            objects.append("person")
        if features.mean() > 0.5:
            objects.append("structure")

        # Time of day estimation from lighting
        time_of_day = "day"
        if lighting in ("dark", "dim"):
            time_of_day = "night"

        context = SceneContext(
            location_type=location_hint,
            time_of_day=time_of_day,
            detected_objects=objects,
            detected_actions=actions,
            crowd_density=crowd,
            lighting=lighting,
            additional_context=f"Feature mean: {features.mean():.3f}, "
                               f"Motion detected: {', '.join(actions)}"
        )

        return context


# ══════════════════════════════════════════════════════════════════════════════
# REASONING ENGINE (LLM-based)
# ══════════════════════════════════════════════════════════════════════════════

class ReasoningEngine:
    """
    Uses an LLM (GPT-4o-mini) to apply common-sense reasoning to anomaly
    alerts, filtering out false alarms based on context.

    "My system doesn't just detect movement; it applies 'Common Sense 
     Reasoning' to filter out false alarms that usually plague security guards."
    """

    SYSTEM_PROMPT = """You are an intelligent CCTV security analysis system. Your job is to apply
common-sense reasoning to anomaly alerts from a computer vision system.

Given a scene description and an anomaly alert, you must decide:
- SUPPRESS: This is a false alarm. The activity is normal given the context.
- CONFIRM: This is a genuine anomaly that should alert a security guard.
- ESCALATE: This is a critical situation requiring immediate response.

Respond ONLY in this exact JSON format:
{
    "decision": "suppress" | "confirm" | "escalate",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your reasoning"
}"""

    def __init__(self, api_key: str = OPENAI_API_KEY,
                 model: str = LLM_MODEL):
        self.api_key = api_key
        self.model = model
        self.available = bool(api_key)

        if self.available:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                print("[ReasoningEngine] ✓ OpenAI API connected.")
            except ImportError:
                print("[ReasoningEngine] ⚠ openai package not installed. Using mock engine.")
                self.available = False
        else:
            print("[ReasoningEngine] ⚠ No OPENAI_API_KEY set. Using mock engine.")

    def reason(self, alert: AnomalyAlert,
               context: SceneContext) -> AlertResult:
        """
        Apply LLM reasoning to an anomaly alert given scene context.

        Args:
            alert: Raw anomaly detection result.
            context: Structured scene description.

        Returns:
            AlertResult with decision, confidence, and reasoning.
        """
        if not self.available:
            return self._mock_reason(alert, context)

        user_prompt = self._build_prompt(alert, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=200,
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                result_json = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                match = re.search(r'\{[^}]+\}', result_text)
                if match:
                    result_json = json.loads(match.group())
                else:
                    result_json = {
                        "decision": "confirm",
                        "confidence": 0.5,
                        "reasoning": f"Could not parse LLM response: {result_text[:100]}",
                    }

            decision = AlertDecision(result_json.get("decision", "confirm"))
            confidence = float(result_json.get("confidence", 0.5))
            reasoning = result_json.get("reasoning", "No reasoning provided")

            return AlertResult(
                original_alert=alert,
                scene_context=context,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                llm_used=True,
            )

        except Exception as e:
            print(f"[ReasoningEngine] API error: {e}. Falling back to mock.")
            return self._mock_reason(alert, context)

    def _build_prompt(self, alert: AnomalyAlert,
                      context: SceneContext) -> str:
        """Build the prompt for the LLM."""
        return f"""ANOMALY ALERT:
- Type: {alert.alert_type}
- Severity Score: {alert.anomaly_score:.2f}
- Reconstruction Error: {alert.reconstruction_error:.4f}
- Audio-Visual Mismatch Score: {alert.audio_visual_score:.2f}

SCENE CONTEXT:
- Location: {context.location_type}
- Time of Day: {context.time_of_day}
- Detected Objects: {', '.join(context.detected_objects) if context.detected_objects else 'none'}
- Detected Actions: {', '.join(context.detected_actions) if context.detected_actions else 'none'}
- Crowd Density: {context.crowd_density}
- Lighting: {context.lighting}
- Weather: {context.weather}
- Additional: {context.additional_context}

Based on the above, should this alert be SUPPRESSED, CONFIRMED, or ESCALATED?"""

    def _mock_reason(self, alert: AnomalyAlert,
                     context: SceneContext) -> AlertResult:
        """
        Rule-based fallback reasoning for when LLM is not available.
        Applies common-sense heuristics.
        Rules are ordered so that alert_type (from the vision model) takes
        priority over detected_actions (from optical flow on the frame).
        """
        decision = AlertDecision.CONFIRM
        confidence = 0.5
        reasoning = ""

        alert_type = alert.alert_type.lower()
        actions = [a.lower() for a in context.detected_actions]
        location = context.location_type.lower()

        # ── Rule 1: Fighting → always escalate (check alert_type first) ──
        if "fighting" in alert_type or "fight" in alert_type or "fast_movement" in actions:
            decision = AlertDecision.ESCALATE
            confidence = 0.85
            reasoning = (
                "Aggressive physical activity detected. "
                "Potential altercation requiring immediate intervention."
            )

        # ── Rule 2: Running person at bus stop → suppress ──
        elif "running" in alert_type or "running" in actions:
            if location in ("bus_stop", "train_station", "transit"):
                decision = AlertDecision.SUPPRESS
                confidence = 0.85
                reasoning = (
                    "Person is running near a transit stop. "
                    "Likely running to catch a bus/train — not a threat."
                )
            elif context.crowd_density == "high" and context.time_of_day == "day":
                decision = AlertDecision.SUPPRESS
                confidence = 0.7
                reasoning = (
                    "Fast movement in a crowded daytime area. "
                    "Likely someone in a hurry, not suspicious."
                )
            else:
                decision = AlertDecision.CONFIRM
                confidence = 0.6
                reasoning = (
                    "Person running in a non-transit area. "
                    "Warrants investigation."
                )

        # ── Rule 3: Falling detection → escalate if confirmed ──
        elif "falling" in alert_type or "fall" in alert_type or "falling" in actions:
            if context.crowd_density == "empty" and context.lighting == "dark":
                decision = AlertDecision.ESCALATE
                confidence = 0.9
                reasoning = (
                    "Person falling in an empty, poorly lit area. "
                    "Potential medical emergency or assault. Immediate response needed."
                )
            else:
                decision = AlertDecision.CONFIRM
                confidence = 0.7
                reasoning = (
                    "Potential fall detected. Could be a slip or medical event. "
                    "Security should check."
                )

        # ── Rule 4: High anomaly score with audio mismatch → escalate ──
        elif alert.anomaly_score > 0.8 and alert.audio_visual_score > 0.7:
            decision = AlertDecision.ESCALATE
            confidence = 0.8
            reasoning = (
                "High visual anomaly score combined with audio-visual mismatch. "
                "Multiple sensors confirm abnormal activity."
            )

        # ── Rule 5: Loitering → context-dependent ──
        elif "loitering" in alert_type:
            if context.time_of_day == "night" and context.crowd_density in ("empty", "low"):
                decision = AlertDecision.CONFIRM
                confidence = 0.65
                reasoning = (
                    "Person standing in a low-traffic area at night. "
                    "Could be suspicious behavior."
                )
            elif context.time_of_day == "day" and location in ("lobby", "entrance", "bus_stop"):
                decision = AlertDecision.SUPPRESS
                confidence = 0.8
                reasoning = (
                    "Person waiting in a normal waiting area during daytime. "
                    "Not suspicious."
                )
            else:
                decision = AlertDecision.SUPPRESS
                confidence = 0.6
                reasoning = "Standing/loitering with no additional suspicious indicators."

        # ── Rule 6: Moderate anomaly with context ──
        elif alert.anomaly_score > 0.5:
            if context.lighting in ("dark", "dim"):
                decision = AlertDecision.CONFIRM
                confidence = 0.6
                reasoning = (
                    "Moderate anomaly in poor lighting conditions. "
                    "Worth investigating."
                )
            else:
                decision = AlertDecision.CONFIRM
                confidence = 0.5
                reasoning = (
                    "Moderate anomaly detected. "
                    "Possible but not certain threat."
                )

        # ── Default: Low-confidence confirm ──
        else:
            decision = AlertDecision.CONFIRM
            confidence = 0.4
            reasoning = "Generic anomaly with insufficient context for suppression."

        return AlertResult(
            original_alert=alert,
            scene_context=context,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            llm_used=False,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ALERT MANAGER — Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class AlertManager:
    """
    End-to-end alert pipeline:
    
    Raw Anomaly → Scene Description → LLM Reasoning → Final Decision
    
    Manages alert history, suppression cooldowns, and escalation routing.
    """

    def __init__(self, use_llm: bool = True,
                 device: torch.device = DEVICE):
        self.device = device
        self.scene_describer = SceneDescriber(device=device)

        if use_llm and OPENAI_API_KEY:
            self.reasoning = ReasoningEngine()
        else:
            self.reasoning = ReasoningEngine(api_key="")  # Forces mock

        self.alert_history: list[AlertResult] = []
        self.suppression_count = 0
        self.confirmation_count = 0
        self.escalation_count = 0

    def process_alert(self, alert: AnomalyAlert,
                      frame: np.ndarray,
                      prev_frame: np.ndarray | None = None,
                      location_hint: str = "hallway") -> AlertResult:
        """
        Process a single anomaly alert through the full reasoning pipeline.

        Args:
            alert: Raw anomaly from the vision model.
            frame: Current video frame (H, W), float32 [0,1].
            prev_frame: Previous frame for motion analysis.
            location_hint: Camera location metadata.

        Returns:
            AlertResult with final decision.
        """
        # Step 1: Describe the scene
        context = self.scene_describer.describe(frame, prev_frame, location_hint)

        # Step 2: Apply reasoning
        result = self.reasoning.reason(alert, context)

        # Step 3: Track stats
        self.alert_history.append(result)
        if result.decision == AlertDecision.SUPPRESS:
            self.suppression_count += 1
        elif result.decision == AlertDecision.CONFIRM:
            self.confirmation_count += 1
        elif result.decision == AlertDecision.ESCALATE:
            self.escalation_count += 1

        return result

    def process_batch(self, alerts: list[AnomalyAlert],
                      frames: list[np.ndarray],
                      prev_frames: list[np.ndarray | None] | None = None,
                      location_hint: str = "hallway") -> list[AlertResult]:
        """Process multiple alerts."""
        if prev_frames is None:
            prev_frames = [None] * len(alerts)

        results = []
        for alert, frame, prev_frame in zip(alerts, frames, prev_frames):
            result = self.process_alert(alert, frame, prev_frame, location_hint)
            results.append(result)

        return results

    def get_stats(self) -> dict:
        """Get alert processing statistics."""
        total = len(self.alert_history)
        return {
            "total_alerts": total,
            "suppressed": self.suppression_count,
            "confirmed": self.confirmation_count,
            "escalated": self.escalation_count,
            "suppression_rate": self.suppression_count / max(1, total),
            "escalation_rate": self.escalation_count / max(1, total),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST / DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("'Common Sense' Reasoning Engine — Demo")
    print("=" * 70)

    # Initialize the AlertManager (falls back to mock if no API key)
    print("\n[Demo] Initializing AlertManager…")
    manager = AlertManager(use_llm=True, device=DEVICE)

    # --- Create test scenarios ---
    print("\n[Demo] Creating test scenarios…\n")

    scenarios = [
        {
            "name": "Person running at a bus stop (FALSE ALARM)",
            "alert": AnomalyAlert(
                timestamp=time.time(), clip_index=1, frame_index=50,
                anomaly_score=0.65, alert_type="running_person",
                reconstruction_error=0.045, audio_visual_score=0.1,
            ),
            "location": "bus_stop",
            "brightness": 0.6,
        },
        {
            "name": "Person falling in dark hallway (REAL ALERT)",
            "alert": AnomalyAlert(
                timestamp=time.time(), clip_index=2, frame_index=30,
                anomaly_score=0.85, alert_type="falling_person",
                reconstruction_error=0.08, audio_visual_score=0.6,
            ),
            "location": "hallway",
            "brightness": 0.12,
        },
        {
            "name": "Person standing in lobby during day (FALSE ALARM)",
            "alert": AnomalyAlert(
                timestamp=time.time(), clip_index=3, frame_index=100,
                anomaly_score=0.45, alert_type="loitering",
                reconstruction_error=0.03, audio_visual_score=0.05,
            ),
            "location": "lobby",
            "brightness": 0.55,
        },
        {
            "name": "Fight detected in parking lot (CRITICAL)",
            "alert": AnomalyAlert(
                timestamp=time.time(), clip_index=4, frame_index=20,
                anomaly_score=0.92, alert_type="fighting",
                reconstruction_error=0.12, audio_visual_score=0.85,
            ),
            "location": "parking_lot",
            "brightness": 0.3,
        },
        {
            "name": "Running in empty corridor at night (SUSPICIOUS)",
            "alert": AnomalyAlert(
                timestamp=time.time(), clip_index=5, frame_index=70,
                anomaly_score=0.7, alert_type="running_person",
                reconstruction_error=0.06, audio_visual_score=0.3,
            ),
            "location": "hallway",
            "brightness": 0.18,
        },
    ]

    # Process each scenario
    print("-" * 90)
    print(f"{'Scenario':<50s} {'Decision':<12s} {'Conf':>5s} {'LLM?':>5s}")
    print("-" * 90)

    for scenario in scenarios:
        # Create a synthetic frame with appropriate brightness
        brightness = scenario["brightness"]
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH), brightness, dtype=np.float32)
        frame += np.random.randn(FRAME_HEIGHT, FRAME_WIDTH).astype(np.float32) * 0.02
        frame = np.clip(frame, 0, 1)

        result = manager.process_alert(
            scenario["alert"], frame,
            location_hint=scenario["location"]
        )

        decision_icon = {
            AlertDecision.SUPPRESS: "🟢 SUPPRESS",
            AlertDecision.CONFIRM: "🟡 CONFIRM ",
            AlertDecision.ESCALATE: "🔴 ESCALATE",
        }
        dec_str = decision_icon[result.decision]
        llm_str = "Yes" if result.llm_used else "No"

        print(f"{scenario['name']:<50s} {dec_str:<12s} {result.confidence:>5.2f} {llm_str:>5s}")
        print(f"  └─ Reasoning: {result.reasoning[:80]}")
        print()

    # Print statistics
    stats = manager.get_stats()
    print("-" * 90)
    print("\nALERT STATISTICS:")
    print(f"  Total processed:   {stats['total_alerts']}")
    print(f"  Suppressed:        {stats['suppressed']} ({stats['suppression_rate']:.0%})")
    print(f"  Confirmed:         {stats['confirmed']}")
    print(f"  Escalated:         {stats['escalated']} ({stats['escalation_rate']:.0%})")

    print("\n" + "=" * 70)
    print("'Common Sense' Reasoning Engine — Demo Complete")
    print("=" * 70)
