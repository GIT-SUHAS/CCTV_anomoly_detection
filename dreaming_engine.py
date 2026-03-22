"""
Digital Twin "Dreaming" Engine — Phase 5
=========================================
Generative Data Augmentation for CCTV Anomaly Detection.

Solves the data scarcity problem by generating synthetic anomaly footage
on real CCTV backgrounds. Two-tier approach:
  • Tier 1 (GPU): Stable Diffusion img2img for photorealistic anomaly synthesis
  • Tier 2 (CPU): OpenCV-based lightweight composite augmentations

Usage:
    python dreaming_engine.py          # Run self-contained demo
    from dreaming_engine import DreamingEngine
"""

import os
import math
import random
import warnings
import numpy as np
import cv2

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
SYNTHETIC_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_anomalies")


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class BackgroundExtractor:
    """
    Extracts a clean background plate from a video clip using temporal median.
    Works best on static-camera CCTV footage where the background is mostly
    consistent across frames.
    """

    @staticmethod
    def extract(clip_frames: list[np.ndarray], method: str = "median") -> np.ndarray:
        """
        Extract a clean background from a sequence of frames.

        Args:
            clip_frames: List of grayscale frames (H, W), float32 [0,1].
            method: 'median' (robust) or 'mean' (faster).

        Returns:
            Background image (H, W), float32 [0,1].
        """
        if not clip_frames:
            raise ValueError("clip_frames is empty")

        stack = np.stack(clip_frames, axis=0)  # (T, H, W)

        if method == "median":
            background = np.median(stack, axis=0).astype(np.float32)
        elif method == "mean":
            background = np.mean(stack, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unknown method: {method}")

        return background

    @staticmethod
    def extract_adaptive(clip_frames: list[np.ndarray], learning_rate: float = 0.01) -> np.ndarray:
        """
        Extract background using MOG2 adaptive subtraction (returns the
        learned background model rather than foreground masks).
        """
        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=len(clip_frames), varThreshold=16, detectShadows=False
        )
        for frame in clip_frames:
            frame_uint8 = (frame * 255).astype(np.uint8)
            bg_sub.apply(frame_uint8, learningRate=learning_rate)

        background = bg_sub.getBackgroundImage()
        if background is None:
            # Fallback to median
            return BackgroundExtractor.extract(clip_frames, method="median")
        return background.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE SYNTHETIC AUGMENTOR (CPU — Tier 2)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleSyntheticAugmentor:
    """
    Lightweight CPU-based anomaly synthesizer.
    Composites synthetic visual anomalies onto real backgrounds using OpenCV.
    Supports: fire/glow, fallen person, moving intruder, object left behind.
    """

    ANOMALY_TYPES = ["fire", "fallen_person", "intruder", "abandoned_object"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    # ── Fire / Glow Effect ────────────────────────────────────────────────

    def _generate_fire(self, background: np.ndarray,
                       intensity: float = 0.7) -> np.ndarray:
        """
        Overlay a procedural fire/glow effect at a random location.
        Creates flickering orange-yellow glow on grayscale backgrounds.
        """
        h, w = background.shape[:2]
        frame = background.copy()

        # Random fire location (bottom 60% of frame — floor level)
        cx = self.rng.randint(w // 4, 3 * w // 4)
        cy = self.rng.randint(h // 2, int(h * 0.85))
        radius = self.rng.randint(15, 45)

        # Create fire mask with radial gradient
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2).astype(np.float32)
        fire_mask = np.clip(1.0 - dist / radius, 0, 1) ** 2

        # Add turbulence noise
        noise = self.np_rng.randn(h, w).astype(np.float32) * 0.15
        fire_mask = np.clip(fire_mask + noise * fire_mask, 0, 1)

        # Apply bright glow (on grayscale: fire = bright spot)
        fire_color = intensity * fire_mask
        frame = np.clip(frame + fire_color, 0, 1)

        # Add flickering halo
        halo_radius = radius * 2.5
        halo_mask = np.clip(1.0 - dist / halo_radius, 0, 1) ** 1.5
        frame = np.clip(frame + 0.15 * halo_mask * intensity, 0, 1)

        return frame

    # ── Fallen Person ─────────────────────────────────────────────────────

    def _generate_fallen_person(self, background: np.ndarray) -> np.ndarray:
        """
        Overlay a synthetic fallen person (dark horizontal blob on ground).
        """
        h, w = background.shape[:2]
        frame = background.copy()

        # Person position (lower half, lying down)
        cx = self.rng.randint(w // 4, 3 * w // 4)
        cy = self.rng.randint(int(h * 0.6), int(h * 0.82))
        body_w = self.rng.randint(40, 70)
        body_h = self.rng.randint(12, 22)

        # Body (horizontal ellipse — lying down)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (body_w, body_h), 0, 0, 360, 1.0, -1)

        # Head (circle at one end)
        head_x = cx - body_w + 5
        head_r = body_h + 2
        cv2.circle(mask, (head_x, cy - 3), head_r, 1.0, -1)

        # Smooth edges
        mask = cv2.GaussianBlur(mask, (7, 7), 2.0)

        # Person is darker than background
        person_shade = self.np_rng.uniform(0.1, 0.3)
        frame = frame * (1 - mask) + person_shade * mask

        return frame.astype(np.float32)

    # ── Intruder (Moving Shadow) ──────────────────────────────────────────

    def _generate_intruder(self, background: np.ndarray) -> np.ndarray:
        """
        Overlay a standing person silhouette with slight motion blur.
        """
        h, w = background.shape[:2]
        frame = background.copy()

        # Person position
        cx = self.rng.randint(w // 5, 4 * w // 5)
        foot_y = self.rng.randint(int(h * 0.65), int(h * 0.88))
        person_h = self.rng.randint(50, 90)
        person_w = self.rng.randint(18, 30)

        # Body silhouette
        mask = np.zeros((h, w), dtype=np.float32)

        # Torso (rectangle)
        top_y = foot_y - person_h
        cv2.rectangle(mask,
                      (cx - person_w // 2, top_y + person_w),
                      (cx + person_w // 2, foot_y),
                      1.0, -1)

        # Head (circle)
        cv2.circle(mask, (cx, top_y + person_w // 2), person_w // 2, 1.0, -1)

        # Legs (two thin rectangles)
        leg_w = person_w // 4
        cv2.rectangle(mask,
                      (cx - leg_w - 2, foot_y - person_h // 4),
                      (cx - 2, foot_y),
                      1.0, -1)
        cv2.rectangle(mask,
                      (cx + 2, foot_y - person_h // 4),
                      (cx + leg_w + 2, foot_y),
                      1.0, -1)

        # Motion blur effect
        motion_kernel = np.zeros((1, 7), dtype=np.float32)
        motion_kernel[0, :] = 1.0 / 7.0
        mask = cv2.filter2D(mask, -1, motion_kernel)

        # Smooth
        mask = cv2.GaussianBlur(mask, (5, 5), 1.5)
        mask = np.clip(mask, 0, 1)

        # Person shade (darker than background)
        person_shade = self.np_rng.uniform(0.08, 0.25)
        frame = frame * (1 - mask * 0.85) + person_shade * mask * 0.85

        # Add shadow on ground
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(shadow_mask, (cx + 10, foot_y + 3),
                    (person_w, 5), 0, 0, 360, 0.4, -1)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (9, 9), 3)
        frame = np.clip(frame - shadow_mask * 0.2, 0, 1)

        return frame.astype(np.float32)

    # ── Abandoned Object ──────────────────────────────────────────────────

    def _generate_abandoned_object(self, background: np.ndarray) -> np.ndarray:
        """
        Place a suspicious object (bag/box shaped) in the scene.
        """
        h, w = background.shape[:2]
        frame = background.copy()

        # Object position (middle-lower area)
        cx = self.rng.randint(w // 4, 3 * w // 4)
        cy = self.rng.randint(int(h * 0.55), int(h * 0.8))
        obj_w = self.rng.randint(15, 30)
        obj_h = self.rng.randint(12, 25)

        # Box/bag shape
        mask = np.zeros((h, w), dtype=np.float32)
        # Main body
        cv2.rectangle(mask,
                      (cx - obj_w, cy - obj_h),
                      (cx + obj_w, cy + obj_h),
                      1.0, -1)
        # Handle/strap
        cv2.ellipse(mask, (cx, cy - obj_h),
                    (obj_w // 2, obj_h // 3), 0, 180, 360, 1.0, 2)

        mask = cv2.GaussianBlur(mask, (5, 5), 1.5)
        mask = np.clip(mask, 0, 1)

        obj_shade = self.np_rng.uniform(0.15, 0.35)
        frame = frame * (1 - mask * 0.9) + obj_shade * mask * 0.9

        return frame.astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────

    def generate(self, background: np.ndarray,
                 anomaly_type: str | None = None,
                 num_frames: int = 10) -> list[np.ndarray]:
        """
        Generate a sequence of synthetic anomaly frames.

        Args:
            background: Clean background frame (H, W), float32 [0,1].
            anomaly_type: One of ANOMALY_TYPES, or None for random.
            num_frames: Number of frames to generate (with slight variation).

        Returns:
            List of anomaly frames (H, W), float32 [0,1].
        """
        if anomaly_type is None:
            anomaly_type = self.rng.choice(self.ANOMALY_TYPES)

        generators = {
            "fire": self._generate_fire,
            "fallen_person": self._generate_fallen_person,
            "intruder": self._generate_intruder,
            "abandoned_object": self._generate_abandoned_object,
        }

        if anomaly_type not in generators:
            raise ValueError(
                f"Unknown anomaly_type '{anomaly_type}'. "
                f"Choose from: {self.ANOMALY_TYPES}"
            )

        gen_fn = generators[anomaly_type]
        frames = []
        for i in range(num_frames):
            # Slight jitter for temporal variation
            bg_jitter = background + self.np_rng.randn(*background.shape).astype(
                np.float32) * 0.005
            bg_jitter = np.clip(bg_jitter, 0, 1)
            frame = gen_fn(bg_jitter)
            frames.append(frame)

        return frames


# ══════════════════════════════════════════════════════════════════════════════
# DIFFUSION-BASED ANOMALY GENERATOR (GPU — Tier 1)
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyDiffusionGenerator:
    """
    Uses Stable Diffusion img2img pipeline to generate photorealistic
    synthetic anomalies on real CCTV backgrounds.

    Requires: torch, diffusers, transformers (GPU recommended).
    Falls back gracefully to SimpleSyntheticAugmentor if unavailable.
    """

    ANOMALY_PROMPTS = {
        "fire": (
            "A realistic fire with flames and smoke in a building corridor, "
            "captured by a CCTV security camera, grayscale surveillance footage"
        ),
        "fallen_person": (
            "A person collapsed on the ground in a hallway, lying motionless, "
            "captured by overhead CCTV security camera, grayscale surveillance footage"
        ),
        "intrusion": (
            "A suspicious person sneaking through a restricted area at night, "
            "captured by CCTV security camera, grayscale surveillance footage"
        ),
        "vandalism": (
            "A person breaking and vandalizing property in a corridor, "
            "captured by CCTV security camera, grayscale surveillance footage"
        ),
        "fight": (
            "Two people fighting aggressively in a hallway, "
            "captured by overhead CCTV security camera, grayscale surveillance footage"
        ),
    }

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str | None = None):
        self.model_id = model_id
        self.pipe = None
        self.available = False
        self.fallback = SimpleSyntheticAugmentor()

        # Attempt to load the diffusion pipeline
        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline

            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            self.device = device
            print(f"[DreamingEngine] Loading Stable Diffusion on {device}…")

            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self.pipe = self.pipe.to(device)

            # Memory optimisation
            if device == "cuda":
                self.pipe.enable_attention_slicing()

            self.available = True
            print("[DreamingEngine] ✓ Stable Diffusion loaded successfully.")

        except ImportError:
            print("[DreamingEngine] ⚠ diffusers not installed. Using CPU fallback.")
        except Exception as e:
            print(f"[DreamingEngine] ⚠ Could not load diffusion model: {e}. Using CPU fallback.")

    def generate(self, background: np.ndarray,
                 anomaly_type: str = "fire",
                 strength: float = 0.6,
                 num_frames: int = 5,
                 guidance_scale: float = 7.5) -> list[np.ndarray]:
        """
        Generate synthetic anomaly frames using Stable Diffusion.

        Args:
            background: Background frame (H, W), float32 [0,1].
            anomaly_type: Key from ANOMALY_PROMPTS.
            strength: How much to modify the image (0=no change, 1=full generation).
            num_frames: Number of frames to generate.
            guidance_scale: Classifier-free guidance scale.

        Returns:
            List of anomaly frames (H, W), float32 [0,1].
        """
        if not self.available:
            print("[DreamingEngine] Falling back to SimpleSyntheticAugmentor.")
            # Map diffusion anomaly types to simple types
            type_map = {
                "fire": "fire",
                "fallen_person": "fallen_person",
                "intrusion": "intruder",
                "vandalism": "intruder",
                "fight": "intruder",
            }
            simple_type = type_map.get(anomaly_type, "fire")
            return self.fallback.generate(background, simple_type, num_frames)

        from PIL import Image
        import torch

        prompt = self.ANOMALY_PROMPTS.get(
            anomaly_type,
            f"A {anomaly_type} scene in a building, CCTV camera footage"
        )

        # Convert grayscale to RGB PIL image
        bg_uint8 = (background * 255).astype(np.uint8)
        bg_rgb = cv2.cvtColor(bg_uint8, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(bg_rgb).resize((512, 512))

        frames = []
        for i in range(num_frames):
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    image=pil_image,
                    strength=strength + random.uniform(-0.05, 0.05),
                    guidance_scale=guidance_scale,
                    num_inference_steps=30,
                )
            gen_image = result.images[0]
            # Convert back to grayscale numpy
            gen_np = np.array(gen_image.resize((FRAME_WIDTH, FRAME_HEIGHT)))
            gen_gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            frames.append(gen_gray)

        return frames


# ══════════════════════════════════════════════════════════════════════════════
# DREAMING ENGINE — Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class DreamingEngine:
    """
    The complete 'Dreaming Engine' that creates synthetic training data.

    Takes normal CCTV clips → extracts clean backgrounds → synthesizes anomalies
    → produces augmented training sets ready for the ConvLSTM Autoencoder.

    "I solved the data scarcity problem by building a 'Dreaming Engine' that
     creates its own training data, allowing the system to learn rare threats
     that have never actually happened in that building."
    """

    def __init__(self, use_diffusion: bool = False, seed: int = 42):
        """
        Args:
            use_diffusion: If True, attempt to use Stable Diffusion (Tier 1).
                           If False or unavailable, use OpenCV augmentation (Tier 2).
            seed: Random seed for reproducibility.
        """
        self.bg_extractor = BackgroundExtractor()
        self.simple_augmentor = SimpleSyntheticAugmentor(seed=seed)

        if use_diffusion:
            self.diffusion_gen = AnomalyDiffusionGenerator()
        else:
            self.diffusion_gen = None

        self.generated_count = 0

    def extract_background(self, clip_frames: list[np.ndarray],
                           method: str = "median") -> np.ndarray:
        """Extract clean background from a normal clip."""
        return self.bg_extractor.extract(clip_frames, method=method)

    def dream_anomalies(self, background: np.ndarray,
                        anomaly_types: list[str] | None = None,
                        frames_per_type: int = 10,
                        use_diffusion: bool = False) -> dict[str, list[np.ndarray]]:
        """
        Generate synthetic anomaly frames for multiple anomaly types.

        Args:
            background: Clean background frame (H, W), float32 [0,1].
            anomaly_types: List of anomaly types to generate.
                           Defaults to all available from SimpleSyntheticAugmentor.
            frames_per_type: Number of frames to generate per anomaly type.
            use_diffusion: Use Stable Diffusion if available.

        Returns:
            Dict mapping anomaly_type → list of synthetic frames.
        """
        if anomaly_types is None:
            anomaly_types = SimpleSyntheticAugmentor.ANOMALY_TYPES

        results = {}

        for atype in anomaly_types:
            if use_diffusion and self.diffusion_gen and self.diffusion_gen.available:
                frames = self.diffusion_gen.generate(
                    background, anomaly_type=atype, num_frames=frames_per_type
                )
            else:
                frames = self.simple_augmentor.generate(
                    background, anomaly_type=atype, num_frames=frames_per_type
                )

            results[atype] = frames
            self.generated_count += len(frames)

        return results

    def augment_training_set(self, normal_clips: list[list[np.ndarray]],
                             anomaly_types: list[str] | None = None,
                             frames_per_type: int = 10,
                             clips_to_use: int = 3) -> dict:
        """
        Full pipeline: take normal training clips, generate synthetic anomalies,
        and return augmented dataset ready for training.

        Args:
            normal_clips: List of normal video clips (list of frames each).
            anomaly_types: Types of anomalies to generate.
            frames_per_type: Frames per anomaly type per background.
            clips_to_use: Number of clips to extract backgrounds from.

        Returns:
            Dict with 'backgrounds', 'synthetic_anomalies', 'all_frames', 'labels'.
        """
        # Select clips for background extraction
        selected = normal_clips[:clips_to_use]
        backgrounds = []
        all_synthetic = {}

        for i, clip in enumerate(selected):
            print(f"  [DreamingEngine] Extracting background from clip {i + 1}/{len(selected)}…")
            bg = self.extract_background(clip)
            backgrounds.append(bg)

            print(f"  [DreamingEngine] Generating anomalies on background {i + 1}…")
            anomalies = self.dream_anomalies(
                bg,
                anomaly_types=anomaly_types,
                frames_per_type=frames_per_type,
            )

            for atype, frames in anomalies.items():
                if atype not in all_synthetic:
                    all_synthetic[atype] = []
                all_synthetic[atype].extend(frames)

        # Compile augmented dataset
        all_frames = []
        labels = []

        # Normal frames
        for clip in normal_clips:
            for frame in clip:
                all_frames.append(frame)
                labels.append(("normal", 0))

        # Synthetic anomaly frames
        for atype, frames in all_synthetic.items():
            for frame in frames:
                all_frames.append(frame)
                labels.append((atype, 1))

        print(f"  [DreamingEngine] ✓ Augmented dataset: "
              f"{len(all_frames)} total frames "
              f"({sum(1 for _, l in labels if l == 0)} normal + "
              f"{sum(1 for _, l in labels if l == 1)} synthetic anomaly)")

        return {
            "backgrounds": backgrounds,
            "synthetic_anomalies": all_synthetic,
            "all_frames": all_frames,
            "labels": labels,
        }

    def save_samples(self, synthetic_anomalies: dict[str, list[np.ndarray]],
                     output_dir: str = SYNTHETIC_OUTPUT_DIR,
                     max_per_type: int = 5):
        """Save sample synthetic frames to disk for inspection."""
        os.makedirs(output_dir, exist_ok=True)
        saved = 0

        for atype, frames in synthetic_anomalies.items():
            type_dir = os.path.join(output_dir, atype)
            os.makedirs(type_dir, exist_ok=True)
            for i, frame in enumerate(frames[:max_per_type]):
                fpath = os.path.join(type_dir, f"synthetic_{i:03d}.png")
                cv2.imwrite(fpath, (frame * 255).astype(np.uint8))
                saved += 1

        print(f"  [DreamingEngine] Saved {saved} sample frames to {output_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST / DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Digital Twin 'Dreaming' Engine — Demo")
    print("=" * 70)

    # Create a synthetic "hallway" background for demonstration
    print("\n[Demo] Creating synthetic hallway background…")
    bg = np.ones((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32) * 0.7

    # Add floor
    bg[FRAME_HEIGHT // 2:, :] = 0.55

    # Add walls (perspective lines)
    for y in range(FRAME_HEIGHT):
        vanish_x = FRAME_WIDTH // 2
        left_wall = int(vanish_x - (vanish_x * (y / FRAME_HEIGHT)))
        right_wall = int(vanish_x + (vanish_x * (y / FRAME_HEIGHT)))
        if left_wall > 0:
            bg[y, :left_wall] = 0.45
        if right_wall < FRAME_WIDTH:
            bg[y, right_wall:] = 0.45

    # Add some texture/noise
    bg += np.random.randn(FRAME_HEIGHT, FRAME_WIDTH).astype(np.float32) * 0.02
    bg = np.clip(bg, 0, 1)

    # Simulate a normal clip (variations on the background)
    print("[Demo] Simulating normal video clip…")
    normal_clip = []
    for i in range(30):
        frame = bg + np.random.randn(FRAME_HEIGHT, FRAME_WIDTH).astype(np.float32) * 0.005
        normal_clip.append(np.clip(frame, 0, 1))

    # Initialize the Dreaming Engine (CPU mode)
    print("[Demo] Initializing DreamingEngine (CPU mode)…\n")
    engine = DreamingEngine(use_diffusion=False, seed=42)

    # Extract background
    extracted_bg = engine.extract_background(normal_clip)
    print(f"[Demo] Background extracted: shape={extracted_bg.shape}, "
          f"mean={extracted_bg.mean():.3f}")

    # Generate anomalies
    print("\n[Demo] Generating synthetic anomalies…")
    anomalies = engine.dream_anomalies(
        extracted_bg,
        anomaly_types=["fire", "fallen_person", "intruder", "abandoned_object"],
        frames_per_type=5,
    )

    for atype, frames in anomalies.items():
        print(f"  • {atype}: {len(frames)} frames generated "
              f"(mean brightness: {np.mean([f.mean() for f in frames]):.3f})")

    # Save samples
    print("\n[Demo] Saving sample frames…")
    engine.save_samples(anomalies, max_per_type=3)

    # Full augmentation pipeline
    print("\n[Demo] Running full augmentation pipeline…")
    result = engine.augment_training_set(
        normal_clips=[normal_clip],
        frames_per_type=5,
        clips_to_use=1,
    )

    print(f"\n[Demo] ✓ Total generated frames: {engine.generated_count}")
    print("=" * 70)
    print("Digital Twin 'Dreaming' Engine — Demo Complete")
    print("=" * 70)
