"""
Audio-Visual Cross-Modal Verification — Phase 6
=================================================
Integrates Soundscape Analysis with video using Contrastive Learning.

If the camera sees a crowd (normal) but the microphone hears a scream or 
glass breaking (anomaly), the system triggers a high-priority alert.

Uses a CLIP-style dual-encoder (Audio + Video) to align the two streams.
If the "visual embedding" and "audio embedding" don't match the "Normal" 
baseline, it's an anomaly.

Usage:
    python audio_visual_fusion.py      # Run self-contained demo
    from audio_visual_fusion import CrossModalAnomalyDetector
"""

import os
import math
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000        # Audio sample rate (Hz)
MEL_BINS = 64              # Number of mel-spectrogram bins
HOP_LENGTH = 512           # Spectrogram hop length
EMBEDDING_DIM = 128        # Shared embedding dimension
AUDIO_SEGMENT_SEC = 1.0    # Audio segment length in seconds
TEMPERATURE = 0.07         # InfoNCE temperature

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC AUDIO GENERATOR (for testing without real audio)
# ══════════════════════════════════════════════════════════════════════════════

class SyntheticAudioGenerator:
    """
    Generates synthetic audio waveforms for testing the cross-modal system
    without needing real audio hardware.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, seed: int = 42):
        self.sr = sample_rate
        self.rng = np.random.RandomState(seed)

    def generate_ambient(self, duration: float = 1.0) -> np.ndarray:
        """Normal ambient noise — low-level background hum."""
        n_samples = int(duration * self.sr)
        # Pink-ish noise (1/f)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.sr)
        freqs[0] = 1  # Avoid division by zero
        spectrum = self.rng.randn(len(freqs)) + 1j * self.rng.randn(len(freqs))
        spectrum /= np.sqrt(freqs)  # 1/f filter
        signal = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
        signal /= np.abs(signal).max() + 1e-8
        signal *= 0.05  # Low volume
        return signal

    def generate_scream(self, duration: float = 1.0) -> np.ndarray:
        """Synthetic scream — high frequency oscillation with harmonics."""
        n_samples = int(duration * self.sr)
        t = np.linspace(0, duration, n_samples, dtype=np.float32)

        # Fundamental + harmonics (high pitch, ~1kHz-3kHz)
        freq_base = self.rng.uniform(800, 1500)
        signal = np.zeros(n_samples, dtype=np.float32)
        for harmonic in range(1, 5):
            freq = freq_base * harmonic
            amplitude = 0.4 / harmonic
            signal += amplitude * np.sin(2 * np.pi * freq * t)

        # Amplitude envelope (attack, sustain, release)
        envelope = np.ones(n_samples, dtype=np.float32)
        attack = int(0.05 * n_samples)
        release = int(0.1 * n_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        signal *= envelope

        # Add vibrato
        vibrato = 0.15 * np.sin(2 * np.pi * 6 * t)
        signal *= (1 + vibrato)

        signal /= np.abs(signal).max() + 1e-8
        return signal

    def generate_glass_breaking(self, duration: float = 1.0) -> np.ndarray:
        """Synthetic glass breaking — broadband impulse with high-freq ringing."""
        n_samples = int(duration * self.sr)
        t = np.linspace(0, duration, n_samples, dtype=np.float32)

        signal = np.zeros(n_samples, dtype=np.float32)

        # Initial impact (broadband impulse)
        impact_len = int(0.02 * self.sr)
        signal[:impact_len] = self.rng.randn(impact_len).astype(np.float32)

        # Tinkling shards (decaying high-freq oscillations)
        n_shards = self.rng.randint(8, 20)
        for _ in range(n_shards):
            start = self.rng.randint(impact_len, max(impact_len + 1, n_samples // 2))
            freq = self.rng.uniform(3000, 8000)
            decay = self.rng.uniform(5, 20)
            shard_len = min(n_samples - start, int(0.3 * self.sr))
            t_shard = np.arange(shard_len, dtype=np.float32) / self.sr
            shard = 0.3 * np.sin(2 * np.pi * freq * t_shard) * np.exp(-decay * t_shard)
            signal[start:start + shard_len] += shard

        signal /= np.abs(signal).max() + 1e-8
        return signal

    def generate_gunshot(self, duration: float = 1.0) -> np.ndarray:
        """Synthetic gunshot — sharp impulse with reverb tail."""
        n_samples = int(duration * self.sr)
        signal = np.zeros(n_samples, dtype=np.float32)

        # Sharp impulse
        impulse_len = int(0.005 * self.sr)
        signal[:impulse_len] = self.rng.randn(impulse_len).astype(np.float32) * 2

        # Exponential decay reverb
        t = np.arange(n_samples, dtype=np.float32) / self.sr
        reverb = np.exp(-8 * t) * self.rng.randn(n_samples).astype(np.float32) * 0.3
        signal += reverb

        signal /= np.abs(signal).max() + 1e-8
        return signal

    def generate(self, sound_type: str = "ambient",
                 duration: float = 1.0) -> np.ndarray:
        """Generate a synthetic audio clip of the given type."""
        generators = {
            "ambient": self.generate_ambient,
            "scream": self.generate_scream,
            "glass_breaking": self.generate_glass_breaking,
            "gunshot": self.generate_gunshot,
        }
        if sound_type not in generators:
            raise ValueError(f"Unknown sound type: {sound_type}. "
                             f"Available: {list(generators.keys())}")
        return generators[sound_type](duration)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class AudioFeatureExtractor:
    """
    Extracts mel-spectrogram features from audio waveforms.
    Designed as a pure-numpy implementation (no librosa dependency).
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 n_mels: int = MEL_BINS,
                 hop_length: int = HOP_LENGTH,
                 n_fft: int = 1024):
        self.sr = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.mel_filterbank = self._create_mel_filterbank()

    def _hz_to_mel(self, hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self) -> np.ndarray:
        """Create a mel-scale filterbank matrix."""
        low_freq_mel = 0
        high_freq_mel = self._hz_to_mel(self.sr / 2)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        bins = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)

        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1), dtype=np.float32)
        for m in range(self.n_mels):
            for k in range(bins[m], bins[m + 1]):
                if k < filterbank.shape[1]:
                    filterbank[m, k] = (k - bins[m]) / max(1, (bins[m + 1] - bins[m]))
            for k in range(bins[m + 1], bins[m + 2]):
                if k < filterbank.shape[1]:
                    filterbank[m, k] = (bins[m + 2] - k) / max(1, (bins[m + 2] - bins[m + 1]))

        return filterbank

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract log mel-spectrogram from audio waveform.

        Args:
            waveform: 1D float32 array of audio samples.

        Returns:
            (n_mels, n_time_frames) log mel-spectrogram, float32.
        """
        # STFT
        window = np.hanning(self.n_fft).astype(np.float32)
        n_frames = 1 + (len(waveform) - self.n_fft) // self.hop_length
        if n_frames <= 0:
            # Pad short audio
            padded = np.zeros(self.n_fft + self.hop_length, dtype=np.float32)
            padded[:len(waveform)] = waveform
            waveform = padded
            n_frames = 2

        stft_matrix = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = waveform[start:start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            windowed = frame * window
            spectrum = np.fft.rfft(windowed)
            stft_matrix[:, i] = np.abs(spectrum).astype(np.float32)

        # Power spectrogram
        power_spec = stft_matrix ** 2

        # Apply mel filterbank
        mel_spec = self.mel_filterbank @ power_spec  # (n_mels, n_frames)

        # Log scale
        log_mel = np.log(mel_spec + 1e-9)

        return log_mel.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# VISUAL FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class VisualFeatureExtractor:
    """
    Wraps a pre-trained ResNet-18 to produce visual embeddings from video frames.
    """

    def __init__(self, device: torch.device = DEVICE):
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.device = device

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.eval()
        # Remove final FC → 512-d output
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract 512-d feature vector from a single frame.

        Args:
            frame: Grayscale frame (H, W), float32 [0,1].

        Returns:
            512-d feature vector, float32.
        """
        # Convert to 3-channel uint8
        frame_uint8 = (frame * 255).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2RGB)

        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        feat = self.backbone(tensor).squeeze().cpu().numpy()
        return feat

    @torch.no_grad()
    def extract_batch(self, frames: list[np.ndarray]) -> np.ndarray:
        """Extract features from multiple frames. Returns (N, 512)."""
        features = []
        for frame in frames:
            features.append(self.extract(frame))
        return np.stack(features)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO-VISUAL CONTRASTIVE MODEL (CLIP-style)
# ══════════════════════════════════════════════════════════════════════════════

class AudioEncoder(nn.Module):
    """Encodes mel-spectrograms into the shared embedding space."""

    def __init__(self, n_mels: int = MEL_BINS, embed_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # (1, n_mels, T) → feature extraction
            nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (B, 256, 1)
        )
        self.projector = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: (B, n_mels, T) mel-spectrogram.
        Returns:
            (B, embed_dim) normalized embedding.
        """
        x = self.conv_layers(mel_spec)  # (B, 256, 1)
        x = x.squeeze(-1)               # (B, 256)
        x = self.projector(x)           # (B, embed_dim)
        x = F.normalize(x, dim=-1)      # L2 normalize
        return x


class VisualEncoder(nn.Module):
    """Projects ResNet features (512-d) into the shared embedding space."""

    def __init__(self, input_dim: int = 512, embed_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, visual_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feat: (B, 512) ResNet feature vectors.
        Returns:
            (B, embed_dim) normalized embedding.
        """
        x = self.projector(visual_feat)
        x = F.normalize(x, dim=-1)
        return x


class AudioVisualContrastiveModel(nn.Module):
    """
    CLIP-style dual-encoder that aligns audio and visual streams
    into a shared embedding space using InfoNCE contrastive loss.

    During inference, if audio and visual embeddings diverge significantly,
    it indicates a cross-modal anomaly.
    """

    def __init__(self, n_mels: int = MEL_BINS,
                 visual_dim: int = 512,
                 embed_dim: int = EMBEDDING_DIM,
                 temperature: float = TEMPERATURE):
        super().__init__()
        self.audio_encoder = AudioEncoder(n_mels, embed_dim)
        self.visual_encoder = VisualEncoder(visual_dim, embed_dim)
        self.temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature)), requires_grad=True
        )

    def forward(self, audio: torch.Tensor,
                visual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both modalities and return embeddings.

        Args:
            audio: (B, n_mels, T) mel-spectrograms.
            visual: (B, 512) ResNet features.

        Returns:
            audio_embeds: (B, embed_dim)
            visual_embeds: (B, embed_dim)
        """
        audio_embeds = self.audio_encoder(audio)
        visual_embeds = self.visual_encoder(visual)
        return audio_embeds, visual_embeds

    def compute_loss(self, audio_embeds: torch.Tensor,
                     visual_embeds: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss (symmetric).
        Matched pairs (diagonal) should have high similarity,
        mismatched pairs should have low similarity.
        """
        logit_scale = self.temperature.exp()

        # Cosine similarity matrix
        logits_av = logit_scale * audio_embeds @ visual_embeds.T  # (B, B)
        logits_va = logits_av.T

        B = audio_embeds.size(0)
        labels = torch.arange(B, device=audio_embeds.device)

        loss_av = F.cross_entropy(logits_av, labels)
        loss_va = F.cross_entropy(logits_va, labels)

        return (loss_av + loss_va) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-MODAL ANOMALY DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class CrossModalAnomalyDetector:
    """
    Detects anomalies by comparing audio and visual modalities.

    If what the camera sees and what the microphone hears are inconsistent
    with the learned "normal" baseline, it signals an anomaly.

    Workflow:
        1. Train on paired (audio, video) from normal scenes.
        2. At inference, compute embeddings for live audio & video.
        3. If cosine similarity deviates from the normal distribution → anomaly.
    """

    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.model = AudioVisualContrastiveModel().to(device)
        self.audio_extractor = AudioFeatureExtractor()
        self.visual_extractor = None  # Lazy-loaded
        self.normal_similarities = []  # Baseline distribution
        self.sim_mean = 0.0
        self.sim_std = 1.0

    def _ensure_visual_extractor(self):
        if self.visual_extractor is None:
            self.visual_extractor = VisualFeatureExtractor(self.device)

    def train_on_normal(self, audio_clips: list[np.ndarray],
                        video_frames: list[np.ndarray],
                        epochs: int = 20,
                        lr: float = 1e-3,
                        batch_size: int = 8):
        """
        Train the contrastive model on paired normal audio-video data.

        Args:
            audio_clips: List of 1D audio waveforms (float32).
            video_frames: List of grayscale frames (H, W), matched by index.
            epochs: Training epochs.
            lr: Learning rate.
            batch_size: Batch size.
        """
        self._ensure_visual_extractor()
        print("[Audio-Visual] Extracting audio features…")
        audio_features = []
        for clip in audio_clips:
            mel = self.audio_extractor.extract(clip)
            audio_features.append(mel)

        print("[Audio-Visual] Extracting visual features…")
        visual_features = self.visual_extractor.extract_batch(video_frames)

        # Pad/truncate audio features to uniform time dimension
        max_t = max(f.shape[1] for f in audio_features)
        audio_padded = []
        for f in audio_features:
            if f.shape[1] < max_t:
                f = np.pad(f, ((0, 0), (0, max_t - f.shape[1])))
            else:
                f = f[:, :max_t]
            audio_padded.append(f)

        audio_tensor = torch.tensor(np.stack(audio_padded), dtype=torch.float32).to(self.device)
        visual_tensor = torch.tensor(visual_features, dtype=torch.float32).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        n_samples = len(audio_clips)
        print(f"[Audio-Visual] Training contrastive model ({n_samples} pairs, {epochs} epochs)…")

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                if end - start < 2:
                    continue  # Need at least 2 for contrastive loss

                batch_idx = indices[start:end]
                audio_batch = audio_tensor[batch_idx]
                visual_batch = visual_tensor[batch_idx]

                optimizer.zero_grad()
                audio_embeds, visual_embeds = self.model(audio_batch, visual_batch)
                loss = self.model.compute_loss(audio_embeds, visual_embeds)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3d}/{epochs} — Loss: {avg_loss:.4f}")

        # Compute normal baseline similarities
        self.model.eval()
        with torch.no_grad():
            audio_embeds, visual_embeds = self.model(audio_tensor, visual_tensor)
            sims = F.cosine_similarity(audio_embeds, visual_embeds, dim=-1)
            self.normal_similarities = sims.cpu().numpy()
            self.sim_mean = float(self.normal_similarities.mean())
            self.sim_std = float(self.normal_similarities.std()) + 1e-8

        print(f"[Audio-Visual] ✓ Training complete. "
              f"Normal similarity: {self.sim_mean:.4f} ± {self.sim_std:.4f}")

    @torch.no_grad()
    def detect(self, audio_clip: np.ndarray,
               video_frame: np.ndarray,
               z_threshold: float = 2.0) -> dict:
        """
        Detect cross-modal anomaly for a single audio-video pair.

        Args:
            audio_clip: 1D audio waveform (float32).
            video_frame: Grayscale frame (H, W), float32 [0,1].
            z_threshold: Z-score threshold for anomaly detection.

        Returns:
            Dict with 'similarity', 'z_score', 'is_anomaly', 'severity'.
        """
        self._ensure_visual_extractor()
        self.model.eval()

        # Extract features
        mel = self.audio_extractor.extract(audio_clip)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(self.device)

        visual_feat = self.visual_extractor.extract(video_frame)
        visual_tensor = torch.tensor(visual_feat, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Get embeddings
        audio_embed, visual_embed = self.model(mel_tensor, visual_tensor)

        # Compute similarity
        similarity = F.cosine_similarity(audio_embed, visual_embed, dim=-1).item()

        # Z-score against normal baseline
        z_score = (self.sim_mean - similarity) / self.sim_std  # Lower sim = higher z

        is_anomaly = z_score > z_threshold
        severity = "low"
        if z_score > z_threshold * 2:
            severity = "critical"
        elif z_score > z_threshold * 1.5:
            severity = "high"
        elif z_score > z_threshold:
            severity = "medium"

        return {
            "similarity": similarity,
            "z_score": z_score,
            "is_anomaly": bool(is_anomaly),
            "severity": severity,
        }

    @torch.no_grad()
    def detect_batch(self, audio_clips: list[np.ndarray],
                     video_frames: list[np.ndarray],
                     z_threshold: float = 2.0) -> list[dict]:
        """Detect anomalies for multiple audio-video pairs."""
        results = []
        for audio, frame in zip(audio_clips, video_frames):
            results.append(self.detect(audio, frame, z_threshold))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST / DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Audio-Visual Cross-Modal Verification — Demo")
    print("=" * 70)

    # --- Generate synthetic data ---
    print("\n[Demo] Generating synthetic paired data…")
    audio_gen = SyntheticAudioGenerator(seed=42)
    n_normal = 30

    # Normal pairs: ambient audio + calm visual scene
    normal_audio = [audio_gen.generate("ambient", 1.0) for _ in range(n_normal)]

    # Create variety of normal visual frames
    normal_frames = []
    for i in range(n_normal):
        bg = np.random.uniform(0.4, 0.6, (224, 224)).astype(np.float32)
        bg += np.random.randn(224, 224).astype(np.float32) * 0.02
        normal_frames.append(np.clip(bg, 0, 1))

    # --- Train the cross-modal model ---
    print("\n[Demo] Training cross-modal contrastive model…")
    detector = CrossModalAnomalyDetector(device=DEVICE)
    detector.train_on_normal(normal_audio, normal_frames, epochs=15)

    # --- Test with various scenarios ---
    print("\n" + "-" * 70)
    print("CROSS-MODAL ANOMALY DETECTION RESULTS")
    print("-" * 70)
    print(f"{'Scenario':<30s}  {'Sim':>6s}  {'Z-score':>8s}  {'Anomaly?':>8s}  {'Severity':>10s}")
    print("-" * 70)

    test_cases = [
        ("Normal: ambient + calm",     "ambient",        0.5),
        ("Scream + calm scene",         "scream",         0.5),
        ("Glass breaking + calm",       "glass_breaking", 0.5),
        ("Gunshot + calm scene",        "gunshot",        0.5),
        ("Normal: ambient + calm #2",   "ambient",        0.55),
    ]

    for name, audio_type, bg_val in test_cases:
        audio = audio_gen.generate(audio_type, 1.0)
        frame = np.full((224, 224), bg_val, dtype=np.float32)
        frame += np.random.randn(224, 224).astype(np.float32) * 0.02
        frame = np.clip(frame, 0, 1)

        result = detector.detect(audio, frame)
        anomaly_str = "YES ⚠" if result["is_anomaly"] else "no"
        print(f"{name:<30s}  {result['similarity']:>6.3f}  "
              f"{result['z_score']:>8.3f}  {anomaly_str:>8s}  "
              f"{result['severity']:>10s}")

    print("-" * 70)
    print("\n" + "=" * 70)
    print("Audio-Visual Cross-Modal Verification — Demo Complete")
    print("=" * 70)
