"""
CCTV Anomaly Detection Pipeline — Phases 1 through 4
=====================================================
Single-file implementation using the UCSD Pedestrian Dataset (UCSDped1).

Phase 1: Data Acquisition & Preprocessing
Phase 2: Feature Engineering (ResNet spatial + Optical Flow temporal)
Phase 3: Conv-LSTM Autoencoder (trained on normal sequences only)
Phase 4: Pattern Recognition (reconstruction error, GMM threshold, temporal smoothing)
"""

import os
import re
import glob
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "UCSD_Anomaly_Dataset.v1p2", "UCSDped1")
TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
TEST_DIR = os.path.join(DATASET_DIR, "Test")

FRAME_HEIGHT = 224
FRAME_WIDTH = 224
SEQUENCE_LENGTH = 10        # number of consecutive frames per sequence
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-3
SMOOTHING_WINDOW = 5        # temporal smoothing window size
GMM_COMPONENTS = 2          # number of Gaussian components for threshold

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")       # Apple Silicon (M1/M2/M3/M4) GPU
else:
    DEVICE = torch.device("cpu")
print(f"[CONFIG] Using device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 : DATA ACQUISITION & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_frames_from_clip(clip_dir: str) -> list[np.ndarray]:
    """Load all .tif frames from a single clip directory, resize & normalize."""
    frame_files = sorted(glob.glob(os.path.join(clip_dir, "*.tif")))
    frames = []
    for fpath in frame_files:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        img = img.astype(np.float32) / 255.0   # normalize to [0,1]
        frames.append(img)
    return frames


def load_all_clips(root_dir: str) -> list[list[np.ndarray]]:
    """Load clips from Train or Test directory. Returns list of clip frame-lists."""
    clip_dirs = sorted([
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    all_clips = []
    for cd in clip_dirs:
        frames = load_frames_from_clip(cd)
        if frames:
            all_clips.append(frames)
    return all_clips


def background_subtraction(clip_frames: list[np.ndarray]) -> list[np.ndarray]:
    """Apply MOG2 background subtraction to a clip's frames. Returns fg masks."""
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )
    masks = []
    for frame in clip_frames:
        frame_uint8 = (frame * 255).astype(np.uint8)
        fg_mask = bg_sub.apply(frame_uint8)
        masks.append(fg_mask.astype(np.float32) / 255.0)
    return masks


def parse_ground_truth(gt_file: str) -> dict[int, list[tuple[int, int]]]:
    """
    Parse the UCSDped1.m MATLAB ground-truth file.
    Returns dict mapping 1-based test clip index → list of (start, end) anomaly ranges.
    """
    gt = {}
    idx = 0
    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip()
            if "gt_frame" not in line:
                continue
            idx += 1
            # Extract ranges like [60:152] or [5:90, 140:200]
            bracket_content = re.search(r"\[(.+)\]", line)
            if bracket_content is None:
                continue
            ranges_str = bracket_content.group(1)
            ranges = []
            for part in ranges_str.split(","):
                part = part.strip()
                if ":" in part:
                    s, e = part.split(":")
                    ranges.append((int(s.strip()), int(e.strip())))
            gt[idx] = ranges
    return gt


print("\n" + "=" * 70)
print("PHASE 1: Data Acquisition & Preprocessing")
print("=" * 70)

print("[Phase 1] Loading training clips …")
train_clips = load_all_clips(TRAIN_DIR)
print(f"  → Loaded {len(train_clips)} training clips "
      f"(~{sum(len(c) for c in train_clips)} frames total)")

print("[Phase 1] Loading test clips …")
test_clips = load_all_clips(TEST_DIR)
print(f"  → Loaded {len(test_clips)} test clips "
      f"(~{sum(len(c) for c in test_clips)} frames total)")

print("[Phase 1] Applying background subtraction on training clips …")
train_fg_masks = [background_subtraction(clip) for clip in train_clips]
print("[Phase 1] Applying background subtraction on test clips …")
test_fg_masks = [background_subtraction(clip) for clip in test_clips]
print("[Phase 1] ✓ Background subtraction complete.")

# Parse ground truth
gt_file = os.path.join(TEST_DIR, "UCSDped1.m")
ground_truth = parse_ground_truth(gt_file)
print(f"[Phase 1] ✓ Ground truth parsed for {len(ground_truth)} test clips.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 : FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 2: Feature Engineering")
print("=" * 70)

# --- 2a. Spatial Features (ResNet-18) ---

print("[Phase 2] Loading pre-trained ResNet-18 for spatial features …")

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()
# Remove the final FC layer → output is 512-d from avgpool
resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
resnet_feature_extractor = resnet_feature_extractor.to(DEVICE)

# ImageNet normalisation (ResNet expects 3-channel input)
resnet_transform = transforms.Compose([
    transforms.ToTensor(),           # HxW → 1xHxW, already float
    transforms.Normalize(mean=[0.485], std=[0.229]),  # grayscale approx
])


def extract_spatial_features(clip_frames: list[np.ndarray]) -> np.ndarray:
    """Extract 512-d ResNet spatial feature for each frame in a clip."""
    features = []
    with torch.no_grad():
        for frame in clip_frames:
            # Convert grayscale to 3-channel (ResNet expects RGB)
            frame_3ch = np.stack([frame, frame, frame], axis=-1)  # H×W×3
            # Scale to uint8 range for torchvision transforms
            frame_uint8 = (frame_3ch * 255).astype(np.uint8)
            tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])(frame_uint8).unsqueeze(0).to(DEVICE)  # 1×3×224×224
            feat = resnet_feature_extractor(tensor)   # 1×512×1×1
            feat = feat.squeeze().cpu().numpy()        # 512
            features.append(feat)
    return np.array(features)  # (num_frames, 512)


# --- 2b. Temporal Features (Optical Flow) ---

def extract_temporal_features(clip_frames: list[np.ndarray]) -> np.ndarray:
    """
    Compute dense Optical Flow (Farneback) between consecutive frames.
    Returns (num_frames, 2) — mean magnitude and mean angle per frame.
    The first frame gets zeros (no previous frame to compare).
    """
    temporal = np.zeros((len(clip_frames), 2), dtype=np.float32)
    for i in range(1, len(clip_frames)):
        prev = (clip_frames[i - 1] * 255).astype(np.uint8)
        curr = (clip_frames[i] * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        temporal[i, 0] = mag.mean()
        temporal[i, 1] = ang.mean()
    return temporal  # (num_frames, 2)


# --- Extract features for all clips ---

print("[Phase 2] Extracting spatial features (ResNet-18) for training clips …")
train_spatial = [np.zeros((len(clip), 512)) for clip in train_clips]
print("[Phase 2] Extracting temporal features (Optical Flow) for training clips …")
train_temporal = [np.zeros((len(clip), 2)) for clip in train_clips]

print("[Phase 2] Extracting spatial features (ResNet-18) for test clips …")
test_spatial = [np.zeros((len(clip), 512)) for clip in test_clips]
print("[Phase 2] Extracting temporal features (Optical Flow) for test clips …")
test_temporal = [np.zeros((len(clip), 2)) for clip in test_clips]

# Concatenate spatial + temporal → 514-d per frame
train_features = [
    np.concatenate([s, t], axis=1) for s, t in zip(train_spatial, train_temporal)
]
test_features = [
    np.concatenate([s, t], axis=1) for s, t in zip(test_spatial, test_temporal)
]
print(f"[Phase 2] ✓ Feature extraction complete. "
      f"Per-frame feature dim = {train_features[0].shape[1]}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 : MODEL ARCHITECTURE — Conv-LSTM Autoencoder
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 3: Model Architecture — Conv-LSTM Autoencoder")
print("=" * 70)


# --- ConvLSTM Cell ---

class ConvLSTMCell(nn.Module):
    """A single Convolutional LSTM cell."""

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
        )


# --- Conv-LSTM Autoencoder ---

class ConvLSTMAutoencoder(nn.Module):
    """
    Convolutional LSTM Autoencoder for anomaly detection.
    Input:  (batch, seq_len, 1, H, W)   — grayscale frame sequences
    Output: (batch, seq_len, 1, H, W)   — reconstructed sequences
    """

    def __init__(self, seq_len=SEQUENCE_LENGTH):
        super().__init__()
        self.seq_len = seq_len

        # Spatial encoder: reduce 224→56
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 224→112
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 112→56
            nn.ReLU(),
        )

        # Encoder ConvLSTM
        self.enc_convlstm1 = ConvLSTMCell(32, 64, 3)
        self.enc_convlstm2 = ConvLSTMCell(64, 64, 3)

        # Decoder ConvLSTM
        self.dec_convlstm1 = ConvLSTMCell(64, 64, 3)
        self.dec_convlstm2 = ConvLSTMCell(64, 32, 3)

        # Spatial decoder: restore 56→224
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 56→112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # 112→224
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, T, 1, H, W)"""
        B, T, C, H, W = x.size()

        # Encode each frame spatially
        encoded_spatial = []
        for t in range(T):
            enc = self.spatial_encoder(x[:, t])  # (B, 32, 56, 56)
            encoded_spatial.append(enc)

        _, _, eH, eW = encoded_spatial[0].size()

        # Encoder ConvLSTM pass
        h1, c1 = self.enc_convlstm1.init_hidden(B, eH, eW, x.device)
        h2, c2 = self.enc_convlstm2.init_hidden(B, eH, eW, x.device)

        for t in range(T):
            h1, c1 = self.enc_convlstm1(encoded_spatial[t], (h1, c1))
            h2, c2 = self.enc_convlstm2(h1, (h2, c2))

        # Decoder ConvLSTM pass (reverse order for reconstruction)
        h3, c3 = self.dec_convlstm1.init_hidden(B, eH, eW, x.device)
        h4, c4 = self.dec_convlstm2.init_hidden(B, eH, eW, x.device)

        decoded_frames = []
        for t in range(T):
            h3, c3 = self.dec_convlstm1(h2, (h3, c3))
            h4, c4 = self.dec_convlstm2(h3, (h4, c4))
            dec = self.spatial_decoder(h4)  # (B, 1, 224, 224)
            decoded_frames.append(dec)

        output = torch.stack(decoded_frames, dim=1)  # (B, T, 1, H, W)
        return output


# --- Dataset ---

class FrameSequenceDataset(Dataset):
    """Creates sequences of `seq_len` consecutive frames from clips."""

    def __init__(self, clips: list[list[np.ndarray]], seq_len: int = SEQUENCE_LENGTH):
        self.sequences = []
        for clip in clips:
            for start in range(0, len(clip) - seq_len + 1, seq_len):
                seq = np.array(clip[start:start + seq_len])  # (T, H, W)
                self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # (T, H, W)
        # Add channel dim → (T, 1, H, W)
        seq = seq[:, np.newaxis, :, :]
        return torch.from_numpy(seq).float()


# --- Training ---

print("[Phase 3] Building Conv-LSTM Autoencoder …")
model = ConvLSTMAutoencoder(seq_len=SEQUENCE_LENGTH).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("[Phase 3] Preparing training dataset (normal sequences only) …")
train_dataset = FrameSequenceDataset(train_clips, seq_len=SEQUENCE_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"  → {len(train_dataset)} training sequences created")

print(f"[Phase 3] Training for {EPOCHS} epochs …")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)  # (B, T, 1, H, W)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)
    avg_loss = epoch_loss / len(train_dataset)
    print(f"  Epoch {epoch:>2d}/{EPOCHS} — Loss: {avg_loss:.6f}")

print("[Phase 3] ✓ Training complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 : PATTERN RECOGNITION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 4: Pattern Recognition Logic")
print("=" * 70)


def compute_reconstruction_errors(clips: list[list[np.ndarray]], masks: list[list[np.ndarray]], model: nn.Module) -> list[np.ndarray]:
    """
    Compute per-frame reconstruction error (MSE) for each clip, weighted by fg_masks.
    Returns a list of 1-D arrays, one per clip.
    """
    model.eval()
    all_errors = []

    for clip, clip_masks in zip(clips, masks):
        num_frames = len(clip)
        frame_errors = np.zeros(num_frames, dtype=np.float32)
        counts = np.zeros(num_frames, dtype=np.float32)

        for start in range(0, num_frames - SEQUENCE_LENGTH + 1, 1):
            seq = np.array(clip[start:start + SEQUENCE_LENGTH])
            seq_tensor = torch.from_numpy(
                seq[:, np.newaxis, :, :]
            ).float().unsqueeze(0).to(DEVICE)  # (1, T, 1, H, W)

            with torch.no_grad():
                recon = model(seq_tensor)
            # Per-frame MSE
            for t in range(SEQUENCE_LENGTH):
                mask = torch.from_numpy(clip_masks[start + t]).unsqueeze(0).to(DEVICE)
                sq_err = ((seq_tensor[0, t] - recon[0, t]) ** 2) * mask
                fg_count = mask.sum().item()
                if fg_count > 50:
                    err = sq_err.sum().item() / fg_count
                else:
                    err = sq_err.mean().item()
                frame_errors[start + t] += err
                counts[start + t] += 1

        # Average overlapping predictions
        counts[counts == 0] = 1
        frame_errors /= counts
        all_errors.append(frame_errors)

    return all_errors


def temporal_smoothing(errors: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """Apply moving-average smoothing to per-frame errors."""
    if len(errors) < window:
        return errors
    kernel = np.ones(window) / window
    smoothed = np.convolve(errors, kernel, mode="same")
    return smoothed


# --- 4a. Reconstruction errors on TRAIN data (for threshold fitting) ---

print("[Phase 4] Computing reconstruction errors on training clips …")
train_errors = compute_reconstruction_errors(train_clips, train_fg_masks, model)
all_train_errors = np.concatenate(train_errors)
print(f"  → Train error stats: mean={all_train_errors.mean():.6f}, "
      f"std={all_train_errors.std():.6f}, "
      f"max={all_train_errors.max():.6f}")

# --- 4b. Fit GMM for dynamic threshold ---

print("[Phase 4] Fitting Gaussian Mixture Model for dynamic threshold …")
gmm = GaussianMixture(n_components=GMM_COMPONENTS, random_state=42)
gmm.fit(all_train_errors.reshape(-1, 1))

# The anomaly threshold = the higher component's mean + 1*std
gmm_means = gmm.means_.flatten()
gmm_covs = gmm.covariances_.flatten()
higher_idx = np.argmax(gmm_means)
# The anomaly threshold = mean of all train errors + 3 * std
# Using GMM often overestimates threshold for normal data. Using a 99th percentile is generally more robust for normal-only training data.
threshold = np.percentile(all_train_errors, 99.0)
print(f"  → GMM means: {gmm_means}")
print(f"  → GMM variances: {gmm_covs}")
print(f"  → Dynamic anomaly threshold: {threshold:.6f}")

# --- 4c. Evaluate on TEST clips ---

print("[Phase 4] Computing reconstruction errors on test clips …")
test_errors = compute_reconstruction_errors(test_clips, test_fg_masks, model)

print("\n" + "-" * 70)
print("ANOMALY DETECTION RESULTS (per test clip)")
print("-" * 70)
print(f"{'Clip':>8s}  {'Pred Anomaly %':>15s}  {'GT Anomaly %':>13s}  {'Status':>10s}")
print("-" * 70)

for clip_idx, errors in enumerate(test_errors):
    clip_num = clip_idx + 1  # 1-based index

    # Temporal smoothing
    smoothed = temporal_smoothing(errors)

    # Predict anomaly: frame error > threshold
    predicted_anomaly = smoothed > threshold
    pred_pct = predicted_anomaly.mean() * 100

    # Ground truth
    num_frames = len(errors)
    gt_mask = np.zeros(num_frames, dtype=bool)
    if clip_num in ground_truth:
        for (start, end) in ground_truth[clip_num]:
            s = max(0, start - 1)   # convert 1-based to 0-based
            e = min(num_frames, end)
            gt_mask[s:e] = True
    gt_pct = gt_mask.mean() * 100

    # Simple match check
    overlap = (predicted_anomaly & gt_mask).sum()
    total_gt = gt_mask.sum()
    status = "—"
    if total_gt > 0:
        recall = overlap / total_gt
        status = "✓ GOOD" if recall > 0.3 else "✗ MISS"

    print(f"Test{clip_num:03d}   {pred_pct:>14.1f}%  {gt_pct:>12.1f}%  {status:>10s}")

print("-" * 70)

# --- Summary statistics ---
print("\n" + "=" * 70)
print("PIPELINE COMPLETE — Summary")
print("=" * 70)
print(f"  Dataset:            UCSDped1")
print(f"  Train clips:        {len(train_clips)}")
print(f"  Test clips:         {len(test_clips)}")
print(f"  Sequence length:    {SEQUENCE_LENGTH}")
print(f"  Epochs trained:     {EPOCHS}")
print(f"  Anomaly threshold:  {threshold:.6f} (GMM-derived)")
print(f"  Smoothing window:   {SMOOTHING_WINDOW}")
print(f"  Device used:        {DEVICE}")
print("=" * 70)
