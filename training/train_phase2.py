"""
train_phase2.py — Sequential Training for IVF Phase 2 (Causal Window Attention)
=================================================================================
Each embryo (patient) is an "episode" — the model processes each frame chronologically.

Workflow:
  1. Load Phase 1 model (frozen) — Mamba2VisionMorph
  2. For each embryo in the training set:
     a. Retrieve the list of frames in order (RUN1, RUN2, ...) — ALL frames
     b. Slide a 5-frame window, stride=1
     c. At each position t:
        - Phase 1 (frozen) → feat_t (640-dim) + probs_t (7-dim)
        - Phase 2 Causal Attention: [feat_t || probs_t || temporal_pos] → KV-cache → logits_t
        - Calculate loss vs GT[t] using OrdinalProgressionLoss (CE + backward-regression penalty)
  3. TBPTT: accumulate gradients over K steps, detach KV-cache, update weights

Causal Window Attention (EmbryoTemporalNet):
  - KV-cache sliding window W (default 64) — keeps the W most recent frames
  - Frame t attends to frames t-W..t (causal, no future look-ahead)
  - ALiBi position bias: closer frames receive more attention
  - Sinusoidal PE on absolute frame index (independent of total_frames)
  - Temporal position features: stage_dur (from Phase 1 argmax, fully causal)

Usage:
  python training/train_phase2.py \
      --phase1_ckpt ./output/20260412-173502-Mamba2VisionMorph-224/model_best.pth.tar \
      --data_root ./dataset/embryo_dataset \
      --ann_root  ./dataset/embryo_dataset_annotations \
      --splits_json ./processdata/splits.json \
      --output_dir ./output/phase2_attn \
      --epochs 30 --lr 3e-4 \
      --d_model 256 --d_state 64 --expand 2 --headdim 64
"""

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Telegram helper
# ---------------------------------------------------------------------------

def send_telegram(token: str, chat_id: str, text: str):
    """Send Telegram message. Silent-fail if no token or requests library."""
    if not token or not chat_id or not _HAS_REQUESTS:
        return
    try:
        url = f'https://api.telegram.org/bot{token}/sendMessage'
        _requests.post(url, data={'chat_id': chat_id, 'text': text,
                                  'parse_mode': 'HTML'}, timeout=10)
    except Exception as e:
        print(f'[Telegram] Failed: {e}')


# Constants (must match visualization scripts)
STAGE_NAMES = ['1-tPNf', '2-t2', '3-t3+', '5-t5+', '7-t7+', '9-t9+', '10-tM+']
NUM_CLASSES = len(STAGE_NAMES)

FEAT_DIM = 640  # Phase 1 encoder output dim before head (BN→GAP→640)

RAW_TO_IDX = {
    # Raw stage names (from original embryo_dataset_annotations)
    'tpnf': 0, 'tpna': 0, 'tpn': 0, 't_pn': 0,
    't2': 1,
    't3': 2, 't4': 2, 't3+': 2, 't4+': 2,
    't5': 3, 't6': 3, 't5+': 3, 't6+': 3,
    't7': 4, 't8': 4, 't7+': 4, 't8+': 4,
    't9+': 5, 't9': 5,
    'tm': 6, 'tsb': 6, 'tm+': 6,
    # Merged class names (from dataset_timelapse annotations)
    '1-tpnf': 0, '1-tpna+': 0,
    '2-t2': 1,
    '3-t3+': 2,
    '5-t5+': 3,
    '7-t7+': 4,
    '9-t9+': 5,
    '10-tm+': 6, '10-tsb+': 6,
}
EXCLUDED_STAGES = {'tpb2', 'tb', 'teb', 'thb'}

IMG_EXTS = {'.jpeg', '.jpg', '.png'}

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sort_key(p: Path) -> int:
    m = re.search(r'RUN(\d+)', p.stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+', p.stem)
    return int(nums[-1]) if nums else 0


def load_phases_csv(csv_path: Path):
    """Read {embryo}_phases.csv → [(raw_stage, start, end), ...]"""
    if not csv_path.exists():
        return []
    phases = []
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            stage = row[0].strip().lower()
            try:
                start, end = int(row[1]), int(row[2])
            except ValueError:
                continue
            phases.append((stage, start, end))
    return phases


def assign_labels(frames, phases):
    """frame_num (1-indexed) → class_idx or None if excluded."""
    labels = []
    for i, f in enumerate(frames):
        # Get frame number from filename
        m = re.search(r'RUN(\d+)', f.stem, re.IGNORECASE)
        frame_num = int(m.group(1)) if m else (i + 1)
        assigned = None
        for stage, start, end in phases:
            if start <= frame_num <= end:
                s_low = stage.lower()
                if s_low in EXCLUDED_STAGES:
                    break
                idx = RAW_TO_IDX.get(s_low)
                if idx is not None:
                    assigned = idx
                break
        labels.append(assigned)
    return labels


def _apply_clahe_sobel_np(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE + Sobel — matches balance_dataset_v5_3d.py (Phase 1 training preprocessing)."""
    import cv2
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    sx = cv2.Sobel(cl, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(cl, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sx, sy)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    edges_c = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(img_clahe, 0.85, edges_c, 0.15, 0)



# ---------------------------------------------------------------------------
# Load Phase 1 model
# ---------------------------------------------------------------------------

def load_phase1_model(ckpt_path: str, num_frames: int, device: str):
    """Load EmbryoMambaNet (Phase 1 2D) — per-frame inference, no 5-frame clip."""
    import importlib.util, sys
    model_script = Path(__file__).parent / 'models' / 'ivf_2d_phase1.py'
    spec = importlib.util.spec_from_file_location('ivf_2d_phase1', model_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules['ivf_2d_phase1'] = module
    spec.loader.exec_module(module)

    model = module.EmbryoMambaNet(num_classes=NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt
    for key in ('state_dict', 'model', 'model_state_dict'):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    print(f'✅ Phase1 EmbryoMambaNet loaded ({ckpt_path}), params frozen')
    return model

def _cv2_resize(img_np, size):
    """cv2 resize — ~3x faster than PIL."""
    import cv2
    return cv2.resize(img_np, (size, size), interpolation=cv2.INTER_LINEAR)


def _np_to_tensor_normalized(img_np, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """np.ndarray RGB uint8 → torch.Tensor (3, H, W) normalized. Faster than TF.to_tensor."""
    t = torch.from_numpy(img_np).permute(2, 0, 1).float().div_(255.0)
    for c in range(3):
        t[c].sub_(mean[c]).div_(std[c])
    return t


# Phase 1 Feature Cache — run once, used for all epochs

def cache_phase1_features(
    phase1_model,
    embryo_ids: list,
    data_root: Path,
    ann_root: Path,
    num_frames: int,
    img_size: int,
    device: str,
    cache_dir: Path,
    p1_batch_size: int = 192,
    num_workers: int = 8,
) -> dict:
    """
    Run Phase 1 inference for all embryos, save feat+probs to disk.

    Disk cache:
      - Each embryo → 1 file: cache_dir/{eid}.pt
      - File contains: {'feats': (N,640), 'probs': (N,7), 'labels': list}
      - If file exists → skip (no rerun needed)
      - No items kept in RAM during build to avoid OOM

    Speed:
      - First time: ~60-90 minutes (once only)
      - Subsequent times: ~10 seconds (path list load only)

    Returns:
        cache: dict[eid] = Path(cache_file) — lazy load when needed
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check which embryos are not yet cached
    to_build = []
    already_cached = 0
    for eid in embryo_ids:
        cache_file = cache_dir / f'{eid}.pt'
        if cache_file.exists():
            already_cached += 1
        else:
            frame_paths = sorted(
                [p for p in (data_root / eid).glob('*')
                 if p.suffix.lower() in IMG_EXTS],
                key=_run_sort_key,
            )
            if not frame_paths:
                continue
            phases = load_phases_csv(ann_root / f'{eid}_phases.csv')
            if not phases:
                continue
            labels = assign_labels(frame_paths, phases)
            to_build.append((eid, frame_paths, labels))

    if already_cached > 0:
        print(f'  ✅ {already_cached}/{len(embryo_ids)} embryos already cached on disk')

    if to_build:
        print(f'  🔄 Building cache for {len(to_build)} embryos → {cache_dir}')
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(total=len(to_build), desc='  Caching P1 features', unit='embryo')

        original_head = phase1_model.head
        try:
            phase1_model.head = nn.Identity()
            phase1_model.eval()

            for eid, frame_paths, labels in to_build:
                N = len(frame_paths)
                all_feats = torch.zeros(N, FEAT_DIM)

                # Load + CLAHE+Sobel in parallel to speed up (resolves CPU bottleneck)
                import os
                from concurrent.futures import ThreadPoolExecutor
                def _load_one(p):
                    try:
                        return _apply_clahe_sobel_np(np.array(Image.open(p).convert('RGB')))
                    except Exception as e:
                        print(f"  [Warning] Error reading image {p}: {e}")
                        return None
                
                with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 2)) as pool:
                    raw_arrays = list(pool.map(_load_one, frame_paths))

                # Per-frame inference — 5-frame clip no longer needed
                # EmbryoMambaNet accepts (B, 3, H, W) — 1 image at a time
                # forward_features() → BN → GAP → 640-d feat
                frames_batch, idx_batch = [], []

                def flush_2d():
                    if not frames_batch:
                        return
                    batch = torch.stack(frames_batch, dim=0).to(device, non_blocking=True)
                    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                        # forward_features: (B,3,H,W) → (B,640,7,7)
                        feat_map = phase1_model.forward_features(batch)
                        # BN → GAP → 640-d (similar to forward() but without head)
                        feat_map = phase1_model.norm(feat_map)
                        feats    = phase1_model.avgpool(feat_map)
                        feats    = torch.flatten(feats, 1).float()
                    for j, frame_idx in enumerate(idx_batch):
                        all_feats[frame_idx] = feats[j].cpu()
                    frames_batch.clear()
                    idx_batch.clear()

                for i, arr in enumerate(raw_arrays):
                    if arr is None:
                        continue
                    t = _np_to_tensor_normalized(_cv2_resize(arr, img_size))
                    frames_batch.append(t)
                    idx_batch.append(i)
                    if len(frames_batch) >= p1_batch_size:
                        flush_2d()
                flush_2d()

                # Save to disk immediately — release RAM
                torch.save({
                    'feats':  all_feats,
                    'labels': labels,
                }, cache_dir / f'{eid}.pt')

                del all_feats, raw_arrays
                pbar.update(1)

        finally:
            phase1_model.head = original_head
            pbar.close()

    # Return dict of paths — lazy load when needed
    cache_paths = {}
    for eid in embryo_ids:
        f = cache_dir / f'{eid}.pt'
        if f.exists():
            cache_paths[eid] = f
    print(f'✅ Cache ready: {len(cache_paths)} embryos at {cache_dir}')
    return cache_paths


def get_cached_features(cache_entry, device: str, is_training: bool = False):
    """
    Load feat+probs from disk cache (Path) or RAM dict.
    Add feat-level augmentation if training.

    Feat-level aug replaces image aug:
      - Gaussian noise σ=0.03 on feat (640-d)
      - Feature dropout 10%
    ~100x faster than image aug, no Phase 1 rerun needed.
    """
    # Load from disk if Path
    if isinstance(cache_entry, Path):
        data = torch.load(cache_entry, map_location='cpu', weights_only=True)
    else:
        data = cache_entry

    feats  = data['feats'].to(device, non_blocking=True)
    labels = data['labels']
    probs  = data.get('probs', torch.zeros(len(labels), 7, device=device)).to(device, non_blocking=True)

    if is_training:
        # Gaussian noise on feat — σ=0.1 (increased from 0.03 for diversity)
        feats = feats + torch.randn_like(feats) * 0.1
        # Feature dropout 15% — force Phase 2 to be more robust
        mask = torch.bernoulli(torch.full_like(feats, 0.85))  # keep 85%
        feats = feats * mask / 0.85

    return feats, probs, labels


def run_phase1_online(
    phase1_model,
    frame_paths: list,
    labels: list,
    num_frames: int,
    img_size: int,
    device: str,
    is_training: bool = False,
    p1_batch_size: int = 192,
    num_workers: int = 8,
):
    """
    Fallback: run Phase 1 online per-frame (EmbryoMambaNet 2D).
    Used when cache is missing or running on new embryos.
    """
    from concurrent.futures import ThreadPoolExecutor
    N = len(frame_paths)
    all_feats = torch.zeros(N, FEAT_DIM, device=device)

    def _load_one(p):
        try:
            img_np = np.array(Image.open(p).convert('RGB'))
            return _apply_clahe_sobel_np(img_np)
        except Exception as e:
            print(f"  [Warning] Error reading image {p}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        raw_arrays = list(pool.map(_load_one, frame_paths))

    frames_batch, idx_batch = [], []

    original_head = phase1_model.head
    try:
        phase1_model.head = nn.Identity()
        phase1_model.eval()

        def flush_2d():
            if not frames_batch:
                return
            batch = torch.stack(frames_batch, dim=0).to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                feat_map = phase1_model.forward_features(batch)
                feat_map = phase1_model.norm(feat_map)
                feats    = phase1_model.avgpool(feat_map)
                feats    = torch.flatten(feats, 1).float()
            for j, frame_idx in enumerate(idx_batch):
                all_feats[frame_idx] = feats[j]
            frames_batch.clear()
            idx_batch.clear()

        for i, arr in enumerate(raw_arrays):
            if arr is None:
                continue
            t = _np_to_tensor_normalized(_cv2_resize(arr, img_size))
            frames_batch.append(t)
            idx_batch.append(i)
            if len(frames_batch) >= p1_batch_size:
                flush_2d()
        flush_2d()
    finally:
        phase1_model.head = original_head

    all_probs = torch.zeros(N, NUM_CLASSES, device=device)
    return all_feats, all_probs, labels


# Train 1 embryo — Causal Window Attention TBPTT chunk-based

def _detach_cache(cache):
    """Detach SSM state between TBPTT chunks.

    cache: list[ssm_state per layer]
    Each element: tensor (B, nheads, headdim, d_state) or None.

    Mamba2 SSM state is much smaller than the old KV-cache:
      SSM state: nheads × headdim × d_state = 8 × 64 × 64 = 32,768 values ≈ 128 KB
      KV-cache W=128: 128 × 256 × 2 = 65,536 values ≈ 512 KB
    """
    return [
        state.detach() if state is not None else None
        for state in cache
    ]


def train_batch_embryos(
    phase2, optimizer, criterion,
    batch_eids: list,
    p1_feat_cache: dict,
    device: str,
    accumulate_steps: int = 32,
    clip_grad: float = 1.0,
    is_training: bool = True,
):
    """
    Train multiple embryos in parallel in one batch — increases GPU utilization.

    Instead of B=1 (1 embryo, 1 frame/step), run B embryos simultaneously:
      - Each embryo has its own SSM state: (B, nheads, headdim, d_state)
      - At each step t: take frame t of ALL embryos → batch (B, 640)
      - Forward through Phase 2 with batch B → logits (B, 7)
      - Embryos that run out of frames are masked out of the loss, keeping SSM state stable

    Args:
        batch_eids: list of embryo IDs in this batch
        p1_feat_cache: dict[eid] = Path or dict cache

    Returns:
        (avg_loss, total_valid_frames, avg_grad_norm)
    """
    phase2.train()
    B = len(batch_eids)
    if B == 0:
        return 0.0, 0, 0.0

    # ── Load features for all embryos in the batch ───────────────────────
    all_data = []
    for eid in batch_eids:
        cache_entry = p1_feat_cache.get(eid)
        if cache_entry is None:
            continue
        feats, _, labels = get_cached_features(cache_entry, device, is_training=is_training)
        valid_idx = [i for i, lbl in enumerate(labels) if lbl is not None]
        N = len(labels)
        all_data.append({
            'feats': feats, 'labels': labels,
            'valid_idx': valid_idx, 'N': N,
        })

    if not all_data:
        return 0.0, 0, 0.0

    B_actual = len(all_data)
    max_valid = max(len(d['valid_idx']) for d in all_data)
    if max_valid == 0:
        return 0.0, 0, 0.0

    # ── Initialize SSM state for B_actual embryos ───────────────────────────
    ssm_cache = phase2.make_initial_hidden(B_actual, device)

    total_loss = 0.0
    total_grad_norm = 0.0
    total_valid = 0
    valid_steps = 0

    for chunk_start in range(0, max_valid, accumulate_steps):
        chunk_end = min(chunk_start + accumulate_steps, max_valid)
        L = chunk_end - chunk_start

        optimizer.zero_grad()
        chunk_loss = torch.tensor(0.0, device=device)
        n_active = 0

        for ci in range(L):
            vi = chunk_start + ci  # valid frame index

            feat_list, gt_list, active_mask, frame_idx_list = [], [], [], []
            for d in all_data:
                if vi < len(d['valid_idx']):
                    e = d['valid_idx'][vi]
                    feat_list.append(d['feats'][e])
                    gt_list.append(d['labels'][e])
                    active_mask.append(True)
                    frame_idx_list.append(e)   # actual frame index
                else:
                    last_e = d['valid_idx'][-1] if d['valid_idx'] else 0
                    feat_list.append(d['feats'][last_e])
                    gt_list.append(None)
                    active_mask.append(False)
                    frame_idx_list.append(last_e)

            feat_batch = torch.stack(feat_list, dim=0)           # (B, 640)

            # Pass actual frame_idx list → each embryo receives PE at the correct position
            logits, ssm_cache, _ = phase2(
                feat_batch, None, ssm_cache,
                frame_idx=frame_idx_list,   # list[int]
            )  # logits: (B, 7)

            for bi, (active, gt) in enumerate(zip(active_mask, gt_list)):
                if active and gt is not None:
                    gt_t = torch.tensor([gt], dtype=torch.long, device=device)
                    chunk_loss = chunk_loss + criterion(logits[bi:bi+1], gt_t)
                    n_active += 1

        if n_active > 0:
            chunk_loss = chunk_loss / n_active
            chunk_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(phase2.parameters(), max_norm=clip_grad)
            optimizer.step()
            total_loss += chunk_loss.item()
            total_grad_norm += grad_norm.item()
            total_valid += n_active
            valid_steps += 1

        ssm_cache = _detach_cache(ssm_cache)

    return total_loss / max(valid_steps, 1), total_valid, total_grad_norm / max(valid_steps, 1)

# ---------------------------------------------------------------------------
# Evaluate 1 embryo — Causal Window Attention full-sequence rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one_embryo(phase1, phase2, frames, labels, num_frames, img_size, device,
                    criterion=None, feat_cache_dir=None, embryo_id='',
                    p1_batch_size=64, num_workers=4, p1_cache=None):
    """
    Eval 1 embryo — uses cached features, Mamba2 SSM full-sequence rollout.
    """
    phase2.eval()

    # Retrieve features from cache (no augmentation)
    if p1_cache is not None:
        all_feats, all_probs, labels = get_cached_features(
            p1_cache, device, is_training=False)
    else:
        all_feats, all_probs, labels = run_phase1_online(
            phase1, frames, labels, num_frames, img_size, device,
            is_training=False,
            p1_batch_size=p1_batch_size,
            num_workers=num_workers,
        )

    n = len(labels)

    ssm_cache = phase2.make_initial_hidden(1, device)
    correct, total = 0, 0
    correct_top3 = 0
    total_loss = 0.0
    preds_all, gts_all = [], []
    frames_seen = 0

    for end in range(n):   # per-frame: start from frame 0
        gt    = labels[end]
        feat  = all_feats[end].unsqueeze(0)

        logits, ssm_cache, _ = phase2(feat, None, ssm_cache, frame_idx=end)
        frames_seen += 1
        pred = logits.argmax(dim=-1).item()

        preds_all.append(pred)
        gts_all.append(gt)

        if gt is not None:
            if criterion is not None:
                gt_t = torch.tensor([gt], dtype=torch.long, device=device)
                total_loss += criterion(logits, gt_t).item()

            top3_preds = logits.topk(3, dim=-1)[1].squeeze(0).tolist()
            if gt in top3_preds:
                correct_top3 += 1
            total += 1
            if pred == gt:
                correct += 1

    acc      = correct / total * 100 if total > 0 else 0
    top3_acc = correct_top3 / total * 100 if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0.0
    return acc, top3_acc, avg_loss, preds_all, gts_all


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
# Compute class weight from training set (inverse frequency)
# ---------------------------------------------------------------------------

def compute_class_weight(
    train_embryos: list,
    data_root: Path,
    ann_root: Path,
    num_classes: int,
    beta: float = 0.999,
) -> torch.Tensor:
    """
    Calculate class weight according to Effective Number of Samples (Class-Balanced Loss, CVPR 2019).

    weight[k] = (1 - beta) / (1 - beta^n_k)

    beta=0.999  (default, decreased from 0.9999):
      Less aggressive — avoids boosting tpnf too strongly (2.19x → ~1.8x).
      For 7-class IVF, a max/min ratio of ~3x is sufficient.

    beta=0.9999: close to pure inverse frequency (ratio can be >5x).
    beta=0:      all weights = 1 (disables class balancing).

    Cap at max_ratio=3.0: prevents any class from being boosted too strongly.
    """
    counts = torch.zeros(num_classes)

    for eid in train_embryos:
        frames = sorted(
            [p for p in (data_root / eid).glob('*')
             if p.suffix.lower() in IMG_EXTS],
            key=_run_sort_key,
        )
        if not frames:
            continue
        phases = load_phases_csv(ann_root / f'{eid}_phases.csv')
        if not phases:
            continue
        labels = assign_labels(frames, phases)
        for lbl in labels:
            if lbl is not None:
                counts[lbl] += 1

    if beta == 0.0:
        weight = torch.ones(num_classes)
    else:
        effective_num = 1.0 - torch.pow(beta, counts.clamp(min=1))
        weight = (1.0 - beta) / effective_num

    # Normalize: mean = 1
    weight = weight / weight.mean()

    # Cap max ratio at 8.0 — tpnf has the fewest frames (~3138 vs ~24000 t9+)
    # actual ratio ~7.6x → need higher cap to boost sufficiently
    max_ratio = 8.0
    weight = weight.clamp(max=max_ratio)
    weight = weight / weight.mean()  # re-normalize after clip

    print('📊 Class weight (effective frequency, beta={:.4f}, cap={:.1f}):'.format(beta, max_ratio))
    for i, (name, w, c) in enumerate(zip(STAGE_NAMES, weight.tolist(), counts.tolist())):
        bar = '█' * int(w * 5) + '░' * max(0, 15 - int(w * 5))
        print(f'   {name:<10}: {bar} {w:.3f}  (frames={int(c):6d})')

    if counts[0] == 0:
        print('⚠️  WARNING: tpnf (class 0) has 0 frames in training set!')
        print('   Check annotation CSV: tPNf must be in RAW_TO_IDX and not in EXCLUDED_STAGES')
    elif counts[0] < 500:
        print(f'⚠️  WARNING: tpnf only {int(counts[0])} frames — model may struggle to learn it')

    return weight


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser('IVF Phase 2 — Causal Window Attention Sequential Training')

    # ── Paths ────────────────────────────────────────────────────────────────
    parser.add_argument('--phase1_ckpt', default='', required=True,
                        help='Path to Phase 1 checkpoint (model_best.pth.tar)')
    parser.add_argument('--data_root',   required=True, help='Raw embryo images root')
    parser.add_argument('--ann_root',    required=True, help='Annotation CSVs root')
    parser.add_argument('--splits_json', required=True, help='train/val split JSON')
    parser.add_argument('--output_dir',  default='./output/phase2_mamba')
    parser.add_argument('--resume',      default='', help='Resume from checkpoint (phase2_last.pth)')
    parser.add_argument('--feat_cache_dir', default='',
                        help='Directory to store Phase 1 feature cache (default: output_dir/p1_cache). '
                             'If cache exists → skip build, load immediately.')

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument('--num_frames',  type=int,   default=5,   help='Frames per clip (Phase 1 inference, default 5 — ignored for 2D EmbryoMambaNet)')
    parser.add_argument('--img_size',    type=int,   default=224, help='Image size (default 224)')
    parser.add_argument('--workers',     type=int,   default=16,   help='Threads CLAHE+Sobel preprocessing per embryo (default 8)')
    parser.add_argument('--p1_batch_size', type=int, default=192,
                        help='Batch size for Phase 1 inference per embryo (default 192). '
                             '3080 16GB: use 192. 4090 24GB: use 256.')
    parser.add_argument('--embryo_batch_size', type=int, default=8,
                        help='Number of embryos running in parallel during Phase 2 training (default 8). '
                             'Increase for better GPU utilization. 3080 16GB: 8-16. '
                             'Each embryo requires ~50MB VRAM for SSM state.')
    parser.add_argument('--no_aug',      action='store_true', default=False,
                        help='Disable augmentation during training (debug only)')

    # ── Augmentation (applied for Phase 1 online inference) ─────────────────
    parser.add_argument('--scale',       type=float, nargs=2, default=[0.7, 1.0],
                        help='RandomResizedCrop scale range (default 0.7 1.0)')
    parser.add_argument('--hflip',       type=float, default=0.5,  help='Horizontal flip prob (default 0.5)')
    parser.add_argument('--vflip',       type=float, default=0.2,  help='Vertical flip prob (default 0.2)')
    parser.add_argument('--color_jitter', type=float, default=0.3, help='ColorJitter strength (default 0.3)')
    parser.add_argument('--reprob',      type=float, default=0.15, help='RandomErasing prob (default 0.15)')
    parser.add_argument('--smoothing',   type=float, default=0.1,  help='Label smoothing (default 0.1)')

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument('--d_model',     type=int,   default=640,
                        help='Mamba2 d_model — must equal feat_dim=640 (no dimensionality reduction)')
    parser.add_argument('--n_layers',    type=int,   default=4,
                        help='Number of Mamba2 blocks (default 4). Cache stores n_layers SSM states.')
    parser.add_argument('--d_state',     type=int,   default=64,
                        help='SSM state dim per head (default 64). '
                             'Larger = better memory, but more memory consumption. '
                             'Mamba2 paper uses 64 (8× Mamba1).')
    parser.add_argument('--expand',      type=int,   default=2,
                        help='d_inner = d_model × expand (default 2). '
                             'nheads = d_inner / headdim = 256×2/64 = 8.')
    parser.add_argument('--headdim',     type=int,   default=64,
                        help='SSM head dim (default 64). nheads = d_inner / headdim.')
    parser.add_argument('--chunk_size',  type=int,   default=128,
                        help='Mamba2 SSM scan chunk size (default 128). '
                             'Larger = faster during training with large L.')
    parser.add_argument('--dropout',     type=float, default=0.15, help='Dropout (default 0.15)')
    # backward compat — no longer used but kept to avoid errors in legacy scripts
    parser.add_argument('--n_heads',     type=int,   default=4,   help='[deprecated] use --headdim instead')
    parser.add_argument('--window_size', type=int,   default=128,
                        help='[deprecated] Mamba2 does not use KV-cache. '
                             'Kept to calculate stage_dur = P1_run / window_size.')
    parser.add_argument('--ordinal_alpha', type=float, default=0.3,
                        help='Backward regression penalty weight (default 0.3)')
    parser.add_argument('--forward_alpha', type=float, default=0.2,
                        help='Forward jump penalty weight (default 0.2). '
                             'Penalizes prediction stages higher than GT by more than max_forward_jump levels.')
    parser.add_argument('--max_forward_jump', type=int, default=2,
                        help='Maximum allowed jump levels (default 2). '
                             'tpnf→t3+ OK (2 levels), tpnf→t5+ penalized (3 levels).')

    # ── Optimizer ────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',       type=int,   default=30,   help='Number of epochs (default 30)')
    parser.add_argument('--lr',           type=float, default=1e-4, help='Learning rate (default 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (default 1e-4)')
    parser.add_argument('--clip_grad',    type=float, default=1.0,
                        help='Gradient clip norm (default 1.0)')
    parser.add_argument('--warmup_epochs', type=int,  default=2,
                        help='LR warmup epochs (default 2)')

    # ── TBPTT ────────────────────────────────────────────────────────────────
    parser.add_argument('--accumulate_steps', type=int, default=32,
                        help='TBPTT chunk size — number of frames per backward pass. '
                             'Larger = longer gradient flow in sequence. '
                             'default 32: ~8 minutes context @ 1 frame/15 minutes.')
    parser.add_argument('--warmup_frames', type=int, default=0,
                        help='Number of initial frames per embryo running forward without loss (default 0)')

    # ── Class weight ─────────────────────────────────────────────────────────
    parser.add_argument('--class_weight_beta', type=float, default=0.9999,
                        help='Effective Number of Samples beta. 0=off, 0.9999=aggressive (default)')

    # ── AMP (mixed precision) ────────────────────────────────────────────────
    parser.add_argument('--amp',        action='store_true', default=False,
                        help='Enable Native AMP (float16) — speeds up Phase 1 inference ~2x')
    parser.add_argument('--bfloat',     action='store_true', default=False,
                        help='Use bfloat16 instead of float16 (more stable on Ampere+)')

    # ── EMA ──────────────────────────────────────────────────────────────────
    parser.add_argument('--model_ema',       action='store_true', default=False,
                        help='Enable EMA weights — val uses EMA model, usually ~0.5-1% better')
    parser.add_argument('--model_ema_decay', type=float, default=0.9998,
                        help='EMA decay (default 0.9998)')

    # ── Early stopping ───────────────────────────────────────────────────────
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Stop early if no improvement for N epochs. 0=off (default 10)')

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument('--seed',    type=int, default=42, help='Random seed (default 42)')
    parser.add_argument('--device',  default='cuda')

    # ── Live monitor ─────────────────────────────────────────────────────────
    parser.add_argument('--live_monitor',      action='store_true')
    parser.add_argument('--monitor_ws_port',   type=int, default=8765)
    parser.add_argument('--monitor_http_port', type=int, default=8766)
    parser.add_argument('--monitor_every',     type=int, default=5,
                        help='Push step to dashboard every N frames (default 5)')

    # ── Telegram ─────────────────────────────────────────────────────────────
    parser.add_argument('--telegram-token', default='', help='Telegram Bot Token')
    parser.add_argument('--telegram-group',  default='', help='Telegram Chat ID')

    args = parser.parse_args()

    # ── Seed ─────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── CUDA optimizations ───────────────────────────────────────────────
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True   # auto-tune convolution algorithms
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 cho matmul (Ampere+)
        torch.backends.cudnn.allow_tf32 = True         # TF32 for conv (Ampere+)

    # ── AMP setup ────────────────────────────────────────────────────────────
    use_amp = args.amp or args.bfloat
    amp_dtype = torch.bfloat16 if args.bfloat else torch.float16
    scaler = torch.cuda.amp.GradScaler() if (use_amp and not args.bfloat) else None
    if use_amp:
        print(f'⚡ AMP enabled: {amp_dtype}')


    # ── Live Monitor ─────────────────────────────────────────────────────────
    monitor = None
    if args.live_monitor:
        try:
            from live_monitor import LiveMonitor
            monitor = LiveMonitor(port=args.monitor_ws_port)
            monitor.start()
            monitor.start_http(http_port=args.monitor_http_port)
        except Exception as e:
            print(f'⚠️  LiveMonitor failed to start: {e}')
            monitor = None

    # Load splits
    with open(args.splits_json) as f:
        splits = json.load(f)
    train_embryos = splits.get('train', [])
    val_embryos   = splits.get('val', [])
    print(f'📋 Train: {len(train_embryos)} embryos | Val: {len(val_embryos)} embryos')

    # Load Phase 1 (frozen) — mandatory, no longer using cache-only path
    if not args.phase1_ckpt:
        print('❌ Must provide --phase1_ckpt')
        sys.exit(1)
    phase1 = load_phase1_model(args.phase1_ckpt, args.num_frames, device)

    # ── Cache Phase 1 features for all embryos (once only) ────────
    # Save to disk → next time load immediately, no need to rerun Phase 1.
    feat_cache_dir = Path(args.feat_cache_dir) if args.feat_cache_dir \
                     else output_dir / 'p1_cache'
    print(f'\n🔄 Phase 1 feature cache: {feat_cache_dir}')
    all_embryos = list(set(train_embryos + val_embryos))
    p1_feat_cache = cache_phase1_features(
        phase1, all_embryos,
        data_root=Path(args.data_root),
        ann_root=Path(args.ann_root),
        num_frames=args.num_frames,
        img_size=args.img_size,
        device=device,
        cache_dir=feat_cache_dir,
        p1_batch_size=args.p1_batch_size,
        num_workers=args.workers,
    )
    # Release Phase 1 from GPU — no longer needed
    phase1.cpu()
    del phase1
    torch.cuda.empty_cache()
    print('🗑️  Phase 1 model released from GPU')

    # Init Phase 2 — Causal Window Attention model
    import importlib.util as _ilu
    _p2_script = Path(__file__).parent / 'models' / 'ivf_phase2_cnn.py'
    _p2_spec = _ilu.spec_from_file_location('ivf_phase2_cnn', _p2_script)
    _p2_mod = _ilu.module_from_spec(_p2_spec)
    _p2_spec.loader.exec_module(_p2_mod)
    EmbryoTemporalNet = _p2_mod.EmbryoTemporalNet
    phase2 = EmbryoTemporalNet(
        num_classes=NUM_CLASSES,
        feat_dim=FEAT_DIM,
        d_model=args.d_model,
        d_state=args.d_state,
        expand=args.expand,
        headdim=args.headdim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        chunk_size=args.chunk_size,
        window_size=args.window_size,   # kept to calculate stage_dur
    ).to(device)
    trainable = sum(p.numel() for p in phase2.parameters() if p.requires_grad)
    print(f'🧠 Phase 2 EmbryoTemporalNet (Mamba2): {trainable/1e3:.1f}K trainable params '
          f'(d_model={args.d_model}, d_state={args.d_state}, '
          f'expand={args.expand}, headdim={args.headdim}, '
          f'nheads={int(args.d_model * args.expand / args.headdim)}, '
          f'n_layers={args.n_layers})')

    # ── EMA ──────────────────────────────────────────────────────────────
    model_ema = None
    if args.model_ema:
        from copy import deepcopy
        model_ema = deepcopy(phase2)
        model_ema.eval()
        for p in model_ema.parameters():
            p.requires_grad = False
        print(f'📊 EMA enabled (decay={args.model_ema_decay})')

    def update_ema(model, ema, decay):
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    # ── AdamW với no_weight_decay cho SSM params ─────────────────────────
    # dt_bias, A_log, D have attribute _no_weight_decay=True (Mamba2 convention)
    # AdamW does not automatically read this attribute -> must manually split into 2 param groups.
    decay_params     = []
    no_decay_params  = []
    for name, param in phase2.named_parameters():
        if not param.requires_grad:
            continue
        if getattr(param, '_no_weight_decay', False):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = AdamW(
        [
            {'params': decay_params,    'weight_decay': args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=args.lr,
    )
    print(f'   Optimizer: {len(decay_params)} params w/ wd={args.weight_decay}, '
          f'{len(no_decay_params)} params (dt_bias/A_log/D) w/ wd=0')

    # ── LR Scheduler: Linear warmup + Cosine decay ───────────────────────
    warmup_epochs = args.warmup_epochs
    def lr_lambda(epoch_0idx):
        if epoch_0idx < warmup_epochs:
            return (epoch_0idx + 1) / max(warmup_epochs, 1)
        progress = (epoch_0idx - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 1e-2 + 0.5 * (1 - 1e-2) * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    data_root = Path(args.data_root)
    ann_root  = Path(args.ann_root)

    # ── Loss function ────────────────────────────────────────────────────────
    # OrdinalProgressionLoss = CE + backward-regression penalty.
    # Penalize model when predicting a stage lower than GT (t5+ -> t2, t5+ -> tm+).
    # alpha=0.3: light penalty, does not overwhelm CE signal.

    # Calculate class weight from training set (inverse effective frequency)
    print('\n⚖️  Computing class weights from training set...')
    class_weight = compute_class_weight(
        train_embryos=train_embryos,
        data_root=data_root,
        ann_root=ann_root,
        num_classes=NUM_CLASSES,
        beta=args.class_weight_beta,
    ).to(device)

    # Import OrdinalProgressionLoss from model file
    criterion = _p2_mod.OrdinalProgressionLoss(
        num_classes=NUM_CLASSES,
        alpha=args.ordinal_alpha,
        forward_alpha=args.forward_alpha,
        max_forward_jump=args.max_forward_jump,
        label_smoothing=args.smoothing,
        class_weight=class_weight,
    ).to(device)
    criterion_eval = _p2_mod.OrdinalProgressionLoss(
        num_classes=NUM_CLASSES,
        alpha=0.0,
        forward_alpha=0.0,
        label_smoothing=0.0,
        class_weight=None,
    ).to(device)

    start_epoch = 1
    best_val_acc = 0.0
    best_epoch   = 0

    # Resume checkpoint if provided
    if args.resume and Path(args.resume).exists():
        print(f"🔄 Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        phase2.load_state_dict(ckpt['model_state_dict'])
        # Optimizer: skip if param groups count does not match
        _opt_ok = False
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                _opt_ok = True
            except ValueError as e:
                print(f"   ⚠️  Optimizer state skipped ({e})")
        # Scheduler: only load if optimizer OK (base_lrs depends on param groups)
        if _opt_ok and 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception as e:
                print(f"   ⚠️  Scheduler state skipped ({e})")
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('val_acc', 0.0)
        best_epoch   = ckpt.get('best_epoch', 0)
        print(f"   -> Resumed from epoch {start_epoch-1}, best_val_acc: {best_val_acc:.1f}%")

    tg_token = getattr(args, 'telegram_token', '')
    tg_chat  = getattr(args, 'telegram_group', '')

    # Notify training start
    send_telegram(tg_token, tg_chat,
        f'⚪️ <b>Phase 2 EmbryoTemporalNet (Mamba2) Training Start</b>\n'
        f'Model: MambaVision(frozen) → Mamba2 SSM (d_state={args.d_state}) → head\n'
        f'Epochs: {args.epochs} | LR: {args.lr} | d_model: {args.d_model}\n'
        f'TBPTT chunk: {args.accumulate_steps} frames | Augmentation: online\n'
        f'Loss: OrdinalProgressionLoss(alpha={args.ordinal_alpha}, smooth={args.smoothing})\n'
        f'Train: {len(train_embryos)} | Val: {len(val_embryos)} embryos')

    for epoch in range(start_epoch, args.epochs + 1):

        print(f'\n══ Epoch {epoch}/{args.epochs}  (Mamba2 SSM · d_state={args.d_state} · chunk={args.accumulate_steps}) ══')
        t0 = time.time()
        t_train_start = t0

        # --- Training (batched embryos for GPU utilization) ---
        phase2.train()
        random.shuffle(train_embryos)
        train_loss_total, train_frames_total = 0.0, 0
        train_grad_total = 0.0
        n_train = len(train_embryos)
        log_interval = max(1, n_train // 5)
        ebs = getattr(args, 'embryo_batch_size', 8)

        # Group embryos into mini-batches of size ebs
        embryo_batches = [train_embryos[i:i+ebs] for i in range(0, n_train, ebs)]
        n_batches = len(embryo_batches)

        for bi, batch_eids in enumerate(embryo_batches):
            loss, steps, grad_norm = train_batch_embryos(
                phase2, optimizer, criterion,
                batch_eids=batch_eids,
                p1_feat_cache=p1_feat_cache,
                device=device,
                accumulate_steps=args.accumulate_steps,
                clip_grad=args.clip_grad,
                is_training=not args.no_aug,
            )
            if model_ema is not None:
                update_ema(phase2, model_ema, args.model_ema_decay)
            train_loss_total  += loss * max(steps, 1)
            train_frames_total += steps
            train_grad_total  += grad_norm

            avg_so_far = train_loss_total / max(train_frames_total, 1)
            t_now      = time.time()
            elapsed_s  = t_now - t_train_start
            pct_done   = (bi + 1) / n_batches
            eta_s      = elapsed_s / pct_done * (1 - pct_done) if pct_done > 0 else 0

            if (bi + 1) % max(1, n_batches // 5) == 0 or (bi + 1) == n_batches:
                print(f'  Train [{bi+1:3d}/{n_batches}] {int(pct_done*100):3d}%'
                      f'  loss={loss:.4f}  avg={avg_so_far:.4f}'
                      f'  embryos={len(batch_eids)}  steps={steps}'
                      f'  elapsed={elapsed_s:5.0f}s  ETA={eta_s:5.0f}s')

        scheduler.step()
        avg_train_loss = train_loss_total / max(train_frames_total, 1)
        avg_grad_norm  = train_grad_total / max(n_train, 1)
        t_train_end = time.time()
        t_train_elapsed = t_train_end - t_train_start
        print(f'  ── Train done: avg_loss={avg_train_loss:.4f}  grad_norm={avg_grad_norm:.3f}  time={t_train_elapsed:.0f}s ──')

        # --- Validation ---
        t_val_start = time.time()
        # --- Validation: calculate average ACC per patient (macro-patient) ---
        phase2.eval()
        embryo_accs = []          # acc of each individual embryo
        embryo_top3s = []
        embryo_losses = []
        per_class_c = defaultdict(int)
        per_class_t = defaultdict(int)

        from tqdm import tqdm as _tqdm
        val_pbar = _tqdm(val_embryos, desc=f'  Val  E{epoch:02d}', unit='embryo', leave=True)
        for eid in val_pbar:
            frames = sorted(
                [p for p in (data_root / eid).glob('*')
                 if p.suffix.lower() in IMG_EXTS],
                key=_run_sort_key,
            )
            if not frames:
                continue
            phases = load_phases_csv(ann_root / f'{eid}_phases.csv')
            if not phases:
                continue
            labels = assign_labels(frames, phases)

            acc, top3, eloss, preds, gts = eval_one_embryo(
                None, phase2, frames, labels,
                args.num_frames, args.img_size, device,
                criterion=criterion_eval,
                embryo_id=eid,
                p1_batch_size=args.p1_batch_size,
                num_workers=args.workers,
                p1_cache=p1_feat_cache.get(eid),
            )

            # Only calculate acc if embryo has at least 1 frame with GT
            valid_pairs = [(p, g) for p, g in zip(preds, gts) if g is not None]
            if not valid_pairs:
                continue

            embryo_accs.append(acc)
            embryo_top3s.append(top3)
            embryo_losses.append(eloss)

            for p, g in valid_pairs:
                per_class_t[g] += 1
                if p == g:
                    per_class_c[g] += 1

        # Macro-patient accuracy = average acc of 70 embryos
        val_acc  = float(np.mean(embryo_accs))  if embryo_accs else 0.0
        val_top3 = float(np.mean(embryo_top3s)) if embryo_top3s else 0.0
        val_loss = float(np.mean(embryo_losses)) if embryo_losses else 0.0
        val_std  = float(np.std(embryo_accs))   if embryo_accs else 0.0

        # ── EMA validation ────────────────────────────────────────────────
        if model_ema is not None:
            ema_accs, ema_losses = [], []
            for eid in val_embryos[:min(len(val_embryos), 20)]:  # quick EMA check on 20 embryos
                frames_e = sorted([p for p in (data_root / eid).glob('*')
                                   if p.suffix.lower() in IMG_EXTS], key=_run_sort_key)
                if not frames_e: continue
                phases_e = load_phases_csv(ann_root / f'{eid}_phases.csv')
                if not phases_e: continue
                labels_e = assign_labels(frames_e, phases_e)
                a, _, el, _, _ = eval_one_embryo(
                    None, model_ema, frames_e, labels_e,
                    args.num_frames, args.img_size, device, criterion=criterion_eval,
                    p1_batch_size=args.p1_batch_size, num_workers=args.workers,
                    p1_cache=p1_feat_cache.get(eid))
                ema_accs.append(a); ema_losses.append(el)
            if ema_accs:
                ema_acc = float(np.mean(ema_accs))
                print(f'  │  EMA  Top1 : {ema_acc:.1f}% (quick, 20 embryos)')
                # Use EMA acc if better
                if ema_acc > val_acc:
                    val_acc = ema_acc
                    print(f'  │  → Using EMA acc for best model tracking')
        t_val_elapsed = time.time() - t_val_start
        elapsed       = time.time() - t0   # total epoch time

        # Save summary.csv file
        csv_path = output_dir / 'summary.csv'
        write_header = not csv_path.exists() or (epoch == start_epoch and epoch == 1)
        mode = 'w' if (epoch == start_epoch and epoch == 1) else 'a'
        with open(csv_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'train_loss', 'eval_loss', 'eval_top1', 'eval_top3'])
            writer.writerow([epoch, f'{avg_train_loss:.5f}', f'{val_loss:.5f}', f'{val_acc:.3f}', f'{val_top3:.3f}'])

        # Print to console
        print(f'\n  ┌─ Epoch {epoch}/{args.epochs} Summary ─────────────────────────────────')
        print(f'  │  Train Loss : {avg_train_loss:.4f}  grad_norm={avg_grad_norm:.3f}  ({t_train_elapsed:.0f}s)')
        print(f'  │  Val   Loss : {val_loss:.4f}  ({t_val_elapsed:.0f}s)')
        print(f'  │  Val   Top1 : {val_acc:.1f}% ± {val_std:.1f}%')
        print(f'  │  Val   Top3 : {val_top3:.1f}%')
        print(f'  │  Total time : {elapsed:.0f}s  |  LR: {scheduler.get_last_lr()[0]:.2e}')
        print(f'  └─ Per-class ──────────────────────────────────────────────────────────')
        per_class_lines = []
        for i, name in enumerate(STAGE_NAMES):
            t_cnt = per_class_t[i]
            c_cnt = per_class_c[i]
            pct = c_cnt / t_cnt * 100 if t_cnt > 0 else 0
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            line = f'    {name:<8}: {bar} {pct:5.1f}% ({c_cnt}/{t_cnt})'
            print(line)
            per_class_lines.append(f'{name}: {pct:.0f}%')

        # ── Push epoch summary lên Live Monitor ──────────────────────────
        if monitor is not None:
            monitor.push_epoch(
                epoch=epoch,
                total_epochs=args.epochs,
                train_loss=avg_train_loss,
                val_acc=val_acc,
                val_std=val_std,
                per_class={
                    STAGE_NAMES[i]: {
                        'correct': per_class_c[i],
                        'total':   per_class_t[i],
                        'acc': per_class_c[i] / per_class_t[i] * 100 if per_class_t[i] > 0 else 0,
                    }
                    for i in range(NUM_CLASSES)
                },
                elapsed=elapsed,
                window_size=args.window_size,
            )

        # Send Telegram message each epoch
        tg_msg = (
            f'🔄 <b>Epoch {epoch}/{args.epochs}</b>\n'
            f'📉 Loss: {avg_train_loss:.4f}\n'
            f'🎯 Val Acc (macro): <b>{val_acc:.1f}% ± {val_std:.1f}%</b>\n'
            f'⏱ Time: {elapsed:.0f}s\n'
            + '\n'.join(per_class_lines)
        )
        send_telegram(tg_token, tg_chat, tg_msg)

        # ── SAVE CHECKPOINTS ──────────────────────────────────────────────
        save_data = {
            'epoch': epoch,
            'model_state_dict': phase2.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'best_epoch': best_epoch,
            'embryo_accs': embryo_accs,
            'args': vars(args),
        }
        # 1. Save Last
        last_path = output_dir / 'phase2_last.pth'
        torch.save(save_data, last_path)

        # 2. Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_path = output_dir / 'phase2_best.pth'
            torch.save(save_data, best_path)
            print(f'  ✅ Best model saved → {best_path}  (val_acc={val_acc:.1f}%)')
            send_telegram(tg_token, tg_chat,
                f'✅ <b>New Best!</b> Epoch {epoch}\n'
                f'Val Acc: <b>{val_acc:.1f}%</b>')

        # ── Early stopping ────────────────────────────────────────────────
        if args.early_stopping_patience > 0:
            epochs_no_improve = epoch - best_epoch
            if epochs_no_improve >= args.early_stopping_patience:
                print(f'\n⏹ Early stopping: no improvement for {args.early_stopping_patience} epochs '
                      f'(best={best_val_acc:.1f}% @ epoch {best_epoch})')
                send_telegram(tg_token, tg_chat,
                    f'⏹ Early stopping @ epoch {epoch}\nBest: {best_val_acc:.1f}% @ epoch {best_epoch}')
                break

    # End
    final_msg = f'🎉 Training done!\nBest val acc: {best_val_acc:.1f}%'
    print(f'\n{final_msg}')
    send_telegram(tg_token, tg_chat, final_msg)


if __name__ == '__main__':
    main()
