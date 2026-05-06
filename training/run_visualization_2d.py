"""
run_visualization_2d.py — IVF 2D MambaVision Visualization Script
==================================================================
Run inference + create video + charts + CSV log for ivf_3d_mamba2 model
combined with Phase 2 CausalWindowAttention (ivf_phase2_cnn.py).

Usage:
  # Entire test split (Phase 1 only)
  python val/run_visualization_3d.py \\
      --model  output/best.pth \\
      --data_root  dataset/embryo_dataset \\
      --ann_root   dataset/embryo_dataset_annotations \\
      --splits_json processdata/splits.json \\
      --split test

  # Phase 1 + Phase 2 CausalWindowAttention
  python val/run_visualization_3d.py \\
      --model output/phase1_best.pth \\
      --phase2_ckpt output/phase2_attn_v2/phase2_best.pth \\
      --data_root dataset/embryo_dataset \\
      --ann_root  dataset/embryo_dataset_annotations \\
      --splits_json processdata/splits.json \\
      --split test

  # Single embryo
  python val/run_visualization_3d.py \\
      --model output/phase1_best.pth \\
      --phase2_ckpt output/phase2_attn_v2/phase2_best.pth \\
      --data_root dataset/embryo_dataset \\
      --ann_root  dataset/embryo_dataset_annotations \\
      --embryo_id AA83-7

  # Skip video (charts + CSV only)
  python val/run_visualization_3d.py ... --no_video
"""

import re
import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib

ImageFile.LOAD_TRUNCATED_IMAGES = True
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants — must match datasets_raw_ivf.py

STAGE_NAMES = ['1-tpnf', '2-t2', '3-t3+', '5-t5+', '7-t7+', '9-t9+', '10-tm+']
DISPLAY_NAMES = [s.split('-')[-1] for s in STAGE_NAMES]
NUM_CLASSES  = len(STAGE_NAMES)

RAW_TO_IDX = {
    # Class 0: tpnf — only tPNf (model not trained on tPNa)
    'tpnf': 0, 'tpn': 0, 't_pn': 0,
    # Class 1: t2
    't2':   1,
    # Class 2: t3+ (merged t3, t4)
    't3':   2, 't4':  2, 't3+': 2, 't4+': 2, 't3_t4': 2,
    # Class 3: t5+ (merged t5, t6)
    't5':   3, 't6':  3, 't5+': 3, 't6+': 3, 't5_t6': 3,
    # Class 4: t7+ (merged t7, t8)
    't7':   4, 't8':  4, 't7+': 4, 't8+': 4, 't7_t8': 4,
    # Class 5: t9+
    't9+':  5, 't9':  5,
    # Class 6: tm+ (merged tM, tSB)
    'tm':   6, 'tsb': 6, 'tm+': 6, 't_m': 6, 't_sb': 6,
}

# tpna excluded as the model was not trained on this stage
EXCLUDED_STAGES = {'tpb2', 'tpna', 'tb', 'teb', 'thb'}

# BGR colors for video panel
STAGE_COLORS_BGR = [
    (180, 180, 255),  # tpnf
    (100, 200, 255),  # t2
    (100, 255, 200),  # t3+
    (100, 255, 100),  # t5+
    (100, 200, 100),  # t7+
    (255, 200, 100),  # t9+
    (200, 100, 255),  # tM+
]

# Matplotlib colors
STAGE_COLORS_MPL = [
    '#b4b4ff', '#64c8ff', '#64ffc8',
    '#64ff64', '#64c864', '#ffc864', '#c864ff',
]

IMG_EXTS = {'.jpeg', '.jpg', '.png'}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sort_key(path):
    p = Path(path)
    m = re.search(r'RUN(\d+)', p.stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+', p.stem)
    return int(nums[-1]) if nums else 0


def apply_clahe_sobel(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE + Sobel — match datasets_raw_ivf.py."""
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


def load_ground_truth_7class(csv_path: Path, total_frames: int,
                              frame_files: list = None):
    """
    Parse *_phases.csv → list[int|None] per frame.
    Supports 2 formats:
      Format A (raw):    tPNf,114,120  → map via RAW_TO_IDX
      Format B (merged): 1-tPNf,114,120 → parse index from numeric prefix

    IMPORTANT: Use RUN number from filename (same as train_phase2.py),
    do not use sequential index. Annotation CSV uses 1-indexed RUN number.
    If frame_files=None → fallback to sequential index (backward compat).

    Returns (frame_labels, last_valid_frame) — last_valid_frame is 1-indexed.
    """
    frame_labels = [None] * total_frames
    last_valid_frame = 0

    if not csv_path.exists():
        return frame_labels, last_valid_frame

    # Parse CSV → dict {run_num: class_idx}
    MERGED_NAMES = {s: i for i, s in enumerate(STAGE_NAMES)}
    run_to_label = {}  # run_number → class_idx

    try:
        with open(csv_path, 'r') as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                stage_lo = row[0].strip().lower()
                try:
                    start = int(row[1].strip())
                    end   = int(row[2].strip())
                except ValueError:
                    continue

                # Format B: merged 7-class
                idx = MERGED_NAMES.get(stage_lo)
                if idx is None:
                    for sn in STAGE_NAMES:
                        if stage_lo == sn.lower():
                            idx = STAGE_NAMES.index(sn)
                            break
                # Format A: raw stages
                if idx is None:
                    if stage_lo in EXCLUDED_STAGES:
                        continue
                    idx = RAW_TO_IDX.get(stage_lo)
                if idx is None:
                    continue

                for fn in range(start, end + 1):
                    run_to_label[fn] = idx
    except Exception as e:
        print(f"  ⚠️ CSV error {csv_path}: {e}")
        return frame_labels, last_valid_frame

    if not run_to_label:
        return frame_labels, last_valid_frame

    if frame_files is not None:
        # Use RUN number from filename — same as train_phase2.py assign_labels()
        for i, fp in enumerate(frame_files):
            p = Path(fp)
            m = re.search(r'RUN(\d+)', p.stem, re.IGNORECASE)
            run_num = int(m.group(1)) if m else (i + 1)
            lbl = run_to_label.get(run_num)
            if lbl is not None:
                frame_labels[i] = lbl
                last_valid_frame = max(last_valid_frame, i + 1)
    else:
        # Fallback: sequential index (backward compat)
        for run_num, lbl in run_to_label.items():
            if 1 <= run_num <= total_frames:
                frame_labels[run_num - 1] = lbl
                last_valid_frame = max(last_valid_frame, run_num)

    return frame_labels, last_valid_frame


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_phase1_model(model_name: str, ckpt_path: str, device: str):
    """
    Load Phase 1 model (EmbryoMambaNet or EmbryoSwin++) using registry.
    """
    import sys
    from pathlib import Path
    root = Path(__file__).parent.parent
    sys.path.append(str(root / 'training'))
    
    from models.registry import create_model
    # Import to ensure models are registered
    import models
    
    # EmbryoSwin++ uses 7 classes for this dataset
    model = create_model(model_name, num_classes=NUM_CLASSES).to(device)
    
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt
    for key in ('state_dict', 'model', 'model_state_dict'):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    
    # Handle module. or backbone. prefix if present
    new_state = {}
    for k, v in state.items():
        name = k
        if name.startswith('module.'): name = name[7:]
        # For Swin++, checkpoint might store the entire model (backbone + heads)
        new_state[name] = v
        
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  ⚠️  Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    
    model.eval()
    print(f"✅ Phase1 ({model_name}) loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


def load_phase2_model(ckpt_path: str, device: str):
    """
    Load EmbryoTemporalNet (Phase 2) from ivf_phase2_cnn.py.
    """
    import importlib.util as _ilu
    root = Path(__file__).parent.parent
    p2_script = root / 'training' / 'models' / 'ivf_phase2_cnn.py'
    spec = _ilu.spec_from_file_location('ivf_phase2_cnn', p2_script)
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    EmbryoTemporalNet = module.EmbryoTemporalNet

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved_args = ckpt.get('args', {})
    state = ckpt.get('model_state_dict', ckpt)

    # ── Automatically detect expand and headdim from checkpoint shape ──────────
    # If saved_args lacks expand/headdim, infer from weight shape.
    # layers.0.mixer.dt_bias: shape (nheads,)
    # layers.0.mixer.in_proj.weight: shape (d_in_proj, d_model)
    #   d_in_proj = 2*d_inner + 2*ngroups*d_state + nheads
    #   d_inner = expand * d_model
    d_model  = saved_args.get('d_model', 640)
    d_state  = saved_args.get('d_state',  64)
    n_layers = saved_args.get('n_layers',  3)
    dropout  = saved_args.get('dropout', 0.1)

    expand  = saved_args.get('expand',  None)
    headdim = saved_args.get('headdim', None)

    if expand is None or headdim is None:
        # Infer from dt_bias shape → nheads, then calculate expand and headdim
        dt_key = 'layers.0.mixer.dt_bias'
        if dt_key in state:
            nheads_ckpt = state[dt_key].shape[0]
            # d_inner = nheads * headdim, try headdim=64 first
            for hd in [64, 32, 128]:
                d_inner_try = nheads_ckpt * hd
                exp_try = d_inner_try / d_model
                if exp_try == int(exp_try) and int(exp_try) >= 1:
                    expand  = int(exp_try)
                    headdim = hd
                    break
            if expand is None:
                expand  = 2   # fallback
                headdim = 64
            print(f"  ℹ️  Auto-detected: expand={expand}, headdim={headdim}, "
                  f"nheads={nheads_ckpt} (from checkpoint shape)")
        else:
            expand  = saved_args.get('expand',  2)
            headdim = saved_args.get('headdim', 64)

    model = EmbryoTemporalNet(
        num_classes = NUM_CLASSES,
        feat_dim    = d_model,
        d_model     = d_model,
        d_state     = d_state,
        n_layers    = n_layers,
        expand      = expand,
        headdim     = headdim,
        dropout     = dropout,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠️  Phase2 missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  ⚠️  Phase2 unexpected keys ({len(unexpected)}): {unexpected[:3]}")
    model.eval().to(device)
    print(f"✅ Phase2 (EmbryoTemporalNet) loaded: {ckpt_path}  "
          f"epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc',0):.1f}%")
    return model


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_embryo(model, phase2_model, data_root: Path, ann_root: Path, embryo_id: str,
                 num_frames: int, img_size: int, device: str, batch_size: int = 128):
    """
    Phase 1: 2D single-frame inference with EmbryoMambaNet + Grad-CAM.
    Phase 2: temporal refinement with EmbryoTemporalNet (optional).
    Returns dict or None if no annotation.
    """
    # --- Grad-CAM setup for Phase 1 ---
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        # Target layer: final BN norm of EmbryoMambaNet
        _cam_target = [model.norm]
        _cam = GradCAM(model=model, target_layers=_cam_target)
        _use_cam = True
    except Exception:
        _use_cam = False
        _cam = None
    embryo_dir = data_root / embryo_id
    if not embryo_dir.exists():
        print(f"  ⚠️ {embryo_id}: directory not found ({embryo_dir})")
        return None

    frame_files = sorted(
        [p for p in embryo_dir.iterdir() if p.suffix.lower() in IMG_EXTS],
        key=_run_sort_key,
    )
    if not frame_files:
        print(f"  ⚠️ {embryo_id}: no frames")
        return None

    n = len(frame_files)
    gt_labels, last_valid = load_ground_truth_7class(
        ann_root / f"{embryo_id}_phases.csv", n,
        frame_files=frame_files,   # passed to use RUN number from filename
    )
    if last_valid == 0:
        print(f"  ⚠️ {embryo_id}: no annotations")
        return None

    valid_files = frame_files[:last_valid]
    valid_gt    = gt_labels[:last_valid]

    # Preprocess all frames — NEED to apply CLAHE+Sobel 
    # because model was trained on preprocessed dataset_balanced_2d.
    processed = []
    for fp in tqdm(valid_files, desc=f"  Preprocess {embryo_id}", leave=False):
        img = np.array(Image.open(fp).convert('RGB'))
        # Apply CLAHE + Sobel to match training data
        img = apply_clahe_sobel(img)
        img = cv2.resize(img, (img_size, img_size))
        t = (img.astype(np.float32) / 255.0 - MEAN) / STD
        processed.append(t)

    N = len(processed)
    
    # ── 1. Batch Phase 1 Inference (Speed Optimization) ────────────────
    all_feats = []
    all_probs_p1 = []
    all_preds_p1 = []
    gradcam_list = []

    for i in tqdm(range(0, N, batch_size), desc=f"  Phase1 {embryo_id}", leave=False):
        batch_arrs = processed[i:i+batch_size]
        # batch_arrs: list of (H, W, 3) numpy arrays
        batch_t = torch.tensor(np.array(batch_arrs)).permute(0, 3, 1, 2).to(device)  # (B, 3, H, W)

        with torch.no_grad():
            if hasattr(model, 'backbone') and not hasattr(model, 'level_3'):
                # EmbryoSwin++ case
                feats = model.forward_features(batch_t)        # (B, 1536) already GAP
                out_p1 = model.cls_head(feats)                # (B, 7)
            else:
                # EmbryoMambaNet case
                feat_map = model.forward_features(batch_t)    # (B, 640, 7, 7)
                feat_bn  = model.norm(feat_map)              # (B, 640, 7, 7)
                feats    = model.avgpool(feat_bn).flatten(1) # (B, 640)
                out_p1   = model.head(feats)                  # (B, 7)
            
            probs_p1_np = F.softmax(out_p1, dim=1).cpu().numpy()
        preds_p1 = probs_p1_np.argmax(axis=1)

        all_feats.append(feats.clone())
        all_probs_p1.extend(probs_p1_np)
        all_preds_p1.extend(preds_p1)

        if _use_cam and _cam is not None:
            try:
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                img_t_req = batch_t.clone().requires_grad_(True)
                targets   = [ClassifierOutputTarget(int(p)) for p in preds_p1]
                gc_maps   = _cam(input_tensor=img_t_req, targets=targets)  # (B, H', W')
                for gc in gc_maps:
                    gc_rs = cv2.resize(gc, (img_size, img_size))
                    gradcam_list.append(gc_rs.astype(np.float32))
            except Exception as e:
                print(f"  [Grad-CAM Error]: {e}")
                gradcam_list.extend([None] * len(batch_arrs))
        else:
            gradcam_list.extend([None] * len(batch_arrs))

    all_feats_tensor = torch.cat(all_feats, dim=0)

    # ── 2. Sequential Phase 2 Inference ────────────────────────────────
    predictions, predictions_p1, confidences, probs_all, probs_p1_list = [], [], [], [], []
    attn_weights_list = []

    # Initialize cache for Phase 2
    kv_cache = phase2_model.make_initial_hidden(1, device) if phase2_model is not None else None

    # stage_dur tracking (using P1 argmax — causal, not GT)
    _p1_prev_stage, _p1_run = None, 0
    window_size = getattr(phase2_model, 'window_size', 128) if phase2_model is not None else 128

    for fi in tqdm(range(N), desc=f"  Phase2 {embryo_id}", leave=False):
        feat = all_feats_tensor[fi:fi+1]
        probs_p1_np = all_probs_p1[fi]
        pred_p1 = int(all_preds_p1[fi])

        # ── Phase 2 (optional) ────────────────────────────────────────────
        if phase2_model is not None:
            if pred_p1 == _p1_prev_stage:
                _p1_run += 1
            else:
                _p1_run  = 1
                _p1_prev_stage = pred_p1
            stage_dur    = min(_p1_run / max(window_size, 1), 2.0)
            temporal_pos = torch.tensor([[stage_dur]], dtype=torch.float32, device=device)

            # Hook into CausalAttnBuffer (Block[2]) to get actual attention weights
            _attn_captured = []
            def _attn_hook(module, inp, out):
                # out = (output_tensor, kv_buffer_new)
                # Calculate attention weights from Q and K in buffer
                try:
                    x_in = inp[0]          # (B, 1, d_model)
                    kv_buf = inp[1]        # (B, W, d_model) or None
                    B_, L_, C_ = x_in.shape
                    ctx = x_in if kv_buf is None else torch.cat([kv_buf, x_in], dim=1)
                    if ctx.shape[1] > module.window:
                        ctx = ctx[:, -module.window:, :]
                    Lctx = ctx.shape[1]
                    qkv_x   = module.qkv(x_in)
                    qkv_ctx = module.qkv(ctx)
                    q = qkv_x[:, :, :C_].reshape(B_, 1, module.num_heads, module.head_dim).transpose(1, 2)
                    k = qkv_ctx[:, :, C_:2*C_].reshape(B_, Lctx, module.num_heads, module.head_dim).transpose(1, 2)
                    # Scaled dot-product → attention scores (B, heads, 1, Lctx)
                    scores = torch.matmul(q, k.transpose(-2, -1)) * (module.head_dim ** -0.5)
                    aw = torch.softmax(scores, dim=-1)  # (B, heads, 1, Lctx)
                    aw_mean = aw[0, :, 0, :].mean(0).cpu().numpy()  # (Lctx,)
                    _attn_captured.append(aw_mean)
                except Exception:
                    pass

            # Find CausalAttnBuffer in Phase 2 model (last block)
            _hook_handle = None
            for layer in phase2_model.layers:
                if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'qkv'):
                    _hook_handle = layer.mixer.register_forward_hook(_attn_hook)
                    break

            with torch.no_grad():
                result     = phase2_model(feat, None, kv_cache,
                                          frame_idx=fi, temporal_pos=temporal_pos)
                logits, kv_cache, _ = result[0], result[1], result[2]
                probs_np = F.softmax(logits, dim=1)[0].cpu().numpy()

            if _hook_handle is not None:
                _hook_handle.remove()

            aw = _attn_captured[0] if _attn_captured else None
            attn_weights_list.append(aw)
        else:
            probs_np = probs_p1_np
            attn_weights_list.append(None)

        pred = int(probs_np.argmax())
        predictions.append(pred)
        predictions_p1.append(pred_p1)
        confidences.append(float(probs_np[pred]))
        probs_all.append(probs_np)
        probs_p1_list.append(probs_p1_np)

    return {
        'embryo_id':      embryo_id,
        'predictions':    predictions,
        'predictions_p1': predictions_p1,
        'probs_p1_all':   np.array(probs_p1_list) if probs_p1_list else None,
        'ground_truth':   valid_gt,
        'confidences':    confidences,
        'probs_all':      np.array(probs_all),
        'frame_files':    valid_files,
        'attn_weights':   attn_weights_list,
        'gradcam':        gradcam_list,   # list[np.ndarray H×W | None]
    }



# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def create_video(result: dict, output_path: str, fps: int = 10):
    """Video: original frame + side panel (pred, GT, correct/incorrect, running acc, prob bars)."""
    eid         = result['embryo_id']
    predictions = result['predictions']
    ground_truth= result['ground_truth']
    confidences = result['confidences']
    probs_all   = result['probs_all']
    frame_files = result['frame_files']

    frames_bgr = []
    for fp in tqdm(frame_files, desc=f"  Load {eid}", leave=False):
        img = cv2.imread(str(fp))
        if img is not None:
            frames_bgr.append(img)
    if not frames_bgr:
        return

    orig_h, orig_w = frames_bgr[0].shape[:2]
    panel_w = 360
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(output_path, fourcc, fps, (orig_w + panel_w, orig_h))

    running_correct = 0
    for i, (frame_bgr, pred, gt, conf) in enumerate(
        tqdm(list(zip(frames_bgr, predictions, ground_truth, confidences)),
             desc=f"  Write {eid}", leave=False)
    ):
        is_correct = (pred == gt) if gt is not None else None
        if is_correct:
            running_correct += 1
        run_acc = running_correct / (i + 1) * 100

        frame = frame_bgr.copy()
        panel = np.full((orig_h, panel_w, 3), 28, dtype=np.uint8)

        # Header + timeline bar
        cv2.rectangle(panel, (0, 0), (panel_w, 44), (50, 50, 50), -1)
        cv2.putText(panel, f"IVF  {eid}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        seg_w = (panel_w - 16) // NUM_CLASSES
        for k in range(NUM_CLASSES):
            x1 = 8 + k * seg_w
            color = STAGE_COLORS_BGR[k] if (gt is not None and k == gt) else (45, 45, 45)
            cv2.rectangle(panel, (x1, 32), (x1 + seg_w, 40), color, -1)
            if k == pred:
                cv2.rectangle(panel, (x1, 32), (x1 + seg_w, 40), (255, 255, 255), 1)

        cv2.putText(panel, f"Frame {i+1}/{len(frames_bgr)}", (8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)
        cv2.line(panel, (8, 67), (panel_w - 8, 67), (70, 70, 70), 1)

        # AI Prediction
        cv2.putText(panel, "AI PREDICTION", (8, 87),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 180, 80), 2)
        pc = STAGE_COLORS_BGR[pred]
        cv2.rectangle(panel, (8, 93), (panel_w - 8, 122), pc, -1)
        cv2.putText(panel, STAGE_NAMES[pred], (14, 114),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
        cv2.putText(panel, f"Conf: {conf*100:.0f}%", (8, 139),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)
        bw = int((panel_w - 16) * conf)
        cv2.rectangle(panel, (8, 144), (panel_w - 8, 152), (55, 55, 55), -1)
        cv2.rectangle(panel, (8, 144), (8 + bw, 152), pc, -1)
        cv2.line(panel, (8, 161), (panel_w - 8, 161), (70, 70, 70), 1)

        # Ground Truth
        cv2.putText(panel, "GROUND TRUTH", (8, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 220, 80), 2)
        if gt is not None:
            gc = STAGE_COLORS_BGR[gt]
            cv2.rectangle(panel, (8, 186), (panel_w - 8, 215), gc, -1)
            cv2.putText(panel, STAGE_NAMES[gt], (14, 207),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
        else:
            cv2.putText(panel, "N/A", (14, 207),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 120, 120), 1)
        cv2.line(panel, (8, 224), (panel_w - 8, 224), (70, 70, 70), 1)

        # Result
        if is_correct is True:
            cv2.rectangle(panel, (8, 230), (panel_w - 8, 262), (40, 140, 40), -1)
            cv2.putText(panel, "CORRECT", (48, 252),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
        elif is_correct is False:
            cv2.rectangle(panel, (8, 230), (panel_w - 8, 262), (40, 40, 140), -1)
            cv2.putText(panel, "INCORRECT", (28, 252),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
        cv2.line(panel, (8, 271), (panel_w - 8, 271), (70, 70, 70), 1)

        # Running accuracy
        cv2.putText(panel, "RUNNING ACC", (8, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        acc_c = (40, 200, 40) if run_acc > 70 else (40, 200, 200) if run_acc > 50 else (40, 40, 200)
        cv2.putText(panel, f"{run_acc:.1f}%", (8, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, acc_c, 2)
        abw = int((panel_w - 16) * run_acc / 100)
        cv2.rectangle(panel, (8, 327), (panel_w - 8, 337), (55, 55, 55), -1)
        cv2.rectangle(panel, (8, 327), (8 + abw, 337), acc_c, -1)
        cv2.putText(panel, f"OK: {running_correct}/{i+1}", (8, 355),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1)

        # Prob bars (if space permits)
        if orig_h > 400:
            cv2.line(panel, (8, 368), (panel_w - 8, 368), (70, 70, 70), 1)
            probs = probs_all[i]
            row_h = min(20, (orig_h - 375) // NUM_CLASSES)
            for k, (sn, p) in enumerate(zip(STAGE_NAMES, probs)):
                y0 = 375 + k * row_h
                if y0 + row_h > orig_h - 4:
                    break
                blen = int((panel_w - 68) * p)
                cv2.rectangle(panel, (48, y0 + 2), (48 + blen, y0 + row_h - 2),
                               STAGE_COLORS_BGR[k], -1)
                cv2.putText(panel, sn, (2, y0 + row_h - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 170, 170), 1)
                cv2.putText(panel, f"{p*100:.0f}%", (panel_w - 58, y0 + row_h - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (170, 170, 170), 1)

        # Overlay on original frame
        ov_c = (0, 255, 0) if is_correct else (0, 0, 255) if is_correct is False else (200, 200, 0)
        cv2.rectangle(frame, (4, 4), (orig_w - 4, 46), (0, 0, 0), -1)
        cv2.rectangle(frame, (4, 4), (orig_w - 4, 46), ov_c, 2)
        cv2.putText(frame, f"Pred: {STAGE_NAMES[pred]}  ({conf*100:.0f}%)",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, ov_c, 2)

        writer.write(np.hstack([frame, panel]))

    writer.release()
    print(f"  🎬 Video → {output_path}")


# ---------------------------------------------------------------------------
# Per-embryo chart (Confusion Matrix Only)
# ---------------------------------------------------------------------------

def create_embryo_chart(result: dict, output_path: str):
    """
    Plot Confusion Matrix for each embryo (P2 vs Delta).
    Focus on Confusion Matrix, making it aesthetically pleasing and clear.
    """
    gts_all = result['ground_truth']
    preds_all = result['predictions']
    preds_p1_all = result.get('predictions_p1')
    eid = result['embryo_id']

    # Filter valid frames
    valid_idx = [i for i, g in enumerate(gts_all) if g is not None]
    if not valid_idx:
        return

    gts   = [gts_all[i] for i in valid_idx]
    preds = [preds_all[i] for i in valid_idx]

    fig, ax = plt.subplots(figsize=(8, 7))

    try:
        from sklearn.metrics import confusion_matrix as _cm
        import seaborn as sns
        
        cm_p2 = _cm(gts, preds, labels=list(range(NUM_CLASSES)))
        
        if preds_p1_all is not None:
            preds_p1 = [preds_p1_all[i] for i in valid_idx]
            cm_p1 = _cm(gts, preds_p1, labels=list(range(NUM_CLASSES)))
            
            # Annotation: P2 count (delta vs P1 in parentheses)
            annot = np.empty_like(cm_p2, dtype=object)
            for i in range(NUM_CLASSES):
                for j in range(NUM_CLASSES):
                    val = int(cm_p2[i, j])
                    d = val - int(cm_p1[i, j])
                    if val == 0 and d == 0:
                        annot[i, j] = '0'  # Simplify 0 (0) to 0
                    else:
                        sign = f'+{d}' if d > 0 else str(d)
                        annot[i, j] = f'{val}\n({sign})'
            
            sns.heatmap(cm_p2, annot=annot, fmt='', cmap='Blues',
                        xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                        ax=ax, linewidths=1.0, linecolor='white', cbar=True,
                        cbar_kws={'shrink': 0.8},
                        annot_kws={'size': 11, 'weight': 'bold'})
            ax.set_title(f"Embryo {eid} — Phase 2 Confusion\n(Δ vs Phase 1 in parentheses)", 
                         pad=15, fontsize=12, fontweight='bold')
        else:
            sns.heatmap(cm_p2, annot=True, fmt='d', cmap='Blues',
                        xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                        ax=ax, linewidths=1.0, linecolor='white', cbar=True,
                        cbar_kws={'shrink': 0.8},
                        annot_kws={'size': 11, 'weight': 'bold'})
            ax.set_title(f"Embryo {eid} — Confusion Matrix (Phase 2)", 
                         pad=15, fontsize=12, fontweight='bold')
            
        ax.set_xlabel('Predicted Stage', fontsize=11, labelpad=10)
        ax.set_ylabel('True Stage', fontsize=11, labelpad=10)
        ax.tick_params(axis='both', labelsize=10)
    except ImportError:
        ax.text(0.5, 0.5, 'Requires scikit-learn & seaborn',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Confusion Matrix → {output_path}")


# Aggregate charts — for Q1 paper
# ---------------------------------------------------------------------------

def create_aggregate_charts(all_results: list, output_dir: Path, split: str):
    """
    4-panel aggregate — compare P1 vs P2:
      1. Normalised confusion matrix P2
      2. Per-class F1: P1 vs P2 side-by-side
      3. Confidence distribution P2 (correct vs wrong)
      4. Per-embryo accuracy scatter P1 vs P2
    """
    try:
        from sklearn.metrics import (
            confusion_matrix, classification_report, f1_score
        )
        import seaborn as sns
    except ImportError:
        print("⚠️  sklearn / seaborn not installed — skipping aggregate charts")
        return

    all_preds, all_preds_p1, all_gts = [], [], []
    all_confs = []
    for r in all_results:
        preds_p1_r = r.get('predictions_p1', r['predictions'])
        for p2, p1, g, c in zip(r['predictions'], preds_p1_r,
                                 r['ground_truth'], r['confidences']):
            if g is not None:
                all_preds.append(p2)
                all_preds_p1.append(p1)
                all_gts.append(g)
                all_confs.append(c)

    if not all_preds:
        return

    p1_acc_total = sum(p == g for p, g in zip(all_preds_p1, all_gts)) / len(all_gts) * 100
    p2_acc_total = sum(p == g for p, g in zip(all_preds,    all_gts)) / len(all_gts) * 100

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f'Aggregate Results — split={split}  '
        f'(n={len(all_results)} embryos, {len(all_preds)} frames)\n'
        f'EmbryoMambaNet P1: {p1_acc_total:.2f}%  →  '
        f'EmbryoTemporalNet P2: {p2_acc_total:.2f}%  '
        f'(Δ={p2_acc_total - p1_acc_total:+.2f}%)',
        fontsize=13, fontweight='bold'
    )

    # 1. Confusion matrix P2
    ax = axes[0, 0]
    cm = confusion_matrix(all_gts, all_preds,
                          labels=list(range(NUM_CLASSES)), normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                ax=ax, linewidths=0.4, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'P2 Normalised Confusion Matrix (Acc={p2_acc_total:.2f}%)')

    # 2. Per-class F1: P1 vs P2 side-by-side
    ax = axes[0, 1]
    report_p1 = classification_report(
        all_gts, all_preds_p1, labels=list(range(NUM_CLASSES)),
        target_names=DISPLAY_NAMES, output_dict=True, zero_division=0
    )
    report_p2 = classification_report(
        all_gts, all_preds, labels=list(range(NUM_CLASSES)),
        target_names=DISPLAY_NAMES, output_dict=True, zero_division=0
    )
    f1s_p1 = [report_p1[s]['f1-score'] for s in DISPLAY_NAMES]
    f1s_p2 = [report_p2[s]['f1-score'] for s in DISPLAY_NAMES]
    x = np.arange(NUM_CLASSES); w = 0.35
    bars_p1 = ax.bar(x - w/2, f1s_p1, w, label='P1 (EmbryoMambaNet)',
                     color='#e67e22', alpha=0.75)
    bars_p2 = ax.bar(x + w/2, f1s_p2, w, label='P2 (EmbryoTemporalNet)',
                     color='#3498db', alpha=0.85)
    # Mark improvement
    for i, (v1, v2) in enumerate(zip(f1s_p1, f1s_p2)):
        delta = v2 - v1
        color = '#27ae60' if delta > 0.01 else '#e74c3c' if delta < -0.01 else '#888'
        ax.text(x[i] + w/2, v2 + 0.01, f'{delta:+.2f}',
                ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(DISPLAY_NAMES, rotation=30, ha='right')
    ax.set_ylim(0, 1.18); ax.set_ylabel('F1 Score')
    macro_f1_p1 = f1_score(all_gts, all_preds_p1, average='macro', zero_division=0)
    macro_f1_p2 = f1_score(all_gts, all_preds,    average='macro', zero_division=0)
    ax.set_title('Per-Class F1: P1 vs P2')
    ax.set_xlabel(f'Macro-F1: P1={macro_f1_p1:.3f}  P2={macro_f1_p2:.3f}  '
                  f'(Δ={macro_f1_p2 - macro_f1_p1:+.3f})')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25)

    # 3. Confidence distribution P2
    ax = axes[1, 0]
    c_ok  = [c for c, p, g in zip(all_confs, all_preds, all_gts) if p == g]
    c_err = [c for c, p, g in zip(all_confs, all_preds, all_gts) if p != g]
    ax.hist(c_ok,  bins=25, alpha=0.65, color='#2ecc71',
            label=f'P2 Correct ({len(c_ok)})',   density=True)
    ax.hist(c_err, bins=25, alpha=0.65, color='#e74c3c',
            label=f'P2 Incorrect ({len(c_err)})', density=True)
    ax.set_xlabel('P2 Confidence'); ax.set_ylabel('Density')
    ax.set_title('P2 Confidence Distribution (All Embryos)')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # 4. Per-embryo accuracy scatter P1 vs P2
    ax = axes[1, 1]
    emb_p1_accs, emb_p2_accs = [], []
    for r in all_results:
        preds_p1_r = r.get('predictions_p1', r['predictions'])
        valid = [(p1, p2, g) for p1, p2, g in zip(
            preds_p1_r, r['predictions'], r['ground_truth']) if g is not None]
        if valid:
            p1_a = sum(p1 == g for p1, p2, g in valid) / len(valid) * 100
            p2_a = sum(p2 == g for p1, p2, g in valid) / len(valid) * 100
            emb_p1_accs.append(p1_a)
            emb_p2_accs.append(p2_a)

    if emb_p1_accs:
        colors_emb = ['#2ecc71' if p2 > p1 else '#e74c3c' if p2 < p1 else '#888'
                      for p1, p2 in zip(emb_p1_accs, emb_p2_accs)]
        ax.scatter(emb_p1_accs, emb_p2_accs, c=colors_emb, s=40,
                   alpha=0.7, edgecolors='white', lw=0.4)
        lim = max(max(emb_p1_accs), max(emb_p2_accs)) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.3, label='P1 = P2')
        improved = sum(1 for p1, p2 in zip(emb_p1_accs, emb_p2_accs) if p2 > p1)
        ax.set_xlabel('P1 Accuracy per Embryo (%)')
        ax.set_ylabel('P2 Accuracy per Embryo (%)')
        ax.set_title(f'Per-Embryo: P1 vs P2 Accuracy\n'
                     f'{improved}/{len(emb_p1_accs)} embryos improved by P2 '
                     f'(green=better, red=worse)')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend(fontsize=9); ax.grid(alpha=0.15)
        ax.fill_between([0, lim], [0, lim], [0, 0], alpha=0.04, color='#2ecc71')
        ax.text(lim * 0.7, lim * 0.15, 'P2 better', fontsize=10,
                color='#27ae60', alpha=0.5, ha='center')

    plt.tight_layout()
    out_path = output_dir / f'aggregate_{split}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Aggregate chart → {out_path}")

    # ---- Phase 1 aggregation ----
    all_preds_p1, all_probs_p1 = [], []
    for r in all_results:
        preds_p1_r = r.get('predictions_p1', r['predictions'])
        probs_p1_r = r.get('probs_p1_all')
        for i, g in enumerate(r['ground_truth']):
            if g is not None:
                all_preds_p1.append(preds_p1_r[i])
                if probs_p1_r is not None:
                    all_probs_p1.append(probs_p1_r[i])

    # ---- Accuracy ----
    overall_acc = sum(p == g for p, g in zip(all_preds, all_gts)) / len(all_preds) * 100
    p1_acc = sum(p == g for p, g in zip(all_preds_p1, all_gts)) / len(all_preds_p1) * 100 if all_preds_p1 else 0

    # ---- Top-3 ----
    all_probs_p2 = []
    for r in all_results:
        for i, g in enumerate(r['ground_truth']):
            if g is not None:
                all_probs_p2.append(r['probs_all'][i])

    top3_p2 = sum(g in np.argsort(p)[::-1][:3] for g, p in zip(all_gts, all_probs_p2)) / len(all_gts) * 100
    if all_probs_p1:
        top3_p1 = sum(g in np.argsort(p)[::-1][:3] for g, p in zip(all_gts, all_probs_p1)) / len(all_gts) * 100
    else:
        top3_p1 = 0.0

    # ---- Confusion matrix Phase 1 ----
    cm_p1 = confusion_matrix(all_gts, all_preds_p1, labels=list(range(NUM_CLASSES)), normalize='true')
    fig_cm1, ax_cm1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_p1, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
                ax=ax_cm1, linewidths=0.4, cbar_kws={'shrink': 0.8})
    ax_cm1.set_xlabel('Predicted'); ax_cm1.set_ylabel('True')
    ax_cm1.set_title(f'Phase 1 Normalised Confusion Matrix (Acc={p1_acc:.2f}%)')
    plt.tight_layout()
    cm1_path = output_dir / f'confusion_matrix_phase1_{split}.png'
    plt.savefig(cm1_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"📊 Phase 1 Confusion Matrix → {cm1_path}")

    # ---- Classification reports ----
    _labels = list(range(NUM_CLASSES))
    with open(output_dir / f'classification_report_phase1_{split}.txt', 'w') as f:
        f.write(classification_report(all_gts, all_preds_p1,
                labels=_labels, target_names=STAGE_NAMES, zero_division=0))
    with open(output_dir / f'classification_report_phase2_{split}.txt', 'w') as f:
        f.write(classification_report(all_gts, all_preds,
                labels=_labels, target_names=STAGE_NAMES, zero_division=0))
    print(f"📝 Reports → {output_dir}/classification_report_phaseX_{split}.txt")

    from sklearn.metrics import f1_score as f1
    w_f1_p1 = f1(all_gts, all_preds_p1, average='weighted', zero_division=0)
    m_f1_p1 = f1(all_gts, all_preds_p1, average='macro',    zero_division=0)
    w_f1_p2 = f1(all_gts, all_preds,    average='weighted', zero_division=0)
    m_f1_p2 = f1(all_gts, all_preds,    average='macro',    zero_division=0)
    f1s_p2  = f1(all_gts, all_preds,    average=None,       zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"  Split: {split}  |  Embryos: {len(all_results)}  |  Frames: {len(all_preds)}")
    print(f"  {'Metric':<22} {'Phase 1':>12} {'Phase 2':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Top-1 Accuracy':<22} {p1_acc:>11.2f}% {overall_acc:>11.2f}%")
    print(f"  {'Top-3 Accuracy':<22} {top3_p1:>11.2f}% {top3_p2:>11.2f}%")
    print(f"  {'Macro F1':<22} {m_f1_p1:>12.4f} {m_f1_p2:>12.4f}")
    print(f"  {'Weighted F1':<22} {w_f1_p1:>12.4f} {w_f1_p2:>12.4f}")
    print(f"  {'-'*46}")
    print(f"  Per-class F1 (Phase 2):")
    for s, fv in zip(STAGE_NAMES, f1s_p2):
        print(f"    {s:<6}: {fv:.3f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Chart A: Per-Embryo Attention Heatmap — Block[2] CausalAttnBuffer
# Phase 2: Mamba2[0]+MLP → Mamba2[1]+MLP → Attention[2]+MLP
# Block[2] is CausalAttnBuffer: attends W=128 most recent frames (local refinement)
# Attention weights are hooked directly from CausalAttnBuffer in infer_embryo().
# ---------------------------------------------------------------------------

def create_attention_heatmap(result: dict, output_path: str):
    """
    3-panel figure for 1 embryo:

    Top panel: 2D Attention heatmap (Block[2] CausalAttnBuffer)
      - X-axis = frame (time)
      - Y-axis = position in KV buffer (0=oldest, W-1=current)
      - Color = attention weight (mean over heads)
      Shows: Does the attention block attend nearby or distant frames? At transitions,
      does the attention spread wider?
      If no attention data → plots P2 probability heatmap instead.

    Middle panel: P1 vs P2 prediction timeline + GT

    Bottom panel: P2 confidence over time (color = correct/wrong)
    """
    attn_list = result.get('attn_weights', [])
    gts       = result['ground_truth']
    preds     = result['predictions']
    preds_p1  = result.get('predictions_p1', preds)
    probs_p2  = result['probs_all']
    confs     = result['confidences']
    eid       = result['embryo_id']

    valid_idx = [i for i, g in enumerate(gts) if g is not None]
    if len(valid_idx) < 5:
        return

    # Filter frames with actual attention weights (from CausalAttnBuffer hook)
    attn_valid = [(i, attn_list[i]) for i in valid_idx
                  if i < len(attn_list) and attn_list[i] is not None]
    has_attn = len(attn_valid) >= 5

    N = len(valid_idx)
    frames_x = list(range(N))
    gts_v   = [gts[i]    for i in valid_idx]
    preds_v = [preds[i]  for i in valid_idx]
    p1s_v   = [preds_p1[i] for i in valid_idx]
    confs_v = [confs[i]  for i in valid_idx]

    n_ticks   = min(20, N)
    tick_step = max(1, N // n_ticks)
    tick_pos  = list(range(0, N, tick_step))

    n_rows = 3 if has_attn else 2
    height_ratios = [2.5, 1.2, 0.8] if has_attn else [1.5, 1.0]
    figsize = (18, 12) if has_attn else (18, 8)

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize,
                             gridspec_kw={'height_ratios': height_ratios})
    
    # Ensure axes is iterable even if 1 row (though min is 2 here)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    fig.suptitle(
        f'Phase 2 Analysis — {eid}  ({N} valid frames)\n'
        f'Block[2] CausalAttnBuffer (W=128) + Mamba2[0,1] SSM state',
        fontsize=12, fontweight='bold', y=1.01
    )

    ax_idx = 0
    if has_attn:
        # ── Panel 0: Attention heatmap ──────────────
        ax = axes[ax_idx]
        max_w    = max(len(aw) for _, aw in attn_valid)
        n_frames = len(attn_valid)
        heatmap  = np.zeros((max_w, n_frames))
        frame_indices = []
        for col, (fi, aw) in enumerate(attn_valid):
            w = len(aw)
            heatmap[max_w - w:, col] = aw
            frame_indices.append(fi)

        im = ax.imshow(heatmap, aspect='auto', cmap='inferno',
                       interpolation='nearest', origin='lower')
        ax.set_ylabel('KV Buffer Position\n(bottom=oldest → top=current)')

        # Mark transition points
        gt_at_valid = [gts[fi] for fi in frame_indices]
        for col in range(1, n_frames):
            g_prev, g_cur = gt_at_valid[col - 1], gt_at_valid[col]
            if g_prev is not None and g_cur is not None and g_prev != g_cur:
                ax.axvline(col, color='cyan', lw=1.5, ls='--', alpha=0.8)

        attn_tick_step = max(1, n_frames // min(20, n_frames))
        attn_tick_pos  = list(range(0, n_frames, attn_tick_step))
        ax.set_xticks(attn_tick_pos)
        ax.set_xticklabels([frame_indices[t] for t in attn_tick_pos],
                           fontsize=6, rotation=45)
        ax.set_xlabel('Frame Index')
        plt.colorbar(im, ax=ax, label='Attention Weight (mean over heads)',
                     shrink=0.8, pad=0.01)
        ax.set_title(
            'Block[2] CausalAttnBuffer — Attention Weights over KV Buffer\n'
            '(cyan dashes = GT stage transitions)',
            fontsize=9
        )
        ax_idx += 1

    # ── Panel 1: Prediction timeline P1 vs P2 vs GT ───────────────────────
    ax = axes[ax_idx]
    ax.step(frames_x, gts_v,   where='post', color='#2ecc71', lw=2.5)
    ax.step(frames_x, p1s_v,   where='post', color='#e67e22', lw=1.2, ls=':',  alpha=0.8)
    ax.step(frames_x, preds_v, where='post', color='#3498db', lw=1.8)
    p1_acc = sum(p == g for p, g in zip(p1s_v, gts_v)) / N * 100
    p2_acc = sum(p == g for p, g in zip(preds_v, gts_v)) / N * 100
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(DISPLAY_NAMES, fontsize=8)
    ax.set_ylim(-0.5, NUM_CLASSES - 0.5)
    ax.set_ylabel('Stage')
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color='#2ecc71', lw=2.5),
            plt.Line2D([0], [0], color='#e67e22', lw=1.2, ls=':'),
            plt.Line2D([0], [0], color='#3498db', lw=1.8),
        ],
        labels=[f'GT', f'P1 ({p1_acc:.1f}%)', f'P2 ({p2_acc:.1f}%)'],
        fontsize=8, loc='upper left'
    )
    ax.grid(alpha=0.2)
    ax.set_title('Prediction Timeline: P1 vs P2 vs GT', fontsize=9)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([valid_idx[t] for t in tick_pos], fontsize=6, rotation=45)

    ax_idx += 1

    # ── Panel 2: Confidence ───────────────────────────────────────────────
    ax = axes[ax_idx]
    colors_pts = ['#2ecc71' if p == g else '#e74c3c'
                  for p, g in zip(preds_v, gts_v)]
    ax.scatter(frames_x, confs_v, c=colors_pts, s=10, zorder=3, alpha=0.7)
    ax.plot(frames_x, confs_v, color='#9b59b6', lw=0.8, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('P2 Confidence')
    ax.set_xlabel('Frame Index')
    ax.grid(alpha=0.2)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([valid_idx[t] for t in tick_pos], fontsize=6, rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Attention Heatmap → {output_path}")


# ---------------------------------------------------------------------------
# Chart A2: Phase 2 Correction Explainer (per-embryo, paper-ready)
# Replaces old attention explainer — Phase 2 uses Mamba2 SSM, no
# attention weights. Replaced with: P1 vs P2 probability comparison at
# interesting frames (transition, error correction, high/low confidence).
# ---------------------------------------------------------------------------

def create_phase2_attention_explainer(result: dict, output_dir: str,
                                      sample_frames: list = None):
    """
    Paper-ready figure for 1 embryo: explains how Phase 2 corrects Phase 1 errors.
    
    Choose 3-5 "interesting" frames and plot for each:
      Row 0: Original embryo image + GT/P1/P2 annotation
      Row 1: P1 vs P2 probability bar comparison (7 classes)
      Row 2: Confidence gauge + correct/wrong indicator
      Row 3: Timeline context — frame position in the entire sequence
    """
    from matplotlib.gridspec import GridSpec

    eid       = result['embryo_id']
    gts       = result['ground_truth']
    preds     = result['predictions']
    preds_p1  = result.get('predictions_p1', preds)
    confs     = result['confidences']
    probs_p2  = result['probs_all']
    probs_p1  = result.get('probs_p1_all')
    files     = result['frame_files']
    gradcam_list = result.get('gradcam', [])
    N         = len(gts)

    # ── Select frames: 1st and last frame for each class ─────────────────
    if sample_frames is None:
        candidates = {}
        class_spans = {}
        for i, g in enumerate(gts):
            if g is not None:
                if g not in class_spans:
                    class_spans[g] = [i, i]
                else:
                    class_spans[g][1] = i

        for g, (start_idx, end_idx) in class_spans.items():
            class_name = DISPLAY_NAMES[g]
            candidates[f"{class_name}_start"] = start_idx
            if end_idx != start_idx:
                candidates[f"{class_name}_end"] = end_idx

        # Sort by chronological order
        sorted_indices = sorted(candidates.values())
        reverse_cand = {v: k for k, v in candidates.items()}
        
        sample_frames = []
        sample_labels = {}
        for idx in sorted_indices:
            if idx not in sample_frames:
                sample_frames.append(idx)
                sample_labels[idx] = reverse_cand[idx]
    else:
        sample_labels = {f: f'frame_{f}' for f in sample_frames}

    # Only keep frames with GT
    sample_frames = [f for f in sample_frames if f < N and gts[f] is not None]
    if len(sample_frames) < 2:
        return

    n_cols = len(sample_frames)
    fig = plt.figure(figsize=(4.5 * n_cols, 12))
    gs = GridSpec(4, n_cols, figure=fig, height_ratios=[1.6, 1.6, 2.0, 0.8],
                  hspace=0.6, wspace=0.3)

    fig.suptitle(f'Phase 2 Correction Explainer — {eid}',
                 fontsize=14, fontweight='bold', y=0.98)

    for col, fi in enumerate(sample_frames):
        gt   = gts[fi]
        p1   = preds_p1[fi]
        p2   = preds[fi]
        conf = confs[fi]
        label = sample_labels.get(fi, '')

        gt_name = DISPLAY_NAMES[gt] if gt is not None else 'N/A'
        p1_name = DISPLAY_NAMES[p1]
        p2_name = DISPLAY_NAMES[p2]

        # ── Row 0: Embryo image ───────────────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        img_rgb = None
        try:
            img_rgb = np.array(Image.open(files[fi]).convert('RGB'))
            ax.imshow(img_rgb)
        except Exception:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

        border_color = '#2ecc71' if p2 == gt else '#e74c3c'
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)

        title_parts = [f'Frame {fi}']
        if label:
            title_parts.append(f'({label})')
        ax.set_title(' '.join(title_parts), fontsize=9, fontweight='bold')

        txt = f'GT: {gt_name}\nP1: {p1_name}  P2: {p2_name}\nConf: {conf:.0%}'
        ax.text(0.5, -0.02, txt, transform=ax.transAxes, fontsize=8,
                ha='center', va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # ── Row 1: Grad-CAM overlay ───────────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        if img_rgb is not None:
            gc = gradcam_list[fi] if fi < len(gradcam_list) else None
            if gc is not None:
                try:
                    from pytorch_grad_cam.utils.image import show_cam_on_image
                    img_f = np.float32(img_rgb) / 255.0
                    gc_r = cv2.resize(gc, (img_rgb.shape[1], img_rgb.shape[0]))
                    overlay = show_cam_on_image(img_f, gc_r, use_rgb=True)
                    ax.imshow(overlay)
                except Exception:
                    ax.imshow(img_rgb)
            else:
                ax.imshow(img_rgb)
        else:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        ax.set_title('Phase 1 Grad-CAM', fontsize=8)

        # ── Row 2: P1 vs P2 probability comparison ───────────────────────
        ax = fig.add_subplot(gs[2, col])
        x = np.arange(NUM_CLASSES)
        w_bar = 0.35
        p2_probs_f = probs_p2[fi]
        p1_probs_f = probs_p1[fi] if probs_p1 is not None else p2_probs_f

        ax.bar(x - w_bar / 2, p1_probs_f, w_bar, color='#e67e22',
               alpha=0.6, label='P1 (EmbryoMambaNet)')
        ax.bar(x + w_bar / 2, p2_probs_f, w_bar, color='#3498db',
               alpha=0.8, label='P2 (EmbryoTemporalNet)')

        if gt is not None:
            ax.axvspan(gt - 0.5, gt + 0.5, alpha=0.12, color='#2ecc71')

        ax.set_xticks(x)
        ax.set_xticklabels(DISPLAY_NAMES, fontsize=7, rotation=45)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Probability', fontsize=7)
        ax.set_title('P1 vs P2 Probabilities\n(green = GT class)', fontsize=8)
        if col == 0:
            ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=6)

        # ── Row 3: Timeline context ───────────────────────────────────────
        ax = fig.add_subplot(gs[3, col])
        for i in range(N):
            g = gts[i]
            color_t = STAGE_COLORS_MPL[g] if (g is not None and 0 <= g < NUM_CLASSES) else '#555'
            ax.axvspan(i, i + 1, color=color_t, alpha=0.6)
        ax.axvline(fi, color='red', lw=2, zorder=5)
        ax.set_xlim(0, N)
        ax.set_yticks([])
        ax.set_xlabel(f'Frame (▼={fi})', fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Phase 2 Correction Explainer → {output_dir}")


# ---------------------------------------------------------------------------
# Metrics Summary Table — summary table for paper
# ---------------------------------------------------------------------------

def create_metrics_summary(all_results: list, output_dir: Path, split: str):
    """
    Create metrics summary table for Phase 1 vs Phase 2 as:
      1. PNG figure (table) — used directly in paper
      2. CSV file — used for LaTeX

    Metrics per class: Precision, Recall, F1
    Aggregate: Macro F1, Weighted F1, Top-1 Accuracy, Top-3 Accuracy
    """
    try:
        from sklearn.metrics import (
            precision_recall_fscore_support, accuracy_score, top_k_accuracy_score
        )
    except ImportError:
        print("  ⚠️ sklearn required for metrics summary")
        return

    all_gts, all_p1, all_p2 = [], [], []
    all_probs_p1, all_probs_p2 = [], []

    for r in all_results:
        probs_p1_r = r.get('probs_p1_all')
        preds_p1_r = r.get('predictions_p1', r['predictions'])
        for i, gt in enumerate(r['ground_truth']):
            if gt is not None:
                all_gts.append(gt)
                all_p1.append(preds_p1_r[i])
                all_p2.append(r['predictions'][i])
                all_probs_p2.append(r['probs_all'][i])
                if probs_p1_r is not None:
                    all_probs_p1.append(probs_p1_r[i])

    if not all_gts:
        return

    all_gts = np.array(all_gts)
    all_p1  = np.array(all_p1)
    all_p2  = np.array(all_p2)
    all_probs_p2 = np.array(all_probs_p2)
    all_probs_p1 = np.array(all_probs_p1) if all_probs_p1 else all_probs_p2

    labels_range = list(range(NUM_CLASSES))

    # Per-class metrics
    p1_prec, p1_rec, p1_f1, _ = precision_recall_fscore_support(
        all_gts, all_p1, labels=labels_range, zero_division=0)
    p2_prec, p2_rec, p2_f1, _ = precision_recall_fscore_support(
        all_gts, all_p2, labels=labels_range, zero_division=0)

    # Aggregate
    p1_acc = accuracy_score(all_gts, all_p1) * 100
    p2_acc = accuracy_score(all_gts, all_p2) * 100

    p1_macro_f1 = np.mean(p1_f1)
    p2_macro_f1 = np.mean(p2_f1)

    _, _, p1_wf1, _ = precision_recall_fscore_support(
        all_gts, all_p1, average='weighted', zero_division=0)
    _, _, p2_wf1, _ = precision_recall_fscore_support(
        all_gts, all_p2, average='weighted', zero_division=0)

    # Top-3
    p1_top3 = top_k_accuracy_score(all_gts, all_probs_p1, k=3,
                                    labels=labels_range) * 100
    p2_top3 = top_k_accuracy_score(all_gts, all_probs_p2, k=3,
                                    labels=labels_range) * 100

    # ── Build table data ──────────────────────────────────────────────────
    rows = []
    for i, name in enumerate(DISPLAY_NAMES):
        rows.append([
            name,
            f'{p1_prec[i]:.3f}', f'{p1_rec[i]:.3f}', f'{p1_f1[i]:.3f}',
            f'{p2_prec[i]:.3f}', f'{p2_rec[i]:.3f}', f'{p2_f1[i]:.3f}',
        ])
    rows.append([''] * 7)  # separator
    rows.append(['Macro F1',    '', '', f'{p1_macro_f1:.4f}',
                                '', '', f'{p2_macro_f1:.4f}'])
    rows.append(['Weighted F1', '', '', f'{p1_wf1:.4f}',
                                '', '', f'{p2_wf1:.4f}'])
    rows.append(['Top-1 Acc',   '', '', f'{p1_acc:.2f}%',
                                '', '', f'{p2_acc:.2f}%'])
    rows.append(['Top-3 Acc',   '', '', f'{p1_top3:.2f}%',
                                '', '', f'{p2_top3:.2f}%'])

    col_labels = ['Stage',
                  'P1 Prec', 'P1 Rec', 'P1 F1',
                  'P2 Prec', 'P2 Rec', 'P2 F1']

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = output_dir / f'metrics_summary_{split}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(col_labels)
        for row in rows:
            writer.writerow(row)
    print(f"📝 Metrics CSV → {csv_path}")

    # ── Save PNG table ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#34495e')
        cell.set_text_props(color='white', fontweight='bold')

    # Color P2 F1 cells: green if better than P1
    for i in range(NUM_CLASSES):
        p1_val = p1_f1[i]
        p2_val = p2_f1[i]
        cell = table[i + 1, 6]  # P2 F1 column
        if p2_val > p1_val + 0.01:
            cell.set_facecolor('#d5f5e3')
        elif p2_val < p1_val - 0.01:
            cell.set_facecolor('#fadbd8')

    fig.suptitle(
        f'EmbryoStageNet — {split} set  |  '
        f'EmbryoMambaNet (Phase 1) + EmbryoTemporalNet Mamba2+Attn (Phase 2)\n'
        f'{len(all_results)} embryos, {len(all_gts)} frames',
        fontsize=12, fontweight='bold', y=0.95)

    plt.tight_layout()
    out = output_dir / f'metrics_summary_{split}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Metrics Summary → {out}")


# ---------------------------------------------------------------------------
# Chart B: Error Correction Flow — How Phase 2 corrects Phase 1 errors
# ---------------------------------------------------------------------------

def create_error_correction_chart(all_results: list, output_dir: Path, split: str):
    """
    Sankey-style flow diagram + confusion delta:

    Panel left: Error Flow Heatmap
      3D Matrix: when Phase 1 predicts X (wrong), GT is Y,
      Phase 2 predicts Z. Displayed as heatmap:
      - Rows = (P1_pred → GT) pair (only when P1 is wrong)
      - Columns = Phase 2 output
      - Color = frame count
      Shows whether Phase 2 corrects to the right GT or to another class.

    Panel right: Confusion Matrix Delta (P2 - P1)
      Confusion matrix difference: positive cells = P2 predicts more than P1,
      negative cells = P2 predicts less than P1. Shows in which direction
      P2 "moves" predictions — expected: diagonal increases (correct),
      off-diagonal decreases.
    """
    try:
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("  ⚠️ Requires sklearn + seaborn — skipping error correction chart")
        return

    all_gts, all_p1, all_p2 = [], [], []
    for r in all_results:
        for p1, p2, gt in zip(
            r.get('predictions_p1', r['predictions']),
            r['predictions'], r['ground_truth']
        ):
            if gt is not None:
                all_gts.append(gt)
                all_p1.append(p1)
                all_p2.append(p2)

    if not all_gts:
        return

    labels_range = list(range(NUM_CLASSES))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f'EmbryoStageNet — Confusion Matrix Comparison — {split}  '
                 f'({len(all_gts)} frames)',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: Confusion Matrix Phase 1 (normalized) ──────────────────
    ax = axes[0]
    cm_p1 = confusion_matrix(all_gts, all_p1, labels=labels_range)
    cm_p1_norm = cm_p1.astype(float)
    for i in range(NUM_CLASSES):
        s = cm_p1_norm[i].sum()
        if s > 0:
            cm_p1_norm[i] /= s

    sns.heatmap(cm_p1_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                ax=ax, linewidths=0.4, vmin=0, vmax=1,
                cbar_kws={'shrink': 0.7})
    p1_acc = np.trace(cm_p1) / cm_p1.sum() * 100
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title(f'EmbryoMambaNet (Phase 1)\nAcc = {p1_acc:.2f}%')

    # ── Panel 2: Confusion Matrix Phase 2 (normalized) ──────────────────
    ax = axes[1]
    cm_p2 = confusion_matrix(all_gts, all_p2, labels=labels_range)
    cm_p2_norm = cm_p2.astype(float)
    for i in range(NUM_CLASSES):
        s = cm_p2_norm[i].sum()
        if s > 0:
            cm_p2_norm[i] /= s

    sns.heatmap(cm_p2_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                ax=ax, linewidths=0.4, vmin=0, vmax=1,
                cbar_kws={'shrink': 0.7})
    p2_acc = np.trace(cm_p2) / cm_p2.sum() * 100
    ax.set_xlabel('Predicted')
    ax.set_ylabel('')
    ax.set_title(f'EmbryoTemporalNet (Phase 2)\nAcc = {p2_acc:.2f}%')

    # ── Panel 3: Improvement (P2 − P1) ───────────────────────────────────
    ax = axes[2]
    delta = (cm_p2_norm - cm_p1_norm) * 100  # percentage points

    # Annotation: use original values, but use improvement logic for colormap
    # Diagonal positive = P2 correct more = good = green
    # Off-diagonal positive = P2 confused more = bad = red
    # Flip off-diagonal sign for consistent colormap: green = good
    improvement = delta.copy()
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j:
                improvement[i, j] = -delta[i, j]

    vmax = max(abs(improvement).max(), 3)

    # Plot heatmap with improvement (green = good)
    sns.heatmap(improvement, annot=False, cmap='RdYlGn', center=0,
                xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                ax=ax, linewidths=0.4, vmin=-vmax, vmax=vmax,
                cbar_kws={'label': 'Improvement (pp)', 'shrink': 0.7})
    # Annotate with original delta values (not flipped) for readability
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = delta[i, j]
            ax.text(j + 0.5, i + 0.5, f'{val:+.1f}',
                    ha='center', va='center', fontsize=8,
                    color='black' if abs(improvement[i, j]) < vmax * 0.6 else 'white')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('')
    ax.set_title(f'Δ (P2 − P1) pp\nGreen = improved')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = output_dir / f'error_correction_{split}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Error Correction → {out}")


# ---------------------------------------------------------------------------
# Chart C: Transition Landscape — accuracy based on distance to transition
# ---------------------------------------------------------------------------

def create_transition_landscape(all_results: list, output_dir: Path, split: str):
    """
    Line chart: accuracy P1 vs P2 based on distance to transition point.

    X-axis = offset from transition (−30 → +30 frames)
      - Negative = before transition (still in old stage)
      - 0 = Stage transition frame
      - Positive = after transition (already in new stage)

    Y-axis = accuracy at that offset (aggregated across all embryos)

    Meaning:
      - Phase 1 (frame-level) usually fails more around offset 0
      - Phase 2 (temporal) should recover faster after transition
      - If Phase 2 is better: P2 line is above P1 especially around offset 0
      - Line shape shows the "reaction time" of the model

    Additional: shaded area = confidence interval (std across embryos)
    """
    max_offset = 30
    offsets = list(range(-max_offset, max_offset + 1))

    # Collect per-offset accuracy across all transitions
    p1_correct_at = defaultdict(list)  # offset → [0/1, 0/1, ...]
    p2_correct_at = defaultdict(list)

    for r in all_results:
        gts   = r['ground_truth']
        preds = r['predictions']
        p1s   = r.get('predictions_p1', preds)
        n     = len(gts)

        # Find transition points
        transitions = []
        for i in range(1, n):
            if gts[i] is not None and gts[i - 1] is not None and gts[i] != gts[i - 1]:
                transitions.append(i)

        for tp in transitions:
            for offset in offsets:
                idx = tp + offset
                if 0 <= idx < n and gts[idx] is not None:
                    p1_correct_at[offset].append(int(p1s[idx] == gts[idx]))
                    p2_correct_at[offset].append(int(preds[idx] == gts[idx]))

    if not p1_correct_at:
        print("  ⚠️ No transitions found — skipping transition landscape")
        return

    # Compute mean ± std
    p1_means = [np.mean(p1_correct_at[o]) * 100 if p1_correct_at[o] else np.nan
                for o in offsets]
    p2_means = [np.mean(p2_correct_at[o]) * 100 if p2_correct_at[o] else np.nan
                for o in offsets]
    p1_stds = [np.std(p1_correct_at[o]) * 100 / max(np.sqrt(len(p1_correct_at[o])), 1)
               if p1_correct_at[o] else 0 for o in offsets]
    p2_stds = [np.std(p2_correct_at[o]) * 100 / max(np.sqrt(len(p2_correct_at[o])), 1)
               if p2_correct_at[o] else 0 for o in offsets]
    counts = [len(p1_correct_at[o]) for o in offsets]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Transition Landscape — {split}\n'
                 f'Accuracy vs Distance to Stage Transition Point',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: Accuracy curves ──────────────────────────────────────────
    ax = axes[0]
    p1_m = np.array(p1_means)
    p2_m = np.array(p2_means)
    p1_s = np.array(p1_stds)
    p2_s = np.array(p2_stds)
    ox = np.array(offsets)

    ax.fill_between(ox, p1_m - p1_s, p1_m + p1_s, alpha=0.15, color='#e67e22')
    ax.fill_between(ox, p2_m - p2_s, p2_m + p2_s, alpha=0.15, color='#3498db')
    ax.plot(ox, p1_m, color='#e67e22', lw=2, label='Phase 1 (EmbryoMambaNet)', marker='.',
            markersize=3)
    ax.plot(ox, p2_m, color='#3498db', lw=2, label='Phase 2 (Mamba2+Attn)', marker='.',
            markersize=3)

    # Highlight transition zone
    ax.axvspan(-5, 5, alpha=0.08, color='red', label='Transition zone (±5)')
    ax.axvline(0, color='red', lw=1.5, ls='--', alpha=0.6)
    ax.text(0.5, 0.02, 'transition', ha='center', fontsize=8, color='red',
            transform=ax.get_xaxis_transform())

    ax.set_xlim(-max_offset, max_offset)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Offset from Transition Point (frames)')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.2)

    # ── Panel 2: Sample count per offset ──────────────────────────────────
    ax = axes[1]
    ax.bar(offsets, counts, color='#95a5a6', alpha=0.6, width=1.0)
    ax.set_xlim(-max_offset, max_offset)
    ax.set_ylabel('# Samples')
    ax.set_xlabel('Offset from Transition Point')
    ax.axvline(0, color='red', lw=1, ls='--', alpha=0.4)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    out = output_dir / f'transition_landscape_{split}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Transition Landscape → {out}")


# ---------------------------------------------------------------------------
# Chart D: Confidence Dynamics — P2 confidence over time around transition
# (Replacing old Attention Dynamics — Phase 2 Mamba2 has no attention weights)
# ---------------------------------------------------------------------------

def create_attention_dynamics_chart(all_results: list, output_dir: Path, split: str,
                                    margin: int = 8):
    """
    2-panel aggregate figure:

    Panel left: Confidence vs Correct/Wrong scatter
      - X-axis = P2 confidence
      - Y-axis = density
      - Color = correct (green) / incorrect (red)
      - Shows: when is the model confident? Is calibration good?

    Panel right: P2 confidence time-series mean around transition
      - X-axis = offset from transition (−25 → +25 frames)
      - Y-axis = mean P2 confidence
      - Shows whether the model is "uncertain" (lower confidence) near transition
    """
    confs_correct = []
    confs_wrong   = []

    max_offset = 25
    offsets = list(range(-max_offset, max_offset + 1))
    conf_at_offset = defaultdict(list)
    acc_at_offset  = defaultdict(list)

    for r in all_results:
        gts   = r['ground_truth']
        preds = r['predictions']
        confs = r['confidences']
        n     = len(gts)

        transitions = []
        for i in range(1, n):
            if gts[i] is not None and gts[i - 1] is not None and gts[i] != gts[i - 1]:
                transitions.append(i)

        for i, (pred, gt, conf) in enumerate(zip(preds, gts, confs)):
            if gt is None:
                continue
            if pred == gt:
                confs_correct.append(conf)
            else:
                confs_wrong.append(conf)

            for tp in transitions:
                offset = i - tp
                if -max_offset <= offset <= max_offset:
                    conf_at_offset[offset].append(conf)
                    acc_at_offset[offset].append(int(pred == gt))

    if not confs_correct and not confs_wrong:
        print("  ⚠️ No data — skipping confidence dynamics chart")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Phase 2 Confidence Dynamics — {split}',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: Confidence distribution ─────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 30)
    ax.hist(confs_correct, bins=bins, alpha=0.65, color='#2ecc71', density=True,
            label=f'Correct (n={len(confs_correct)})')
    ax.hist(confs_wrong,   bins=bins, alpha=0.65, color='#e74c3c', density=True,
            label=f'Incorrect (n={len(confs_wrong)})')
    ax.set_xlabel('P2 Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution\n(EmbryoTemporalNet Phase 2)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # ── Panel 2: Confidence around transitions ────────────────────────────
    ax = axes[1]
    means = [np.mean(conf_at_offset[o]) if conf_at_offset[o] else np.nan
             for o in offsets]
    stds  = [np.std(conf_at_offset[o]) / max(np.sqrt(len(conf_at_offset[o])), 1)
             if conf_at_offset[o] else 0 for o in offsets]
    acc_means = [np.mean(acc_at_offset[o]) * 100 if acc_at_offset[o] else np.nan
                 for o in offsets]

    means = np.array(means)
    stds  = np.array(stds)
    ox    = np.array(offsets)

    ax.fill_between(ox, means - stds, means + stds, alpha=0.2, color='#9b59b6')
    ax.plot(ox, means, color='#9b59b6', lw=2.5, marker='.', markersize=4,
            label='Mean Confidence')

    ax2 = ax.twinx()
    ax2.plot(ox, acc_means, color='#3498db', lw=1.5, ls='--', alpha=0.7,
             label='Accuracy (%)')
    ax2.set_ylabel('Accuracy (%)', color='#3498db')
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax2.set_ylim(0, 105)

    ax.axvline(0, color='red', lw=1.5, ls='--', alpha=0.6)
    ax.axvspan(-margin, margin, alpha=0.06, color='red')
    ax.text(0.5, 0.02, 'transition', ha='center', fontsize=8, color='red',
            transform=ax.get_xaxis_transform())

    ax.set_xlim(-max_offset, max_offset)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Offset from Transition Point (frames)')
    ax.set_ylabel('P2 Confidence')
    ax.set_title('Does Confidence Drop Near Stage Transitions?\n'
                 '(Mamba2 SSM — temporal memory effect)')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(alpha=0.15)

    plt.tight_layout()
    out = output_dir / f'attention_dynamics_{split}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confidence Dynamics → {out}")


# ---------------------------------------------------------------------------
# Chart E: Probability Smoothing — Phase 2 reduces oscillation of Phase 1
# ---------------------------------------------------------------------------

def create_probability_smoothing_chart(all_results: list, output_dir: Path, split: str):
    """
      Vẽ dưới dạng rate per stage transition.
    """
    # ── Collect flip counts and backward rates ─────────────────────────────
    all_p1_flips = []
    all_p2_flips = []
    all_p1_backward = []
    all_p2_backward = []

    for r in all_results:
        gts   = r['ground_truth']
        preds = r['predictions']
        p1s   = r.get('predictions_p1', preds)
        n     = len(gts)

        p1_flips, p2_flips = 0, 0
        p1_backward, p2_backward = 0, 0
        valid_pairs = 0

        for i in range(1, n):
            if gts[i] is None or gts[i - 1] is None:
                continue
            valid_pairs += 1
            if p1s[i] != p1s[i - 1]:
                p1_flips += 1
            if preds[i] != preds[i - 1]:
                p2_flips += 1
            if p1s[i] < p1s[i - 1]:
                p1_backward += 1
            if preds[i] < preds[i - 1]:
                p2_backward += 1

        if valid_pairs > 0:
            all_p1_flips.append(p1_flips / valid_pairs * 100)
            all_p2_flips.append(p2_flips / valid_pairs * 100)
            all_p1_backward.append(p1_backward / valid_pairs * 100)
            all_p2_backward.append(p2_backward / valid_pairs * 100)

    if not all_p1_flips:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Temporal Smoothing Effect — {split}  '
                 f'({len(all_p1_flips)} embryos)',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: Flip rate distribution ───────────────────────────────────
    ax = axes[0]
    # Paired scatter: each embryo is 1 point, P1 vs P2
    ax.scatter(all_p1_flips, all_p2_flips, s=30, alpha=0.5,
               color='#3498db', edgecolors='white', lw=0.3, zorder=3)
    # Diagonal line (P1 == P2)
    lim = max(max(all_p1_flips), max(all_p2_flips)) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.3, label='P1 = P2')
    # Points below diagonal = P2 more stable
    below = sum(1 for p1, p2 in zip(all_p1_flips, all_p2_flips) if p2 < p1)
    pct_below = below / len(all_p1_flips) * 100

    ax.set_xlabel('Phase 1 Flip Rate (%)')
    ax.set_ylabel('Phase 2 Flip Rate (%)')
    ax.set_title(f'Prediction Stability (each dot = 1 embryo)\n'
                 f'{pct_below:.0f}% embryos: P2 more stable than P1')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)

    # Fill "P2 better" region
    ax.fill_between([0, lim], [0, lim], [0, 0], alpha=0.05, color='#2ecc71')
    ax.text(lim * 0.7, lim * 0.15, 'P2 more\nstable', fontsize=10,
            color='#27ae60', alpha=0.6, ha='center')

    # ── Panel 2: Backward regression rate ─────────────────────────────────
    ax = axes[1]
    ax.scatter(all_p1_backward, all_p2_backward, s=30, alpha=0.5,
               color='#e74c3c', edgecolors='white', lw=0.3, zorder=3)
    lim2 = max(max(all_p1_backward + [1]), max(all_p2_backward + [1])) * 1.1
    ax.plot([0, lim2], [0, lim2], 'k--', lw=1, alpha=0.3, label='P1 = P2')
    below2 = sum(1 for p1, p2 in zip(all_p1_backward, all_p2_backward) if p2 < p1)
    pct_below2 = below2 / len(all_p1_backward) * 100

    ax.set_xlabel('Phase 1 Backward Rate (%)')
    ax.set_ylabel('Phase 2 Backward Rate (%)')
    ax.set_title(f'Backward Regression (pred goes to earlier stage)\n'
                 f'{pct_below2:.0f}% embryos: P2 less backward regression')
    ax.set_xlim(0, lim2)
    ax.set_ylim(0, lim2)
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    ax.fill_between([0, lim2], [0, lim2], [0, 0], alpha=0.05, color='#2ecc71')
    ax.text(lim2 * 0.7, lim2 * 0.15, 'P2 less\nregression', fontsize=10,
            color='#27ae60', alpha=0.6, ha='center')

    plt.tight_layout()
    out = output_dir / f'temporal_smoothing_{split}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Temporal Smoothing → {out}")


# ---------------------------------------------------------------------------
# CSV log
# ---------------------------------------------------------------------------

def save_csv_log(all_results: list, output_path: Path, split: str):
    """Detailed per-frame log: GT, pred_p1, pred_p2, correct, top3 flags, probs P1 & P2."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['split', 'embryo_id', 'frame_idx', 'frame_file',
                  'ground_truth',
                  'prediction_p1', 'correct_p1', 'top3_p1',
                  'prediction_p2', 'correct_p2', 'top3_p2', 'confidence_p2']
        header += [f'prob_p1_{s}' for s in STAGE_NAMES]
        header += [f'prob_p2_{s}' for s in STAGE_NAMES]
        writer.writerow(header)

        for r in all_results:
            eid = r['embryo_id']
            preds_p1  = r.get('predictions_p1', r['predictions'])
            probs_p1s = r.get('probs_p1_all')
            for i, (fp, pred_p1, pred_p2, gt, conf, probs_p2) in enumerate(zip(
                r['frame_files'], preds_p1, r['predictions'],
                r['ground_truth'], r['confidences'], r['probs_all']
            )):
                gt_name = STAGE_NAMES[gt] if gt is not None else 'N/A'
                p1_name = STAGE_NAMES[pred_p1]
                p2_name = STAGE_NAMES[pred_p2]

                # Top-3
                probs_p1_np = probs_p1s[i] if probs_p1s is not None else probs_p2
                top3_p1 = int(gt in np.argsort(probs_p1_np)[::-1][:3]) if gt is not None else ''
                top3_p2 = int(gt in np.argsort(probs_p2)[::-1][:3])    if gt is not None else ''

                correct_p1 = int(pred_p1 == gt) if gt is not None else ''
                correct_p2 = int(pred_p2 == gt) if gt is not None else ''

                row = [split, eid, i + 1, fp.name,
                       gt_name,
                       p1_name, correct_p1, top3_p1,
                       p2_name, correct_p2, top3_p2, f'{conf:.4f}']
                row += [f'{p:.4f}' for p in probs_p1_np]
                row += [f'{p:.4f}' for p in probs_p2]
                writer.writerow(row)

    print(f"📝 CSV log → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='IVF 2D EmbryoMambaNet Visualization')
    p.add_argument('--model_name', default='embryoswin_plusplus',
                   help='Model name to load from registry (e.g. embryoswin_plusplus)')
    p.add_argument('--model',       required=True,
                   help='Path to Phase 1 checkpoint (pth.tar)')
    p.add_argument('--data_root',   required=True,
                   help='Root folder containing embryo directories')
    p.add_argument('--ann_root',    required=True,
                   help='Folder containing *_phases.csv')
    p.add_argument('--splits_json', default=None,
                   help='processdata/splits.json')
    p.add_argument('--split',       default='test',
                   choices=['train', 'val', 'test'],
                   help='Split to run (default: test)')
    p.add_argument('--embryo_id',   default=None,
                   help='Run specific embryo ID (overrides split)')
    p.add_argument('--output_dir',  default='val/results_3d',
                   help='Directory to save results')
    p.add_argument('--num_frames',  type=int, default=1,
                   help='Not used (legacy from 3D). Phase 1 processes single frames.')
    p.add_argument('--phase2_ckpt', default=None, help='Path to Phase 2 best.pth')
    p.add_argument('--img_size',    type=int, default=224)
    p.add_argument('--fps',         type=int, default=10)
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no_video',    action='store_true',
                   help='Skip video generation (charts + CSV only)')
    p.add_argument('--max_embryos', type=int, default=None,
                   help='Limit number of embryos (for quick testing)')
    p.add_argument('--batch_size',  type=int, default=128,
                   help='Batch size for Phase 1 Inference')
    return p.parse_args()


def main():
    args = parse_args()

    data_root  = Path(args.data_root)
    ann_root   = Path(args.ann_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve embryo list
    if args.embryo_id:
        embryo_ids = [args.embryo_id]
        split_tag  = 'single'
    else:
        if not args.splits_json:
            print("❌ Need --splits_json or --embryo_id")
            sys.exit(1)
        with open(args.splits_json) as f:
            splits = json.load(f)
        embryo_ids = splits.get(args.split, [])
        split_tag  = args.split
        print(f"📋 Split '{split_tag}': {len(embryo_ids)} embryos")

    if args.max_embryos:
        embryo_ids = embryo_ids[:args.max_embryos]

    # Load models
    model = load_phase1_model(args.model_name, args.model, args.device)
    phase2_model = None
    if args.phase2_ckpt:
        phase2_model = load_phase2_model(args.phase2_ckpt, args.device)

    all_results = []

    for eid in tqdm(embryo_ids, desc='Embryos'):
        print(f"\n── {eid} ──")
        result = infer_embryo(
            model, phase2_model, data_root, ann_root, eid,
            args.num_frames, args.img_size, args.device, args.batch_size
        )
        if result is None:
            continue

        all_results.append(result)

        emb_dir = output_dir / eid
        emb_dir.mkdir(exist_ok=True)

        create_embryo_chart(result, str(emb_dir / 'chart.png'))

        # Attention heatmap per-embryo (only if Phase 2 is present)
        if phase2_model is not None:
            create_attention_heatmap(result, str(emb_dir / 'attention_heatmap.png'))
            create_phase2_attention_explainer(
                result, str(emb_dir / 'phase2_explainer.png'))

        if not args.no_video:
            create_video(result, str(emb_dir / 'video.mp4'), fps=args.fps)

    if not all_results:
        print("❌ No results found.")
        return

    create_aggregate_charts(all_results, output_dir, split_tag)

    # Aggregate metrics table (always created)
    create_metrics_summary(all_results, output_dir, split_tag)

    # Additional charts for paper
    if phase2_model is not None:
        create_error_correction_chart(all_results, output_dir, split_tag)
        create_transition_landscape(all_results, output_dir, split_tag)
        create_attention_dynamics_chart(all_results, output_dir, split_tag)
        create_probability_smoothing_chart(all_results, output_dir, split_tag)

    save_csv_log(all_results, output_dir / f'predictions_{split_tag}.csv', split_tag)

    print(f"\n✅ Done. Results at: {output_dir}")


if __name__ == '__main__':
    main()
