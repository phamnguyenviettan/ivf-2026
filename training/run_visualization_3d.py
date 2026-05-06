"""
run_visualization_3d.py — IVF 3D MambaVision Visualization Script
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

# ---------------------------------------------------------------------------
# Constants — must match datasets_raw_ivf.py
# ---------------------------------------------------------------------------

STAGE_NAMES = ['1-tpnf', '2-t2', '3-t3+', '5-t5+', '7-t7+', '9-t9+', '10-tm+']
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
    """CLAHE + Sobel — matches datasets_raw_ivf.py."""
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
    Parse *_phases.csv → list[int|None] per frame (0-based index).
    Returns (frame_labels, last_valid_frame) — last_valid_frame is 1-indexed.

    Supports 2 annotation formats:
      - Raw format:    tpnf, t2, t3, t4, t5, t6, t7, t8, t9+, tm, tsb
                       (original embryo_dataset_annotations)
      - Merged format: 1-tPNf, 2-t2, 3-t3+, 5-t5+, 7-t7+, 9-t9+, 10-tM+
                       (dataset_timelapse/embryo_dataset_annotations)

    IMPORTANT: start/end in CSV are actual RUN numbers (e.g. RUN74..RUN83),
    not the position in the frame list. Need to build RUN→index map from frame_files.
    """
    # Mapping cho merged format (dataset_timelapse)
    MERGED_TO_IDX = {
        '1-tpnf': 0,
        '2-t2':   1,
        '3-t3+':  2,
        '5-t5+':  3,
        '7-t7+':  4,
        '9-t9+':  5,
        '10-tm+': 6,
    }

    frame_labels = [None] * total_frames
    last_valid_frame = 0

    if not csv_path.exists():
        return frame_labels, last_valid_frame

    # Build RUN number → 0-based list index map (if frame_files provided)
    # E.g.: RUN74.jpeg → run_to_idx[74] = 0 (first position after sorting)
    run_to_idx = {}
    if frame_files:
        for list_idx, fp in enumerate(frame_files):
            run_num = _run_sort_key(fp)
            run_to_idx[run_num] = list_idx

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                stage = row[0].strip().lower()
                try:
                    start = int(row[1].strip())
                    end   = int(row[2].strip())
                except ValueError:
                    continue
                if stage in EXCLUDED_STAGES:
                    continue
                # Thử raw format trước, sau đó merged format
                idx = RAW_TO_IDX.get(stage)
                if idx is None:
                    idx = MERGED_TO_IDX.get(stage)
                if idx is None:
                    continue

                if run_to_idx:
                    # Use RUN number map: assign labels to all frames with RUN in [start, end]
                    for run_num, list_idx in run_to_idx.items():
                        if start <= run_num <= end:
                            frame_labels[list_idx] = idx
                            last_valid_frame = max(last_valid_frame, list_idx + 1)
                else:
                    # Fallback: use start/end as 1-indexed frame index
                    for fn in range(start, end + 1):
                        if 1 <= fn <= total_frames:
                            frame_labels[fn - 1] = idx
                            last_valid_frame = max(last_valid_frame, fn)
    except Exception as e:
        print(f"  ⚠️ CSV error {csv_path}: {e}")

    return frame_labels, last_valid_frame


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_3d(ckpt_path: str, num_frames: int, device: str):
    """
    Load Mamba2VisionMorph_Model from checkpoint.
    """
    import importlib.util

    model_file = Path(__file__).parent.parent / 'training' / 'models' / 'ivf_3d_mamba2.py'
    spec = importlib.util.spec_from_file_location('ivf_3d_mamba2', model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules['ivf_3d_mamba2'] = module
    spec.loader.exec_module(module)
    Mamba2VisionMorph_Model = module.Mamba2VisionMorph_Model

    model = Mamba2VisionMorph_Model(
        num_classes=NUM_CLASSES,
        num_frames=num_frames,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt
    for key in ('state_dict', 'model', 'model_state_dict'):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠️  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model.eval()
    print(f"✅ Model loaded: {ckpt_path}  (params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M)")
    return model


def load_phase2_model(ckpt_path: str, device: str):
    """
    Load Phase 2 EmbryoTemporalNet (MambaVision-style: Mamba2→Attention on
    sliding window buffer) from checkpoint.
    """
    import importlib.util as _ilu
    root = Path(__file__).parent.parent
    p2_script = root / 'training' / 'models' / 'ivf_phase2_mamba2MIXatteniton.py'

    spec = _ilu.spec_from_file_location('ivf_phase2_mamba2MIXatteniton', p2_script)
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = module.EmbryoTemporalNet()

    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠️  Phase2 missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  ⚠️  Phase2 unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    model.eval()
    model.to(device)
    val_acc = ckpt.get('val_acc', 0.0)
    epoch   = ckpt.get('epoch',   '?')
    cls = module.EmbryoTemporalNet
    print(f"✅ Phase 2 EmbryoTemporalNet loaded: {ckpt_path} (Epoch {epoch}, Val Acc: {val_acc:.2f}%)")
    print(f"   d_model={cls.D_MODEL}, {cls.N_MAMBA}×Mamba2(seq)+{cls.N_ATTN}×Attn, "
          f"window={cls.WINDOW_SIZE}f, d_state={cls.D_STATE}")
    return model


# ---------------------------------------------------------------------------
# Grad-CAM setup for 3D model
# ---------------------------------------------------------------------------

def setup_gradcam(model):
    try:
        from pytorch_grad_cam import GradCAM

        cam_stage5 = GradCAM(model=model, target_layers=[model.norm])
        
        cam_stage4 = GradCAM(
            model=model, 
            target_layers=[model.level_3],
            reshape_transform=lambda x: x.transpose(1, 2).reshape(-1, x.size(1), x.size(3), x.size(4)) if x.dim() == 5 else x
        )

        # PyTorch Grad-CAM has issues with cv2.resize when input_tensor is 5D because it returns target_size (D, W, H).
        # We monkey-patch this function to only take (W, H) spatial dimensions, as we project attention onto 2D image.
        cam_stage5.get_target_width_height = lambda x: (x.size(-1), x.size(-2))
        cam_stage4.get_target_width_height = lambda x: (x.size(-1), x.size(-2))

        return {'stage4': cam_stage4, 'stage5': cam_stage5}, [model.level_3, model.norm]
    except Exception as e:
        print(f"  ⚠️ Grad-CAM setup failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_embryo(model, phase2_model, data_root: Path, ann_root: Path,
                 embryo_id: str, num_frames: int, img_size: int, device: str,
                 gradcam_model=None):
    embryo_dir = data_root / embryo_id
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
        frame_files=frame_files,
    )
    if last_valid == 0:
        print(f"  ⚠️ {embryo_id}: no annotations")
        return None

    valid_files = frame_files[:last_valid]
    valid_gt    = gt_labels[:last_valid]

    processed = []
    for fp in tqdm(valid_files, desc=f"  Preprocess {embryo_id}", leave=False):
        img = np.array(Image.open(fp).convert('RGB'))
        img = apply_clahe_sobel(img)
        img = cv2.resize(img, (img_size, img_size))
        t = (img.astype(np.float32) / 255.0 - MEAN) / STD
        processed.append(t)

    N = len(processed)
    predictions, predictions_p1, confidences, probs_all, probs_p1_list = [], [], [], [], []
    attn_weights_list = []
    gradcam_list = []

    use_gradcam = gradcam_model is not None
    if use_gradcam:
        try:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError:
            use_gradcam = False

    if phase2_model is not None:
        kv_cache = phase2_model.make_initial_hidden(1, device)
    else:
        kv_cache = None

    total_valid = max(sum(1 for g in valid_gt if g is not None), 1)
    stage_run_gt = []
    _prev_lbl, _run = None, 0
    for lbl in valid_gt:
        if lbl is None:
            stage_run_gt.append(0)
            continue
        if lbl == _prev_lbl:
            _run += 1
        else:
            _run = 1
            _prev_lbl = lbl
        stage_run_gt.append(_run)

    window_size = getattr(phase2_model, 'window_size', 64) if phase2_model is not None else 64

    with torch.no_grad():
        for end in tqdm(range(N), desc=f"  Infer {embryo_id}", leave=False):
            indices = [max(0, end - num_frames + 1 + k) for k in range(num_frames)]
            clip = np.stack([processed[i] for i in indices], axis=0)
            clip_t = torch.from_numpy(clip).permute(0, 3, 1, 2)
            clip_t = clip_t.unsqueeze(0).to(device)

            feat = model.forward_features(clip_t)
            out = model.head(feat)
            probs_p1 = F.softmax(out, dim=1)
            probs_p1_np = probs_p1[0].cpu().numpy()

            if use_gradcam:
                try:
                    with torch.set_grad_enabled(True):
                        clip_for_cam = clip_t.clone().requires_grad_(True)
                        target_class = int(out.argmax(dim=1))
                        targets = [ClassifierOutputTarget(target_class)]

                        if isinstance(gradcam_model, dict):
                            gc4 = gradcam_model['stage4'](input_tensor=clip_for_cam, targets=targets)
                            gc5 = gradcam_model['stage5'](input_tensor=clip_for_cam, targets=targets)
                            
                            gradcam_list.append({
                                'stage4': gc4.astype(np.float32),
                                'stage5': gc5[0].astype(np.float32)
                            })
                        else:
                            gc = gradcam_model(input_tensor=clip_for_cam, targets=targets)
                            gradcam_list.append({'stage5': gc[0].astype(np.float32)})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    gradcam_list.append(None)
            else:
                gradcam_list.append(None)

            if phase2_model is not None:
                real_frame_idx = _run_sort_key(valid_files[end])
                result = phase2_model(feat, probs_p1, kv_cache, frame_idx=end)
                logits, kv_cache, attn_out = result[0], result[1], result[2]
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                attn_weights_list.append(attn_out[0].cpu().numpy())
            else:
                probs = probs_p1_np
                attn_weights_list.append(None)

            pred    = int(probs.argmax())
            pred_p1 = int(probs_p1_np.argmax())

            predictions.append(pred)
            predictions_p1.append(pred_p1)
            confidences.append(float(probs[pred]))
            probs_all.append(probs)
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
        'gradcam':        gradcam_list,
    }


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def create_video(result: dict, output_path: str, fps: int = 10):
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

        if is_correct is True:
            cv2.rectangle(panel, (8, 230), (panel_w - 8, 262), (40, 140, 40), -1)
            cv2.putText(panel, "CORRECT", (48, 252),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
        elif is_correct is False:
            cv2.rectangle(panel, (8, 230), (panel_w - 8, 262), (40, 40, 140), -1)
            cv2.putText(panel, "INCORRECT", (28, 252),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
        cv2.line(panel, (8, 271), (panel_w - 8, 271), (70, 70, 70), 1)

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

        ov_c = (0, 255, 0) if is_correct else (0, 0, 255) if is_correct is False else (200, 200, 0)
        cv2.rectangle(frame, (4, 4), (orig_w - 4, 46), (0, 0, 0), -1)
        cv2.rectangle(frame, (4, 4), (orig_w - 4, 46), ov_c, 2)
        cv2.putText(frame, f"Pred: {STAGE_NAMES[pred]}  ({conf*100:.0f}%)",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, ov_c, 2)

        writer.write(np.hstack([frame, panel]))

    writer.release()
    print(f"  🎬 Video → {output_path}")


# ---------------------------------------------------------------------------
# Per-embryo chart
# ---------------------------------------------------------------------------

def create_embryo_chart(result: dict, output_path: str):
    preds_all = result['predictions']
    preds_p1_all = result.get('predictions_p1', None)
    gts_all   = result['ground_truth']
    confs_all = result['confidences']
    eid       = result['embryo_id']

    valid_idx = [i for i, g in enumerate(gts_all) if g is not None]
    preds = [preds_all[i] for i in valid_idx]
    gts   = [gts_all[i]   for i in valid_idx]
    confs = [confs_all[i] for i in valid_idx]

    frames = list(range(len(preds)))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'Embryo: {eid}  ({len(preds)} valid frames)',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.step(frames, gts, where='post', color='#2ecc71', lw=2, label='Ground Truth')
    if preds_p1_all is not None:
        preds_p1 = [preds_p1_all[i] for i in valid_idx]
        ax.step(frames, preds_p1, where='post', color='#e67e22', lw=1.0,
                ls=':', alpha=0.7, label='MambaEmbryo (P1)')
    ax.step(frames, preds, where='post', color='#3498db', lw=1.5,
            ls='-', alpha=0.85, label='CausalRefiner (P2)')
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(STAGE_NAMES, fontsize=7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Stage')
    ax.set_title('Prediction Timeline')
    ax.set_ylim(-0.5, NUM_CLASSES - 0.5)
    p2_acc = sum(p == g for p, g in zip(preds, gts)) / len(preds) * 100
    legend_labels = ['Ground Truth']
    legend_handles = [plt.Line2D([0], [0], color='#2ecc71', lw=2)]
    if preds_p1_all is not None:
        p1_acc = sum(p == g for p, g in zip(preds_p1, gts)) / len(preds) * 100
        legend_handles.append(plt.Line2D([0], [0], color='#e67e22', lw=1, ls=':'))
        legend_labels.append(f'MambaEmbryo P1 ({p1_acc:.1f}%)')
    legend_handles.append(plt.Line2D([0], [0], color='#3498db', lw=1.5))
    legend_labels.append(f'CausalRefiner P2 ({p2_acc:.1f}%)')
    ax.legend(legend_handles, legend_labels, fontsize=7, loc='upper left')
    ax.grid(alpha=0.2)

    ax = axes[1]
    colors_pts = ['#2ecc71' if p == g else '#e74c3c' for p, g in zip(preds, gts)]
    ax.scatter(frames, confs, c=colors_pts, s=12, zorder=3, alpha=0.7)
    ax.plot(frames, confs, color='#9b59b6', lw=0.8, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence Over Time')
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.step(frames, gts, where='post', color='gray', lw=1.2, ls='--', alpha=0.3)
    ax2.set_ylim(-0.5, NUM_CLASSES - 0.5)
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_yticklabels(STAGE_NAMES, fontsize=6, color='gray')

    ax = axes[2]
    try:
        from sklearn.metrics import confusion_matrix as _cm
        import seaborn as sns
        cm = _cm(gts, preds, labels=list(range(NUM_CLASSES)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
                    ax=ax, linewidths=0.4, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix')
    except ImportError:
        ax.text(0.5, 0.5, 'Requires scikit-learn & seaborn',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Chart → {output_path}")


# ---------------------------------------------------------------------------
# Aggregate charts
# ---------------------------------------------------------------------------

def create_aggregate_charts(all_results: list, output_dir: Path, split: str):
    try:
        from sklearn.metrics import (
            confusion_matrix, classification_report, f1_score
        )
        import seaborn as sns
    except ImportError:
        print("⚠️  sklearn / seaborn not installed — skipping aggregate charts")
        return

    all_preds, all_gts = [], []
    for r in all_results:
        for p, g in zip(r['predictions'], r['ground_truth']):
            if g is not None:
                all_preds.append(p)
                all_gts.append(g)

    if not all_preds:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f'Aggregate Results — split={split}  (n={len(all_results)} embryos, '
        f'{len(all_preds)} frames)',
        fontsize=13, fontweight='bold'
    )

    ax = axes[0, 0]
    cm = confusion_matrix(all_gts, all_preds,
                          labels=list(range(NUM_CLASSES)), normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
                ax=ax, linewidths=0.4, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Normalised Confusion Matrix')

    ax = axes[0, 1]
    report = classification_report(
        all_gts, all_preds, labels=list(range(NUM_CLASSES)),
        target_names=STAGE_NAMES, output_dict=True, zero_division=0
    )
    f1s  = [report[s]['f1-score']  for s in STAGE_NAMES]
    prec = [report[s]['precision'] for s in STAGE_NAMES]
    rec  = [report[s]['recall']    for s in STAGE_NAMES]
    x = np.arange(NUM_CLASSES); w = 0.26
    ax.bar(x - w, prec, w, label='Precision', color='#3498db', alpha=0.85)
    ax.bar(x,     rec,  w, label='Recall',    color='#2ecc71', alpha=0.85)
    ax.bar(x + w, f1s,  w, label='F1',        color='#e67e22', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(STAGE_NAMES, rotation=30, ha='right')
    ax.set_ylim(0, 1.12); ax.set_ylabel('Score')
    macro_f1 = f1_score(all_gts, all_preds, average='macro', zero_division=0)
    ax.set_title('Per-Class Precision / Recall / F1')
    ax.set_xlabel(f'Macro-F1 = {macro_f1:.3f}')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25)

    ax = axes[1, 0]
    c_ok, c_err = [], []
    for r in all_results:
        for c, p, g in zip(r['confidences'], r['predictions'], r['ground_truth']):
            if g is not None:
                (c_ok if p == g else c_err).append(c)
    ax.hist(c_ok,  bins=25, alpha=0.65, color='#2ecc71',
            label=f'Correct ({len(c_ok)})',   density=True)
    ax.hist(c_err, bins=25, alpha=0.65, color='#e74c3c',
            label=f'Incorrect ({len(c_err)})', density=True)
    ax.set_xlabel('Confidence'); ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution (All Embryos)')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    emb_accs, emb_lens = [], []
    for r in all_results:
        valid = [(p, g) for p, g in zip(r['predictions'], r['ground_truth'])
                 if g is not None]
        if valid:
            acc = sum(p == g for p, g in valid) / len(valid) * 100
            emb_accs.append(acc)
            emb_lens.append(len(valid))
    ax.scatter(emb_lens, emb_accs, alpha=0.6, s=40,
               color='#9b59b6', edgecolors='white', lw=0.4)
    mean_acc = np.mean(emb_accs) if emb_accs else 0
    ax.axhline(mean_acc, color='#e74c3c', ls='--', label=f'Mean {mean_acc:.1f}%')
    ax.set_xlabel('# Valid Frames'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Embryo Accuracy vs Sequence Length')
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    plt.tight_layout()
    out_path = output_dir / f'aggregate_{split}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Aggregate chart → {out_path}")

    all_preds_p1, all_probs_p1 = [], []
    for r in all_results:
        preds_p1_r = r.get('predictions_p1', r['predictions'])
        probs_p1_r = r.get('probs_p1_all')
        for i, g in enumerate(r['ground_truth']):
            if g is not None:
                all_preds_p1.append(preds_p1_r[i])
                if probs_p1_r is not None:
                    all_probs_p1.append(probs_p1_r[i])

    overall_acc = sum(p == g for p, g in zip(all_preds, all_gts)) / len(all_preds) * 100
    p1_acc = sum(p == g for p, g in zip(all_preds_p1, all_gts)) / len(all_preds_p1) * 100 if all_preds_p1 else 0

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
    print(f"\n{'='*60}")
    print(f"  Split: {split}  |  Embryos: {len(all_results)}  |  Frames: {len(all_preds)}")
    print(f"  {'Metric':<22} {'Phase 1':>12} {'Phase 2':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Top-1 Accuracy':<22} {p1_acc:>11.2f}% {overall_acc:>11.2f}%")
    print(f"  {'Top-3 Accuracy':<22} {top3_p1:>11.2f}% {top3_p2:>11.2f}%")
    print(f"  {'Macro F1':<22} {m_f1_p1:>12.4f} {macro_f1:>12.4f}")
    print(f"  {'Weighted F1':<22} {w_f1_p1:>12.4f} {w_f1_p2:>12.4f}")
    print(f"  {'-'*46}")
    print(f"  Per-class F1 (Phase 2):")
    for s, fv in zip(STAGE_NAMES, f1s):
        print(f"    {s:<6}: {fv:.3f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Chart A: Per-Embryo Attention Heatmap
# ---------------------------------------------------------------------------

def create_attention_heatmap(result: dict, output_path: str):
    from matplotlib.patches import Patch

    attn_list = result.get('attn_weights', [])
    gts       = result['ground_truth']
    preds     = result['predictions']
    preds_p1  = result.get('predictions_p1', preds)
    probs_p2  = result['probs_all']
    probs_p1  = result.get('probs_p1_all')
    eid       = result['embryo_id']

    valid = [(i, aw) for i, aw in enumerate(attn_list) if aw is not None]
    if len(valid) < 10:
        return

    max_w    = max(len(aw) for _, aw in valid)
    n_frames = len(valid)

    heatmap = np.zeros((max_w, n_frames))
    frame_indices = []
    for col, (fi, aw) in enumerate(valid):
        w = len(aw)
        heatmap[max_w - w:, col] = aw
        frame_indices.append(fi)

    fig, ax = plt.subplots(1, 1, figsize=(18, 5))
    fig.suptitle(f'Mamba2 + Attention Dynamics — {eid}  ({n_frames} frames)',
                 fontsize=13, fontweight='bold', y=1.02)

    im = ax.imshow(heatmap, aspect='auto', cmap='inferno',
                   interpolation='nearest', origin='lower')
    ax.set_ylabel(r'Time Window (bottom=$t-128$   top=$t$)')
    ax.set_xlabel(r'Processing Step (Frame $t$)')

    gt_at_valid = [gts[fi] for fi in frame_indices]
    for col in range(1, n_frames):
        g_prev, g_cur = gt_at_valid[col - 1], gt_at_valid[col]
        if g_prev is not None and g_cur is not None and g_prev != g_cur:
            ax.axvline(col, color='cyan', lw=1.5, ls='--', alpha=0.8)

    n_ticks = min(25, n_frames)
    tick_step = max(1, n_frames // n_ticks)
    tick_pos = list(range(0, n_frames, tick_step))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([frame_indices[t] for t in tick_pos], fontsize=6, rotation=45)
    ax.set_xlabel('Frame Index')
    plt.colorbar(im, ax=ax, label='Attention Weight', shrink=0.8, pad=0.01)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Attention Heatmap → {output_path}")


# ---------------------------------------------------------------------------
# Chart A2: Phase 2 Temporal Attention Explainer
# ---------------------------------------------------------------------------

def create_phase2_attention_explainer(result: dict, output_dir: str,
                                      sample_frames: list = None):
    """
    Paper-ready figure for 1 embryo: explaining what Phase 2 is attending to.

    Select 3-5 "interesting" frames (transition, error correction, high confidence)
    and plot for each frame:
      - Original embryo image (thumbnail)
      - Attention distribution bar: how many frames back the model is looking
      - P1 prediction vs P2 prediction vs GT
      - Confidence gauge

    Layout: mỗi frame 1 cột, 4 hàng:
      Row 0: Ảnh embryo + GT/P1/P2 annotation
      Row 1: Attention distribution (horizontal bar — W positions)
      Row 2: P1 vs P2 probability stacked bar cho 7 classes
      Row 3: Timeline context — frame position in the entire sequence

    If sample_frames=None, automatically select interesting frames:
      - 1 frame during stable stage (P1 correct, P2 correct)
      - 1 frame at transition (GT just changed)
      - 1 frame where P1 is wrong but P2 corrected it
      - 1 frame with highest P2 confidence
      - 1 frame with lowest P2 confidence
    """
    from matplotlib.patches import Patch, FancyBboxPatch
    from matplotlib.gridspec import GridSpec

    eid       = result['embryo_id']
    gts       = result['ground_truth']
    preds     = result['predictions']
    preds_p1  = result.get('predictions_p1', preds)
    confs     = result['confidences']
    probs_p2  = result['probs_all']
    probs_p1  = result.get('probs_p1_all')
    attn_list = result.get('attn_weights', [])
    files     = result['frame_files']
    N         = len(gts)

    # ── Select interesting frames ────────────────────────────────────────────────
    if sample_frames is None:
        candidates = {}

        # Transition frames
        for i in range(1, N):
            if (gts[i] is not None and gts[i - 1] is not None
                    and gts[i] != gts[i - 1]):
                candidates['transition'] = i

        # P1 sai, P2 đúng (error correction)
        for i in range(N):
            if (gts[i] is not None and preds_p1[i] != gts[i]
                    and preds[i] == gts[i]):
                candidates['corrected'] = i
                break

        # Stable (P1 đúng, P2 đúng, giữa stage)
        for i in range(N // 3, 2 * N // 3):
            if (gts[i] is not None and preds_p1[i] == gts[i]
                    and preds[i] == gts[i]):
                candidates['stable'] = i
                break

        # Highest P2 confidence
        valid_confs = [(i, c) for i, c in enumerate(confs) if gts[i] is not None]
        if valid_confs:
            candidates['high_conf'] = max(valid_confs, key=lambda x: x[1])[0]
            candidates['low_conf'] = min(valid_confs, key=lambda x: x[1])[0]

        sample_frames = list(dict.fromkeys(candidates.values()))  # deduplicate, keep order
        sample_labels = {v: k for k, v in candidates.items()}
    else:
        sample_labels = {f: f'frame_{f}' for f in sample_frames}

    # Filter: only keep frames with attention data
    sample_frames = [f for f in sample_frames
                     if f < len(attn_list) and attn_list[f] is not None]
    if len(sample_frames) < 2:
        return

    n_cols = len(sample_frames)

    fig = plt.figure(figsize=(4.5 * n_cols, 14))
    gs = GridSpec(4, n_cols, figure=fig, height_ratios=[2.5, 1.5, 1.5, 0.8],
                  hspace=0.35, wspace=0.3)

    fig.suptitle(f'Phase 2 (Mamba2 + Attention) Explainer — {eid}',
                 fontsize=14, fontweight='bold')

    for col, fi in enumerate(sample_frames):
        gt   = gts[fi]
        p1   = preds_p1[fi]
        p2   = preds[fi]
        conf = confs[fi]
        aw   = attn_list[fi]
        label = sample_labels.get(fi, '')

        gt_name = STAGE_NAMES[gt] if gt is not None else 'N/A'
        p1_name = STAGE_NAMES[p1]
        p2_name = STAGE_NAMES[p2]

        # ── Row 0: Embryo image + annotations ────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        try:
            img = np.array(Image.open(files[fi]).convert('RGB'))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center',
                    transform=ax.transAxes)
        ax.axis('off')

        # Color-coded border
        if p2 == gt:
            border_color = '#2ecc71'
        elif p1 != gt and p2 == gt:
            border_color = '#3498db'
        else:
            border_color = '#e74c3c'
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)

        title_parts = [f'Frame {fi}']
        if label:
            title_parts.append(f'({label})')
        ax.set_title(' '.join(title_parts), fontsize=9, fontweight='bold')

        # Annotation text below image
        txt = f'GT: {gt_name}\nP1: {p1_name}  P2: {p2_name}\nConf: {conf:.0%}'
        ax.text(0.5, -0.02, txt, transform=ax.transAxes, fontsize=8,
                ha='center', va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Row 1: Attention distribution ─────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        w = len(aw)
        positions = np.arange(w)
        # Color: gradient from old (gray) to current (orange)
        colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, w))
        ax.barh(positions, aw, color=colors, height=0.8)
        ax.set_xlabel('Attention Weight', fontsize=8)
        # Y axis: actual frame numbers in window
        # Position 0 = farthest (oldest) frame, W-1 = current frame
        n_yticks = min(6, w)
        ytick_step = max(1, w // n_yticks)
        ytick_pos = list(range(0, w, ytick_step))
        if (w - 1) not in ytick_pos:
            ytick_pos.append(w - 1)
        ax.set_yticks(ytick_pos)
        # Display actual frame numbers instead of t-W
        ytick_labels = []
        for p in ytick_pos:
            idx_in_valid = fi - (w - 1 - p)
            if idx_in_valid >= 0:
                real_f = _run_sort_key(files[idx_in_valid])
                ytick_labels.append(str(real_f))
            else:
                ytick_labels.append("")
        ax.set_yticklabels(ytick_labels, fontsize=6)
        ax.set_ylabel('Actual Frame Number', fontsize=7)
        ax.invert_yaxis()

        # Entropy annotation
        aw_safe = np.clip(aw, 1e-8, None)
        entropy = -np.sum(aw_safe * np.log(aw_safe))
        max_entropy = np.log(w) if w > 1 else 1
        ax.set_title(f'Attention Span (H={entropy:.2f}/{max_entropy:.2f})',
                     fontsize=8)
        ax.tick_params(labelsize=6)

        # ── Row 2: P1 vs P2 probability comparison ───────────────────────
        ax = fig.add_subplot(gs[2, col])
        x = np.arange(NUM_CLASSES)
        w_bar = 0.35
        p2_probs = probs_p2[fi]
        p1_probs_f = probs_p1[fi] if probs_p1 is not None else p2_probs

        bars_p1 = ax.bar(x - w_bar / 2, p1_probs_f, w_bar, color='#e67e22',
                         alpha=0.6, label='P1')
        bars_p2 = ax.bar(x + w_bar / 2, p2_probs, w_bar, color='#3498db',
                         alpha=0.8, label='P2')
def create_gradcam_visualization(result: dict, output_dir: str,
                                  window_size: int = 5):
    """
    Grad-CAM visualization for 3D MambaVision:
      - Separate each class into a distinct file
      - Each file includes the 5-frame clip
      - Stage 4 takes 5 separate frame heatmaps
    """
    from matplotlib.gridspec import GridSpec
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import os
    import cv2

    eid       = result['embryo_id']
    gts       = result['ground_truth']
    preds     = result['predictions']
    preds_p1  = result['predictions_p1']
    confs     = result['confidences']
    files     = result['frame_files']
    gradcam_list = result.get('gradcam', [])
    N         = len(gts)

    if not gradcam_list or all(g is None for g in gradcam_list):
        return

    # Find representative frame for each class (frame with GT and P2 correct, highest confidence)
    class_frames = {}
    for cls in range(NUM_CLASSES):
        candidates = []
        for i in range(N):
            if gts[i] == cls and preds[i] == cls:
                candidates.append((i, confs[i]))
        if candidates:
            best_frame = max(candidates, key=lambda x: x[1])[0]
            class_frames[cls] = best_frame

    def _draw_gradcam_for_class(cls: int):
        if cls not in class_frames:
            return

        fi = class_frames[cls]
        gt   = gts[fi]
        p1   = preds_p1[fi]
        p2   = preds[fi]
        conf = confs[fi]
        gc   = gradcam_list[fi] if fi < len(gradcam_list) else None
        aw   = result.get('attn_weights', [])[fi] if fi < len(result.get('attn_weights', [])) else None

        # ── Identify 5-frame clip ending at fi ──────────────────────────
        clip_indices = [max(0, fi - 4 + k) for k in range(5)]
        T = len(clip_indices)

        fig = plt.figure(figsize=(3.5 * T, 8))
        gs = GridSpec(3, T, figure=fig, height_ratios=[2.5, 2.5, 1.2], hspace=0.3, wspace=0.1)

        fig.suptitle(f'Grad-CAM Analysis — {eid} | Class: {STAGE_NAMES[cls]}\n'
                     f'GT: {STAGE_NAMES[gt] if gt is not None else "N/A"} | '
                     f'P1: {STAGE_NAMES[p1]} → P2: {STAGE_NAMES[p2]} | Conf: {conf:.0%}',
                     fontsize=14, fontweight='bold', y=0.96)

        border_color = '#2ecc71' if p2 == gt else ('#3498db' if p1 != gt and p2 == gt else '#e74c3c')

        for t, f_idx in enumerate(clip_indices):
            try:
                img_rgb = np.array(Image.open(files[f_idx]).convert('RGB'))
            except Exception:
                img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

            img_f = np.float32(cv2.resize(img_rgb, (224, 224))) / 255.0

            # ── Row 0: Original Image ───────────────────────────────────────
            ax0 = fig.add_subplot(gs[0, t])
            ax0.imshow(img_rgb)
            ax0.axis('off')
            # Show actual frame number from filename
            real_f_num = _run_sort_key(files[f_idx])
            ax0.set_title(f'Frame {real_f_num}', fontsize=10, fontweight='bold')
            for spine in ax0.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(3)

            # ── Row 1: Stage 4 Grad-CAM (Mamba 3D) ──────────────────────────
            ax1 = fig.add_subplot(gs[1, t])
            if gc is not None and isinstance(gc, dict) and 'stage4' in gc and t < len(gc['stage4']):
                overlay4 = show_cam_on_image(img_f, gc['stage4'][t], use_rgb=True, colormap=cv2.COLORMAP_JET)
                ax1.imshow(overlay4)
            else:
                ax1.text(0.5, 0.5, 'No Stage 4 CAM', ha='center', va='center', transform=ax1.transAxes)
            ax1.axis('off')
            if t == 0:
                ax1.text(-0.1, 0.5, 'Stage 4\n(Mamba 3D)', va='center', ha='right', transform=ax1.transAxes, fontweight='bold')

        # ── Row 2: Phase 2 Temporal Attention ───────────────────────────────
        ax3 = fig.add_subplot(gs[2, :])
        if aw is not None and len(aw) > 0:
            w_len = len(aw)
            # Get actual frame numbers of the current window for labels
            # aw corresponds to buffer: [f_{t-L+1}, ..., f_t]
            current_real_idx = _run_sort_key(files[fi])
            x_labels = []
            for i in range(w_len):
                idx_in_valid = fi - (w_len - 1 - i)
                real_val = _run_sort_key(files[idx_in_valid])
                x_labels.append(str(real_val))
            
            bars = ax3.bar(range(w_len), aw, color='#f1c40f', alpha=0.8, edgecolor='orange')
            
            # Set ticks
            step = max(1, w_len // 12)
            ax3.set_xticks(range(0, w_len, step))
            ax3.set_xticklabels([x_labels[i] for i in range(0, w_len, step)], fontsize=7)
            
            ax3.set_title(f'Phase 2 Mamba2→Attention (Frame {current_real_idx})', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Weight', fontsize=8)
            ax3.set_ylim(0, max(max(aw) * 1.2, 0.1))
            
            # Highlight current frame
            bars[-1].set_color('#e74c3c') 
            bars[-1].set_edgecolor('red')
        else:
            ax3.text(0.5, 0.5, 'No Phase 2 Attention Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.axis('off')


        output_path = os.path.join(output_dir, f'gradcam_class_{cls}_{STAGE_NAMES[cls]}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  📊 Grad-CAM (Class {cls}) → {output_path}")

    # Generate Grad-CAM for each valid class
    for cls in class_frames.keys():
        _draw_gradcam_for_class(cls)

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
    for i, name in enumerate(STAGE_NAMES):
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
        f'MambaVision (Phase 1) + CausalAttn (Phase 2)\n'
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

    Left panel: Error Flow Heatmap
      3D Matrix: when Phase 1 predicts X (incorrect), GT is Y,
      Phase 2 predicts Z. Displayed as a heatmap:
      - Row = (P1_pred → GT) pair (only when P1 is incorrect)
      - Column = Phase 2 output
      - Color = number of frames
      Shows: Does Phase 2 correct to GT or to another class?

    Right panel: Confusion Matrix Delta (P2 - P1)
      Difference in confusion matrices: positive = P2 predicts more than P1,
      negative = P2 predicts less than P1. Shows which way P2 "moves" predictions
      — expected: diagonal increases (correct), off-diagonal decreases.
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
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
                ax=ax, linewidths=0.4, vmin=0, vmax=1,
                cbar_kws={'shrink': 0.7})
    p1_acc = np.trace(cm_p1) / cm_p1.sum() * 100
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title(f'MambaEmbryo (Phase 1)\nAcc = {p1_acc:.1f}%')

    # ── Panel 2: Confusion Matrix Phase 2 (normalized) ──────────────────
    ax = axes[1]
    cm_p2 = confusion_matrix(all_gts, all_p2, labels=labels_range)
    cm_p2_norm = cm_p2.astype(float)
    for i in range(NUM_CLASSES):
        s = cm_p2_norm[i].sum()
        if s > 0:
            cm_p2_norm[i] /= s

    sns.heatmap(cm_p2_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
                ax=ax, linewidths=0.4, vmin=0, vmax=1,
                cbar_kws={'shrink': 0.7})
    p2_acc = np.trace(cm_p2) / cm_p2.sum() * 100
    ax.set_xlabel('Predicted')
    ax.set_ylabel('')
    ax.set_title(f'CausalRefiner (Phase 2)\nAcc = {p2_acc:.1f}%')

    # ── Panel 3: Improvement (P2 − P1) ───────────────────────────────────
    ax = axes[2]
    delta = (cm_p2_norm - cm_p1_norm) * 100  # percentage points

    # Annotation: show original value, but colormap uses "improvement" logic
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
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
                ax=ax, linewidths=0.4, vmin=-vmax, vmax=vmax,
                cbar_kws={'label': 'Improvement (pp)', 'shrink': 0.7})
    # Annotate with ORIGINAL delta values (not flipped) for readability
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = delta[i, j]
            ax.text(j + 0.5, i + 0.5, f'{val:+.1f}',
                    ha='center', va='center', fontsize=8,
                    color='black' if abs(improvement[i, j]) < vmax * 0.6 else 'white')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('')
    ax.set_title('Δ (P2 − P1) pp\nGreen = improved')

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
    X-axis = offset from transition (-30 to +30 frames)
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
    ax.plot(ox, p1_m, color='#e67e22', lw=2, label='Phase 1 (MambaVision)', marker='.',
            markersize=3)
    ax.plot(ox, p2_m, color='#3498db', lw=2, label='Phase 2 (CausalAttn)', marker='.',
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
# Chart D: Attention Dynamics — entropy + distance + P2 confidence over time
# ---------------------------------------------------------------------------

def create_attention_dynamics_chart(all_results: list, output_dir: Path, split: str,
                                    margin: int = 8):
    """
    2-panel aggregate figure:

    Left panel: Attention Entropy vs Confidence scatter (color = correct/wrong)
      - X-axis = attention entropy (high = broad view, low = focused)
      - Y-axis = P2 confidence
      - Color = correct (green) / incorrect (red)
      - Shows: when is the model confident? When attention is focused or broad?
        Is there a "low entropy + low confidence" region (model confused)?

    Right panel: Average attention entropy time-series around transition
      - Similar to transition landscape but for entropy instead of accuracy
      - Shows if the model "broadens its view" near transitions
    """
    # ── Collect per-frame metrics ─────────────────────────────────────────
    entropies_correct = []
    entropies_wrong   = []
    confs_correct     = []
    confs_wrong       = []

    # Transition entropy curve
    max_offset = 25
    offsets = list(range(-max_offset, max_offset + 1))
    entropy_at_offset = defaultdict(list)

    for r in all_results:
        gts       = r['ground_truth']
        preds     = r['predictions']
        confs     = r['confidences']
        attn_list = r.get('attn_weights', [])
        n         = len(gts)

        # Find transitions
        transitions = []
        for i in range(1, n):
            if gts[i] is not None and gts[i - 1] is not None and gts[i] != gts[i - 1]:
                transitions.append(i)

        for i, aw in enumerate(attn_list):
            if aw is None or gts[i] is None:
                continue
            w = len(aw)
            if w < 2:
                continue

            aw_safe = np.clip(aw, 1e-8, None)
            entropy = -np.sum(aw_safe * np.log(aw_safe))

            if preds[i] == gts[i]:
                entropies_correct.append(entropy)
                confs_correct.append(confs[i])
            else:
                entropies_wrong.append(entropy)
                confs_wrong.append(confs[i])

            # Entropy at offset from nearest transition
            for tp in transitions:
                offset = i - tp
                if -max_offset <= offset <= max_offset:
                    entropy_at_offset[offset].append(entropy)

    if not entropies_correct and not entropies_wrong:
        print("  ⚠️ No attention data — skipping attention dynamics chart")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Attention Dynamics Analysis — {split}',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: Entropy vs Confidence scatter ────────────────────────────
    ax = axes[0]
    if entropies_correct:
        ax.scatter(entropies_correct, confs_correct, s=8, alpha=0.15,
                   color='#2ecc71', label=f'Correct (n={len(entropies_correct)})',
                   rasterized=True)
    if entropies_wrong:
        ax.scatter(entropies_wrong, confs_wrong, s=8, alpha=0.25,
                   color='#e74c3c', label=f'Wrong (n={len(entropies_wrong)})',
                   rasterized=True)

    # Density contours if enough data
    if len(entropies_correct) > 100:
        from scipy.stats import gaussian_kde
        try:
            xy_c = np.vstack([entropies_correct, confs_correct])
            kde_c = gaussian_kde(xy_c)
            xg = np.linspace(min(entropies_correct), max(entropies_correct), 50)
            yg = np.linspace(0, 1, 50)
            X, Y = np.meshgrid(xg, yg)
            Z = kde_c(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, levels=5, colors='#27ae60', alpha=0.4, linewidths=0.8)
        except Exception:
            pass
    if len(entropies_wrong) > 50:
        from scipy.stats import gaussian_kde
        try:
            xy_w = np.vstack([entropies_wrong, confs_wrong])
            kde_w = gaussian_kde(xy_w)
            xg = np.linspace(min(entropies_wrong), max(entropies_wrong), 50)
            yg = np.linspace(0, 1, 50)
            X, Y = np.meshgrid(xg, yg)
            Z = kde_w(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, levels=5, colors='#c0392b', alpha=0.4, linewidths=0.8)
        except Exception:
            pass

    ax.set_xlabel('Attention Entropy (higher = broader temporal span)')
    ax.set_ylabel('Phase 2 Confidence')
    ax.set_title('Attention Breadth vs Model Confidence')
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(alpha=0.15)

    # ── Panel 2: Entropy around transitions ───────────────────────────────
    ax = axes[1]
    means = [np.mean(entropy_at_offset[o]) if entropy_at_offset[o] else np.nan
             for o in offsets]
    stds = [np.std(entropy_at_offset[o]) / max(np.sqrt(len(entropy_at_offset[o])), 1)
            if entropy_at_offset[o] else 0 for o in offsets]
    means = np.array(means)
    stds = np.array(stds)
    ox = np.array(offsets)

    ax.fill_between(ox, means - stds, means + stds, alpha=0.2, color='#9b59b6')
    ax.plot(ox, means, color='#9b59b6', lw=2.5, marker='.', markersize=4)
    ax.axvline(0, color='red', lw=1.5, ls='--', alpha=0.6)
    ax.axvspan(-margin, margin, alpha=0.06, color='red')
    ax.text(0.5, 0.02, 'transition', ha='center', fontsize=8, color='red',
            transform=ax.get_xaxis_transform())

    ax.set_xlim(-max_offset, max_offset)
    ax.set_xlabel('Offset from Transition Point (frames)')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Does the Model Broaden Attention Near Transitions?')
    ax.grid(alpha=0.15)

    plt.tight_layout()
    out = output_dir / f'attention_dynamics_{split}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Attention Dynamics → {out}")


# ---------------------------------------------------------------------------
# Chart E: Probability Smoothing — Phase 2 reduces oscillation of Phase 1
# ---------------------------------------------------------------------------

def create_probability_smoothing_chart(all_results: list, output_dir: Path, split: str):
    """
    Measure and visualize Phase 2 "smoothing" capability.

    Left panel: Prediction Stability
      Counts prediction flips between consecutive frames.
      Phase 1 (frame-level) often flips a lot / noisy.
      Phase 2 (temporal) should flip less / more stable, monotonic.
      Plotted as cumulative flip count over time (mean across embryos).

    Right panel: Backward Regression Rate
      Embryo development is unidirectional: tpnf → t2 → ... → tM+.
      "Backward regression" = model predicts lower stage than previous frame.
      Phase 2 with ordinal loss should have lower backward rate than Phase 1.
      Plotted as rate per stage transition.
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
    p = argparse.ArgumentParser(description='IVF 3D MambaVision Visualization')
    p.add_argument('--model',       required=True,
                   help='Path to checkpoint .pth')
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
                   help='Run a specific embryo ID (overrides split)')
    p.add_argument('--output_dir',  default='val/results_3d',
                   help='Directory to save results')
    p.add_argument('--num_frames',  type=int, default=5)
    p.add_argument('--phase2_ckpt', default=None, help='Path to Phase 2 best.pth')
    p.add_argument('--img_size',    type=int, default=224)
    p.add_argument('--fps',         type=int, default=10)
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no_video',    action='store_true',
                   help='Skip video generation (charts + CSV only)')
    p.add_argument('--no_gradcam',  action='store_true',
                   help='Skip Grad-CAM visualization (Stage 4 Attention)')
    p.add_argument('--max_embryos', type=int, default=None,
                   help='Limit number of embryos (for quick testing)')
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

    # Load model
    model = load_model_3d(args.model, args.num_frames, args.device)
    phase2_model = None
    if args.phase2_ckpt:
        phase2_model = load_phase2_model(args.phase2_ckpt, args.device)

    # Setup Grad-CAM for Stage 4 visualization
    gradcam_model = None if args.no_gradcam else setup_gradcam(model)[0]

    all_results = []

    for eid in tqdm(embryo_ids, desc='Embryos'):
        print(f"\n── {eid} ──")
        emb_dir = output_dir / eid
        emb_dir.mkdir(exist_ok=True)
        
        # Resume logic: if video exists, assume this embryo is already fully visualized.
        # We only need its predictions for the final aggregate metrics.
        # So we disable Grad-CAM and video generation for this embryo to save time.
        result = infer_embryo(
            model, phase2_model, data_root, ann_root, eid,
            args.num_frames, args.img_size, args.device,
            gradcam_model=gradcam_model,
        )
        if result is None:
            continue

        create_embryo_chart(result, str(emb_dir / 'chart.png'))

        # Grad-CAM visualization (Stage 4 attention)
        if gradcam_model is not None:
            create_gradcam_visualization(result, str(emb_dir))

        # Attention heatmap per-embryo (only if Phase 2 is present)
        if phase2_model is not None:
            create_attention_heatmap(result, str(emb_dir / 'attention_heatmap.png'))
            create_phase2_attention_explainer(
                result, str(emb_dir / 'phase2_explainer.png'))

        if not args.no_video:
            create_video(result, str(emb_dir / 'video.mp4'), fps=args.fps)

        # Clear gradcam from results to free RAM, avoiding OOM (Kill) errors
        if 'gradcam' in result:
            del result['gradcam']

        all_results.append(result)

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
