"""
balance_dataset_v5_2d.py — IVF Dataset Builder (1 image, class-balanced)
==============================================================================
v5.1 FIXES:
  1. BỎ pad frame cuối — clip < num_frames bị SKIP (tránh shortcut learning)
  2. cell_count fallback về stage count thay vì 0 khi frame không có annotation
  3. clip_counter across stages — tránh trùng clip_name giữa các stage cùng embryo
  4. Log diversity stats: bao nhiêu embryo đóng góp, overlap ratio

TARGET (default):
  Train: 3500 clips per class
  Val:   750 clips per class
  Test:  750 clips per class
"""

import os, re, csv, json, random, argparse, math, logging, shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Raw stages from CSV
# NOTE: tpna (pronuclei appearance) is EXCLUDED — only tpnf (pronuclei fading) used
RAW_STAGES = ['tpnf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tm', 'tsb']

# Final stage names with numbering (Standard, 11 classes)
VALID_STAGES = ['1-tPNf', '2-t2', '3-t3', '4-t4', '5-t5', '6-t6', '7-t7', '8-t8', '9-t9+', '10-tM', '11-tSB']

# Final stage names with numbering (Merged/Q1) — 7 classes
# tPNf, t2, t3+(t3,t4), t5+(t5,t6), t7+(t7,t8), t9+, tM+(tM,tSB)
VALID_STAGES_MERGED = ['1-tPNf', '2-t2', '3-t3+', '5-t5+', '7-t7+', '9-t9+', '10-tM+']

# Mappings for data processing
MAP_NORMAL = {
    'tpnf': '1-tPNf', 't2': '2-t2',
    't3': '3-t3', 't4': '4-t4', 't5': '5-t5', 't6': '6-t6',
    't7': '7-t7', 't8': '8-t8', 't9+': '9-t9+', 'tm': '10-tM', 'tsb': '11-tSB'
}

MAP_MERGED = {
    'tpnf': '1-tPNf',                          # tPNf only (tpna excluded)
    't2':   '2-t2',
    't3':   '3-t3+',   't4':  '3-t3+',         # t3+(t3, t4)
    't5':   '5-t5+',   't6':  '5-t5+',         # t5+(t5, t6)
    't7':   '7-t7+',   't8':  '7-t7+',         # t7+(t7, t8)
    't9+':  '9-t9+',
    'tm':   '10-tM+',  'tsb': '10-tM+',        # tM+(tM, tSB)
}

IMG_EXTS = {'.jpeg', '.jpg', '.png'}

CELL_COUNTS = {
    # raw stage keys
    'tpnf': 1, 't2': 2, 't3': 3, 't4': 4, 't5': 5, 't6': 6,
    't7': 7, 't8': 8, 't9+': 9, 'tm': 10, 'tsb': 11,
    # standard class keys
    '1-tPNf': 1, '2-t2': 2, '3-t3': 3, '4-t4': 4,
    '5-t5': 5, '6-t6': 6, '7-t7': 7, '8-t8': 8, '9-t9+': 9,
    '10-tM': 10, '11-tSB': 11,
    # merged 7-class keys
    '3-t3+': 3, '5-t5+': 5, '7-t7+': 7, '10-tM+': 10,
}

# ---------------------------------------------------------------------------
# BLACKLIST
# ---------------------------------------------------------------------------
BLACKLIST = set([
    'AAL839-6', 'AL884-2', 'ALR493-10', 'ALR493-6', 'AMT360-1-9', 'AS1015-2',
    'AS662-2', 'BA782-2', 'BC277-10', 'BC396-1', 'BE645-3', 'BJ3371-9',
    'BJ492-11', 'BJ492-8', 'BL285-1-3', 'BM016-2', 'BM256-1', 'BM256-4',
    'BM655-10', 'BM968-3', 'BN1010-5', 'BN356-3', 'BS1086-1', 'BS648-2-4',
    'BS648-7', 'BS836-11', 'BV646-6', 'CA063-10', 'CA063-6', 'CA364-7',
    'CA390-2', 'CA390-6', 'CA658-12', 'CA658-6', 'CA704-2', 'CAV074-1',
    'CAV074-3', 'CC938-4', 'CJ261-10', 'CK601-2', 'CM627-8', 'CM892-5',
    'CS552-2', 'CS552-4', 'CZ594-1', 'CZ594-5', 'DC307-1', 'DC307-2',
    'DH1012-1', 'DHDPI042-3', 'DHDPI042-7', 'DHDPI042-8', 'DJC641-4',
    'DL020-3', 'DM1046-12', 'DRL1048-1', 'DS17-2', 'DS61-1', 'DS666-9',
    'DS947-2', 'DSE41-2', 'DV210-4', 'DV210-8', 'DV305-3', 'EH315-3',
    'EH315-8', 'EJ393-3', 'FA344-5', 'FA662-6', 'FC1164-11', 'FE14-020',
    'FH658-4', 'FM1017-5', 'FM162-6', 'FN852-1', 'GA1087-6', 'GA122-8',
    'GA664-3', 'GA664-8', 'GC340-10', 'GC658-3', 'GC836-4', 'GC851-5',
    'GE1055-6', 'GE218-3', 'GE294-4', 'GE663-5', 'GF083-5', 'GF1042-1-3',
    'GJ165-5', 'GJ316-1', 'GM293-2', 'GM456-3', 'GS334-6', 'GS400-7',
    'GS415-5', 'GS430-2', 'GS490-2', 'GS490-7', 'GS490-_6', 'GS811-3',
    'GS826-2', 'GS980-2', 'HE444-3', 'HE444-4', 'HH569-2', 'HH569-4',
    'HM69-4', 'HS15-11', 'JE021-4', 'JV227-2', 'JV227-5', 'KF460-11',
    'KF460-7', 'KJ1077-3', 'LA367-4', 'LBE649-3', 'LBM519-1', 'LBM659-6',
    'LD400-1', 'LD400-6', 'LEG557-3', 'LM985-4', 'LNA592-8', 'LP284-3',
    'LS058-7', 'LS058-8', 'LS123-3', 'LS366-1', 'LTA908-2', 'LV488-7',
    'LV683-2-3', 'LV683-2-8', 'LV723-9', 'LZ865-2', 'MA1007-3', 'MA1059-3',
    'MA505-2', 'MAS094-5', 'MAS203-4', 'MAS203-6', 'MC427-1', 'MC833-6',
    'MC933-2', 'ME799-5', 'MM445-2-2', 'MM445-2-9', 'MM834-5', 'MM84-8',
    'MRA165-6', 'MRA165-7T', 'MV750-5', 'NC636-4', 'OA333-6', 'OC110-5',
    'OJ319-5', 'OJ319-7', 'PA214-5', 'PA276-3', 'PA916-1-10', 'PC55-2',
    'PC758-2', 'PC809-7', 'PE863-4', 'PG209-3', 'PH394-2', 'PI1004-3',
    'PMDPI029-1-1', 'PMDPI029-1-10', 'PMDPI029-1-11', 'PMDPI029-1-2',
    'PMDPI029-1-3', 'PMDPI029-1-6', 'PN636-1-6', 'PO13-3', 'PV361-2',
    'RA803-4', 'RC1103-1', 'RC545-2-8', 'RC545-2-9', 'RC755-4', 'RD167-7',
    'RL461-4', 'RL948-2', 'RLFS800-2', 'RM126-1', 'RM126-10', 'RM126-4',
    'RM126-6', 'RM126-9', 'RM29-5', 'RMN410-3', 'RV454-6', 'SC385-11',
    'SK308-10', 'SK308-7', 'SK902-1-8', 'SLM044-1-1', 'SM307-1-9', 'SN586-8',
    'ST586-7', 'TA12-1', 'TA757-9', 'TC1047-2', 'TD958-2-1', 'TJ899-2',
    'TL179-5', 'TM312-6', 'TM428-3', 'TN359-10', 'TN359-9', 'TN807-3',
    'TV654-4', 'UL050-_10', 'UL050-_9', 'VA225-6', 'VF269-7', 'VN484-1',
    'VS321-7', 'VS510-2', 'WA1014-3', 'WA402-7', 'WS531-4', 'ZL1077-1',
    'ZS435-6',
])


# ===========================================================================
# Parallel single-image writing helper (2D)
# ===========================================================================
def _write_single_image_2d(args):
    """Worker: load 1 frame, optionally augment, apply CLAHE+Sobel, save as single image."""
    (src_path, out_path, aug_id) = args
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src_path).convert('RGB')
        if aug_id >= 0:
            arr = np.array(img)
            # Reuse augment logic from augment_clip_5frames for single image
            aug_arr = _augment_single(arr, aug_id)
            img = Image.fromarray(aug_arr)
        img = apply_clahe_sobel(img)
        img.save(out_path, 'JPEG', quality=95)
        return str(out_path)
    except Exception as e:
        logging.error(f"Error writing {out_path}: {e}")
        return None


def _augment_single(img: np.ndarray, aug_id: int) -> np.ndarray:
    """16 deterministic augmentations for a single image array."""
    if aug_id == 0:
        aug = cv2.flip(img, 1)
    elif aug_id == 1:
        aug = cv2.flip(img, 0)
    elif aug_id == 2:
        aug = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif aug_id == 3:
        aug = cv2.rotate(img, cv2.ROTATE_180)
    elif aug_id == 4:
        aug = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif aug_id == 5:
        aug = cv2.flip(img, 1); aug = cv2.rotate(aug, cv2.ROTATE_90_CLOCKWISE)
    elif aug_id == 6:
        aug = cv2.flip(img, 0); aug = cv2.rotate(aug, cv2.ROTATE_90_CLOCKWISE)
    elif aug_id == 7:
        aug = cv2.flip(img, 1); aug = cv2.rotate(aug, cv2.ROTATE_180)
    elif aug_id == 8:
        aug = np.clip(img * 1.25, 0, 255).astype(np.uint8)
    elif aug_id == 9:
        aug = np.clip(img * 0.75, 0, 255).astype(np.uint8)
    elif aug_id == 10:
        m = img.mean(); aug = np.clip((img - m) * 1.4 + m, 0, 255).astype(np.uint8)
    elif aug_id == 11:
        m = img.mean(); aug = np.clip((img - m) * 0.6 + m, 0, 255).astype(np.uint8)
    elif aug_id == 12:
        aug = np.clip(img + np.random.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
    elif aug_id == 13:
        k = np.zeros((5, 5)); k[2, :] = 0.2; aug = cv2.filter2D(img, -1, k)
    elif aug_id == 14:
        aug = cv2.medianBlur(img, 5)
    elif aug_id == 15:
        if img.ndim == 3:
            ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
            aug = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
        else:
            aug = cv2.equalizeHist(img)
    else:
        aug = img
    return aug


# ===========================================================================
# CLAHE + Sobel
# ===========================================================================
def apply_clahe_sobel(img: Image.Image) -> Image.Image:
    """Áp dụng CLAHE + Sobel Edge để làm rõ ranh giới màng tế bào phôi.
    KHÔNG thay đổi kích thước ảnh gốc."""
    img_np = np.array(img)

    if len(img_np.shape) == 3:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        sobel_x = cv2.Sobel(cl, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(cl, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        limg = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        final = cv2.addWeighted(img_clahe, 0.85, edges_colored, 0.15, 0)

        return Image.fromarray(final)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(img_np)

        sobel_x = cv2.Sobel(cl, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(cl, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        final = cv2.addWeighted(cl, 0.85, edges, 0.15, 0)
        return Image.fromarray(final)


# ===========================================================================
# Augmentation for SHORT classes (5 frames đồng nhất)
# ===========================================================================
def augment_clip_5frames(clip_frames: List[Path], aug_id: int) -> List[np.ndarray]:
    """
    Augment 5 frames với cùng 1 transform (đồng nhất).
    aug_id: 0-15 → 16 variants
    
    Strategy:
      0-7:  Geometric (flip/rotate) + mild photometric
      8-15: Aggressive photometric (brightness/contrast/blur/noise)
    """
    imgs = []
    for fp in clip_frames:
        try:
            img = Image.open(fp).convert('RGB')
            imgs.append(np.array(img))
        except:
            return []
    if len(imgs) != 5:
        return []
    
    aug_imgs = []
    for img in imgs:
        if aug_id == 0:    # Flip H + brightness +10%
            aug = cv2.flip(img, 1)
            aug = np.clip(aug * 1.1, 0, 255).astype(np.uint8)
        elif aug_id == 1:  # Flip V + brightness -10%
            aug = cv2.flip(img, 0)
            aug = np.clip(img * 0.9, 0, 255).astype(np.uint8)
        elif aug_id == 2:  # Rotate 90 + contrast +20%
            aug = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            mean = aug.mean()
            aug = np.clip((aug - mean) * 1.2 + mean, 0, 255).astype(np.uint8)
        elif aug_id == 3:  # Rotate 180 + contrast -20%
            aug = cv2.rotate(img, cv2.ROTATE_180)
            mean = aug.mean()
            aug = np.clip((aug - mean) * 0.8 + mean, 0, 255).astype(np.uint8)
        elif aug_id == 4:  # Rotate 270 + Gaussian blur
            aug = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            aug = cv2.GaussianBlur(aug, (3, 3), 0.5)
        elif aug_id == 5:  # Flip H + Rotate 90 + sharpen
            aug = cv2.flip(img, 1)
            aug = cv2.rotate(aug, cv2.ROTATE_90_CLOCKWISE)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            aug = cv2.filter2D(aug, -1, kernel)
        elif aug_id == 6:  # Flip V + Rotate 90 + gamma 0.8
            aug = cv2.flip(img, 0)
            aug = cv2.rotate(aug, cv2.ROTATE_90_CLOCKWISE)
            aug = np.power(aug / 255.0, 0.8) * 255
            aug = aug.astype(np.uint8)
        elif aug_id == 7:  # Flip H + Rotate 180 + gamma 1.2
            aug = cv2.flip(img, 1)
            aug = cv2.rotate(aug, cv2.ROTATE_180)
            aug = np.power(img / 255.0, 1.2) * 255
            aug = aug.astype(np.uint8)
        elif aug_id == 8:  # Brightness +25%
            aug = np.clip(img * 1.25, 0, 255).astype(np.uint8)
        elif aug_id == 9:  # Brightness -25%
            aug = np.clip(img * 0.75, 0, 255).astype(np.uint8)
        elif aug_id == 10: # Contrast +40%
            mean = img.mean()
            aug = np.clip((img - mean) * 1.4 + mean, 0, 255).astype(np.uint8)
        elif aug_id == 11: # Contrast -40%
            mean = img.mean()
            aug = np.clip((img - mean) * 0.6 + mean, 0, 255).astype(np.uint8)
        elif aug_id == 12: # Gaussian noise σ=10
            noise = np.random.normal(0, 10, img.shape)
            aug = np.clip(img + noise, 0, 255).astype(np.uint8)
        elif aug_id == 13: # Motion blur (horizontal)
            kernel = np.zeros((5, 5))
            kernel[2, :] = 1 / 5
            aug = cv2.filter2D(img, -1, kernel)
        elif aug_id == 14: # Median blur (remove noise)
            aug = cv2.medianBlur(img, 5)
        elif aug_id == 15: # Histogram equalization
            if len(img.shape) == 3:
                ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                aug = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            else:
                aug = cv2.equalizeHist(img)
        else:
            aug = img
        aug_imgs.append(aug)
    return aug_imgs


# ===========================================================================
# 2D single-image strided sampling
# ===========================================================================
def select_frames_strided_2d(frames: List[Path], target_n: int) -> List[Path]:
    """
    Chọn tối đa `target_n` frame từ danh sách frames của một stage trong một embryo.

    Logic:
      1. Luôn lấy frame đầu (index 0) và frame cuối (index -1) — transition frames.
      2. Tính stride = max(1, (n-1) // (target_n-1)) để lấy đều nhau.
      3. Lấy các frame theo stride, đảm bảo không liền kề khi n > target_n.
      4. Nếu n <= target_n: lấy tất cả (không cần stride).

    VD: n=100, target_n=10 → stride=11 → lấy [0,11,22,...,99] (10 frames)
    VD: n=5,   target_n=10 → lấy tất cả 5 frames
    """
    n = len(frames)
    if n == 0:
        return []
    if n <= target_n:
        return frames[:]   # lấy tất cả

    # Tính stride để lấy đúng target_n frames, bao gồm đầu và cuối
    stride = max(1, (n - 1) // (target_n - 1))
    selected = []
    idx = 0
    while idx < n and len(selected) < target_n - 1:
        selected.append(frames[idx])
        idx += stride
    # Luôn thêm frame cuối
    if frames[-1] not in selected:
        selected.append(frames[-1])
    return selected


# ===========================================================================
# Sparse clip generation (giữ lại cho tương thích, không dùng trong 2D pipeline)
# ===========================================================================
def generate_sparse_clips(frames: List[Path], num_frames: int, start_stride: int) -> List[Tuple[List[Path], int]]:
    """
    Sinh TẤT CẢ sparse clips từ stride=start_stride giảm dần đến stride=1.
    Mỗi stride, dịch start thêm 1 để phủ toàn bộ vị trí bắt đầu hợp lệ.

    VD: num_frames=5, start_stride=15, n=100:
      stride=15: [0,15,30,45,60], [1,16,31,46,61], ..., [39,54,69,84,99]  → 40 clips
      stride=14: [0,14,28,42,56], [1,15,29,43,57], ..., [43,57,71,85,99]  → 44 clips
      ...
      stride=1:  [0,1,2,3,4], [1,2,3,4,5], ..., [95,96,97,98,99]          → 96 clips

    Thứ tự: stride DESC → clips đa dạng nhất (high-stride) được ưu tiên trước.
    Round-robin selection phía trên sẽ tự nhiên ưu tiên các clips này.

    KHÔNG có clip trùng: mỗi (i, s) cho tập frame khác nhau (dãy số học unique).
    Total clips per stride s = max(0, n - (num_frames-1)*s)
    """
    n = len(frames)
    all_clips = []
    for s in range(start_stride, 0, -1):
        clip_span = (num_frames - 1) * s
        if clip_span >= n:
            continue  # stride quá lớn so với n_frames → bỏ qua, thử stride nhỏ hơn
        for i in range(n - clip_span):
            clip = [frames[i + k * s] for k in range(num_frames)]
            all_clips.append((clip, s))
    return all_clips


def compute_per_stage_auto_stride(
    ann_root: Path,
    active_stages: List[str],
    num_frames: int,
    merge_classes: bool,
    min_clips_per_embryo: int = 5,
) -> Dict[str, int]:
    """
    Tự tính stride tối ưu cho từng stage từ annotation CSVs.

    Strategy: Với mỗi stage, thu thập độ dài segment của tất cả embryo,
    lấy median làm đại diện, sau đó tính stride tối đa để tạo được
    ít nhất `min_clips_per_embryo` clips trên median embryo:

        n_clips = n - (num_frames - 1) * stride >= min_clips_per_embryo
        => stride <= (median_n - min_clips_per_embryo) / (num_frames - 1)

    Returns: Dict {stage: start_stride}
    """
    import statistics as _stats

    stage_lengths: Dict[str, List[int]] = {s: [] for s in active_stages}

    for csv_path in ann_root.glob('*_phases.csv'):
        embryo_stage_raw: Dict[str, int] = {}  # raw_stage -> total_frames
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                for row in csv.reader(f):
                    if len(row) < 3:
                        continue
                    stg = row[0].strip().lower()
                    if stg not in RAW_STAGES:
                        continue
                    try:
                        s0, e0 = int(row[1]) - 1, int(row[2]) - 1
                        if e0 >= s0:
                            embryo_stage_raw[stg] = embryo_stage_raw.get(stg, 0) + (e0 - s0 + 1)
                    except ValueError:
                        pass
        except Exception:
            continue

        # Merge if needed (prefix names already in maps)
        for stg, length in embryo_stage_raw.items():
            tgt_stg = (MAP_MERGED if merge_classes else MAP_NORMAL).get(stg, stg)
            if tgt_stg in stage_lengths:
                # Cộng dồn vào stage đã merge (cùng embryo)
                if stage_lengths[tgt_stg] and hasattr(stage_lengths[tgt_stg], '_last_embryo'):
                    stage_lengths[tgt_stg][-1] += length
                else:
                    stage_lengths[tgt_stg].append(length)

    # Tính auto_stride per stage
    stage_strides: Dict[str, int] = {}
    print("\n  [AUTO-STRIDE] Per-stage optimal stride:")
    print(f"    {'Stage':>8} | {'Count':>5} | {'Median':>6} | {'AutoStride':>10}")
    print(f"    {'-'*8}-+-{'-'*5}-+-{'-'*6}-+-{'-'*10}")
    for stage in active_stages:
        lens = stage_lengths[stage]
        if not lens:
            stage_strides[stage] = 1
            print(f"    {stage:>8} | {'N/A':>5} | {'N/A':>6} | {'1 (fallback)':>10}")
            continue
        median_len = _stats.median(lens)
        # stride tối đa để n_clips >= min_clips_per_embryo
        # n_clips = median_len - (num_frames-1)*s >= min_clips_per_embryo
        # => s <= (median_len - min_clips_per_embryo) / (num_frames - 1)
        if num_frames > 1:
            max_s = max(1, int((median_len - min_clips_per_embryo) / (num_frames - 1)))
        else:
            max_s = max(1, int(median_len) - 1)
        stage_strides[stage] = max_s
        print(f"    {stage:>8} | {len(lens):>5} | {int(median_len):>6} | {max_s:>10}")
    print()
    return stage_strides


# ===========================================================================
# File utilities
# ===========================================================================
def _run_sort_key(p: Path) -> int:
    m = re.search(r'RUN(\d+)', p.stem, re.IGNORECASE)
    if m: return int(m.group(1))
    nums = re.findall(r'\d+', p.stem)
    return int(nums[-1]) if nums else 0

def get_all_frames(data_root: Path, embryo_id: str) -> List[Path]:
    d = data_root / embryo_id
    if not d.exists(): return []
    frames = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]
    frames.sort(key=_run_sort_key)
    return frames

def parse_csv(ann_root: Path, embryo_id: str) -> Dict[str, Tuple[int, int]]:
    csv_path = ann_root / f"{embryo_id}_phases.csv"
    if not csv_path.exists(): return {}
    raw_entries: Dict[str, Tuple[int, int]] = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) < 3: continue
            stage = row[0].strip().lower()
            if stage not in RAW_STAGES: continue
            try:
                s = int(row[1].strip()) - 1
                e = int(row[2].strip()) - 1
                if e >= s:
                    raw_entries[stage] = (s, e)
            except ValueError:
                pass
    return raw_entries


# ===========================================================================
# Write clips — NO padding, skip short clips
# ===========================================================================
def write_5frame_clips(
    split: str,
    stage: str,
    embryo_id: str,
    clip_list: List[List[Path]],
    out_data: Path,
    frame_to_cc: Dict[Path, int],
    annotations: list,
    start_idx: int,
    dry_run: bool,
    num_frames: int = 5,
    augment: bool = False,
) -> int:
    """
    Ghi clips N-frame từ danh sách clip.
    augment=True: Tạo thêm 16 augmented variants (flip/rotate/photometric đồng nhất 5 frames)
    
    Output: {split}/{stage}/{embryo_id}-{k}/frame_{0..N-1}.jpeg
    """
    out_stage_dir = out_data / split / stage
    if not dry_run:
        out_stage_dir.mkdir(parents=True, exist_ok=True)

    written_count = 0
    clip_idx = start_idx

    for clip_frames in clip_list:
        # SKIP clips that don't have exactly num_frames (no padding)
        if len(clip_frames) != num_frames:
            continue

        clip_folder = out_stage_dir / f"{embryo_id}-{clip_idx}"

        if not dry_run:
            clip_folder.mkdir(parents=True, exist_ok=True)

            success = True
            for frame_idx, fp in enumerate(clip_frames):
                try:
                    img = Image.open(fp).convert('RGB')
                    img = apply_clahe_sobel(img)
                    img.save(
                        clip_folder / f"frame_{frame_idx}.jpeg",
                        'JPEG', quality=95
                    )
                except Exception as e:
                    logging.error(f"Lỗi frame {fp} clip {embryo_id}-{clip_idx}: {e}")
                    success = False
                    break

            if not success:
                shutil.rmtree(clip_folder, ignore_errors=True)
                clip_idx += 1
                continue

        # Cell count: dùng frame_to_cc, fallback về CELL_COUNTS[stage]
        stage_fallback = CELL_COUNTS.get(stage, 1)
        counts = [frame_to_cc.get(fp, stage_fallback) for fp in clip_frames]
        avg_count = sum(counts) / len(counts)

        annotations.append({
            'clip_name': f"{embryo_id}-{clip_idx}",
            'split': split,
            'stage': stage,
            'cell_count': round(avg_count, 1),
            'embryo_id': embryo_id,
        })

        clip_idx += 1
        written_count += 1
        
        # Augmentation (16 variants)
        if augment and not dry_run:
            for aug_id in range(16):
                aug_imgs = augment_clip_5frames(clip_frames, aug_id)
                if not aug_imgs:
                    continue
                aug_folder = out_stage_dir / f"{embryo_id}-{clip_idx}_aug{aug_id}"
                aug_folder.mkdir(parents=True, exist_ok=True)
                try:
                    for frame_idx, aug_img in enumerate(aug_imgs):
                        img_pil = Image.fromarray(aug_img)
                        img_pil = apply_clahe_sobel(img_pil)
                        img_pil.save(aug_folder / f"frame_{frame_idx}.jpeg", 'JPEG', quality=95)
                    annotations.append({
                        'clip_name': f"{embryo_id}-{clip_idx}_aug{aug_id}",
                        'split': split, 'stage': stage,
                        'cell_count': round(avg_count, 1),
                        'embryo_id': embryo_id,
                    })
                    clip_idx += 1
                    written_count += 1
                except Exception as e:
                    logging.error(f"Aug{aug_id} fail {embryo_id}-{clip_idx}: {e}")
                    shutil.rmtree(aug_folder, ignore_errors=True)

    return written_count


# ===========================================================================
# Main pipeline — 2D single-image, stride-based sampling
# ===========================================================================
def run_pipeline(
    data_root: str,
    ann_root: str,
    embryo_list: Optional[str],
    out_data: str,
    target_train: int,
    target_val: int,
    target_test: int,
    dry_run: bool,
    merge_classes: bool = True,
    seed: int = 42,
):
    data_root = Path(data_root)
    ann_root  = Path(ann_root)
    out_data  = Path(out_data)

    if not dry_run:
        out_data.mkdir(parents=True, exist_ok=True)

    log_file = out_data / "balance_dataset_2d.log" if not dry_run else Path("dry_run_2d.log")
    logging.basicConfig(
        filename=log_file, filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    active_stages = VALID_STAGES_MERGED if merge_classes else VALID_STAGES

    print("=" * 70)
    print(" IVF 2D Dataset Builder — single-image, stride-based, 7-class")
    print("=" * 70)
    print(f"Merge classes  : {merge_classes}")
    print(f"Active stages  : {active_stages}")
    print(f"Targets        : Train={target_train}, Val={target_val}, Test={target_test}")
    print(f"Output         : {out_data}\n")

    # ── 1. Read embryo list & splits ─────────────────────────────────────────
    if not embryo_list:
        raise ValueError("--embryo_list is required!")

    split_of: Dict[str, str] = {}
    with open(embryo_list) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and set(raw.keys()) <= {'train', 'val', 'test'}:
        for sp, ids in raw.items():
            for eid in ids:
                split_of[eid] = sp

    all_ids = [e for e in split_of if e not in BLACKLIST]
    split_of = {e: sp for e, sp in split_of.items() if e not in BLACKLIST}

    # ── 2. Scan embryos ───────────────────────────────────────────────────────
    print("[1/4] Scanning embryos...")
    embryo_info: Dict = {}

    for eid in tqdm(all_ids, desc="Scan"):
        all_frames = get_all_frames(data_root, eid)
        stage_ranges = parse_csv(ann_root, eid)
        if not all_frames or not stage_ranges:
            continue

        sfmap: Dict[str, List[Path]] = {}
        for stg, (s0, e0) in stage_ranges.items():
            s0 = max(0, min(s0, len(all_frames) - 1))
            e0 = max(0, min(e0, len(all_frames) - 1))
            sf = all_frames[s0: e0 + 1]
            if not sf:
                continue
            tgt_stg = (MAP_MERGED if merge_classes else MAP_NORMAL).get(stg)
            if tgt_stg is None:
                continue
            if tgt_stg not in sfmap:
                sfmap[tgt_stg] = []
            sfmap[tgt_stg].extend(sf)

        # Deduplicate while preserving order
        for stg in sfmap:
            seen = set()
            deduped = []
            for p in sfmap[stg]:
                if p not in seen:
                    seen.add(p)
                    deduped.append(p)
            sfmap[stg] = deduped

        if sfmap:
            embryo_info[eid] = {'stage_frames': sfmap}

    print(f"  Valid embryos: {len(embryo_info)}\n")

    # ── 3. Build frame pools per split per class ──────────────────────────────
    print("[2/4] Building frame pools (stride-based per embryo)...")
    splits = ['train', 'val', 'test']
    targets_by_split = {'train': target_train, 'val': target_val, 'test': target_test}

    # pools[split][stage] = list of (src_path, embryo_id)
    pools: Dict[str, Dict[str, List[Tuple[Path, str]]]] = {
        sp: {stg: [] for stg in active_stages} for sp in splits
    }

    for eid, info in embryo_info.items():
        sp = split_of.get(eid)
        if sp not in splits:
            continue
        target = targets_by_split[sp]
        n_embryos_approx = sum(
            1 for e in embryo_info
            if split_of.get(e) == sp
        )
        # Per-embryo quota: how many frames to take from this embryo per stage
        # = ceil(target / n_embryos) — ensures diversity across embryos
        per_embryo_quota = max(1, math.ceil(target / max(n_embryos_approx, 1)))

        for stg, frames in info['stage_frames'].items():
            if stg not in active_stages:
                continue
            # Stride-based selection: first + last + strided middle
            selected = select_frames_strided_2d(frames, per_embryo_quota)
            for fp in selected:
                pools[sp][stg].append((fp, eid))

    print("  Pool sizes (before balancing):")
    print(f"  {'Stage':<12} {'Train':>8} {'Val':>7} {'Test':>7}")
    print(f"  {'-'*38}")
    for stg in active_stages:
        tr = len(pools['train'][stg])
        va = len(pools['val'][stg])
        te = len(pools['test'][stg])
        print(f"  {stg:<12} {tr:>8,} {va:>7,} {te:>7,}")

    # ── 4. Select + write images ──────────────────────────────────────────────
    print("\n[3/4] Writing images...")
    rng = random.Random(seed)
    global_annotations = []

    # Load existing annotations to skip already-written images
    existing_csv = out_data / "annotations.csv"
    existing_names: set = set()
    if existing_csv.exists():
        with open(existing_csv, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                existing_names.add(row['image_name'])
        print(f"  Found {len(existing_names):,} existing images, will skip them\n")

    for sp in splits:
        target = targets_by_split[sp]
        print(f"  [{sp.upper()}] target={target:,}/class")

        for stg in active_stages:
            pool = pools[sp][stg].copy()
            rng.shuffle(pool)
            real_count = len(pool)

            # ── Round-robin across embryos for diversity ──────────────────────
            embryo_pool: Dict[str, List[Path]] = defaultdict(list)
            for fp, eid in pool:
                embryo_pool[eid].append(fp)

            selected_real: List[Tuple[Path, str]] = []
            embryo_keys = sorted(embryo_pool.keys())
            rng.shuffle(embryo_keys)
            max_per = max((len(v) for v in embryo_pool.values()), default=0)
            for i in range(max_per):
                for eid in embryo_keys:
                    if i < len(embryo_pool[eid]):
                        selected_real.append((embryo_pool[eid][i], eid))
            selected_real = selected_real[:target]

            # Build write tasks for real frames
            tasks = []
            for idx, (src, eid) in enumerate(selected_real):
                # Output: {split}/{stage}/{embryo_id}-{original_filename}
                # e.g. train/1-tPNf/AA83-7-D2013.01.28_S0717_I132_WELL7_RUN42.jpeg
                img_name = f"{eid}-{Path(src).name}"
                out_path = out_data / sp / stg / img_name
                ann = {'image_name': img_name, 'split': sp, 'stage': stg,
                       'embryo_id': eid, 'is_aug': False}
                global_annotations.append(ann)
                if img_name not in existing_names and not dry_run:
                    tasks.append((str(src), str(out_path), -1))

            written_real = len(selected_real)
            shortfall = target - written_real

            # ── Augmentation to fill shortfall ────────────────────────────────
            aug_count = 0  # track aug count for display (works in dry_run too)
            aug_tasks = []
            if shortfall > 0:
                aug_pool = [(fp, eid) for fp, eid in selected_real]
                rng.shuffle(aug_pool)
                for aug_idx in range(shortfall):
                    src, eid_aug = aug_pool[aug_idx % len(aug_pool)] if aug_pool else (None, 'unk')
                    if src is None:
                        break
                    aug_id = aug_idx % 16
                    # e.g. AA83-7-D2013.01.28_S0717_I132_WELL7_RUN42_aug3.jpeg
                    stem = Path(src).stem
                    img_name = f"{eid_aug}-{stem}_aug{aug_id}.jpeg"
                    out_path = out_data / sp / stg / img_name
                    ann = {'image_name': img_name, 'split': sp, 'stage': stg,
                           'embryo_id': eid_aug, 'is_aug': True}
                    global_annotations.append(ann)
                    aug_count += 1
                    if img_name not in existing_names and not dry_run:
                        aug_tasks.append((str(src), str(out_path), aug_id))

            # Write in parallel
            all_tasks = tasks + aug_tasks
            if not dry_run and all_tasks:
                with Pool(min(cpu_count(), 8)) as pool_mp:
                    list(tqdm(
                        pool_mp.imap(_write_single_image_2d, all_tasks),
                        total=len(all_tasks),
                        desc=f"    {sp}/{stg}",
                        leave=False
                    ))

            total_out = written_real + aug_count
            status = "✓" if total_out >= target else "⚠"
            print(f"    {stg:<12}: real={written_real:>6,}  aug={aug_count:>5,}  "
                  f"total={total_out:>6,}/{target:,}  {status}")
            logging.info(f"{sp}/{stg}: real={written_real}, aug={len(aug_tasks)}, total={total_out}")

    # ── 5. Write annotations CSV ──────────────────────────────────────────────
    print("\n[4/4] Writing annotations.csv...")
    if not dry_run:
        new_anns = [r for r in global_annotations if r['image_name'] not in existing_names]
        mode = 'a' if existing_csv.exists() else 'w'
        with open(existing_csv, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=['image_name', 'split', 'stage', 'embryo_id', 'is_aug'],
                extrasaction='ignore')
            if mode == 'w':
                writer.writeheader()
            writer.writerows(new_anns)
        total_in_csv = len(existing_names) + len(new_anns)
        print(f"  Added {len(new_anns):,} rows. Total in CSV: {total_in_csv:,}")
        names = [r['image_name'] for r in global_annotations]
        dupes = len(names) - len(set(names))
        if dupes:
            print(f"  ⚠ WARNING: {dupes} duplicate image_names!")
        else:
            print(f"  ✓ All {len(names):,} image_names unique")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)
    summary: Dict[str, Dict[str, int]] = {sp: defaultdict(int) for sp in splits}
    for r in global_annotations:
        summary[r['split']][r['stage']] += 1
    print(f"  {'Stage':<12} {'Train':>8} {'Val':>7} {'Test':>7} {'Total':>8}")
    print(f"  {'-'*48}")
    for stg in active_stages:
        tr = summary['train'].get(stg, 0)
        va = summary['val'].get(stg, 0)
        te = summary['test'].get(stg, 0)
        print(f"  {stg:<12} {tr:>8,} {va:>7,} {te:>7,} {tr+va+te:>8,}")
    print("\nDONE.")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    p = argparse.ArgumentParser(
        description="IVF 2D Dataset Builder — single-image, stride-based, 7-class balanced")
    p.add_argument('--data_root',     required=True,  help='Embryo image folder')
    p.add_argument('--ann_root',      required=True,  help='Annotation CSV folder')
    p.add_argument('--out_data',      required=True,  help='Output dataset folder')
    p.add_argument('--embryo_list',   required=True,  help='JSON file with train/val/test embryo IDs')
    p.add_argument('--target_train',  type=int, default=3500)
    p.add_argument('--target_val',    type=int, default=750)
    p.add_argument('--target_test',   type=int, default=750)
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--merge_classes', action='store_true', default=True,
                   help='Merge into 7 classes (default: True)')
    p.add_argument('--dry_run',       action='store_true')
    args = p.parse_args()

    run_pipeline(
        data_root=args.data_root,
        ann_root=args.ann_root,
        embryo_list=args.embryo_list,
        out_data=args.out_data,
        target_train=args.target_train,
        target_val=args.target_val,
        target_test=args.target_test,
        dry_run=args.dry_run,
        merge_classes=args.merge_classes,
        seed=args.seed,
    )

if __name__ == '__main__':
    main()
