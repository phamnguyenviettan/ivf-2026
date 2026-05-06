"""
balance_dataset_v5_timelapevideo.py — IVF Timelapse Video Dataset Builder
===========================================================================
Tạo dataset video time-lapse cho Phase 2 training.

Mục tiêu:
  - Mỗi embryo là 1 sequence đầy đủ theo thứ tự thời gian
  - Train: cân bằng số frame giữa các giai đoạn TRONG mỗi embryo
    * Tính trung bình số frame trên tất cả stage của embryo đó
    * Stage < trung bình: phải nhân bản frame để đạt trung bình, lưu ý phải xoay hoặc làm gì đó không giới hạn ví dụ 360 độ hoặc lật... nha
    * Stage > trung bình: giữ tất cả 
    * Luôn giữ frame đầu + frame cuối mỗi stage (transition frames)
  - Val/Test: giữ nguyên toàn bộ frame, không cân bằng

Output structure (giống raw dataset):
  out_data/
    images/
      {embryo_id}/          ← folder ảnh theo thứ tự thời gian
        {original_filename}.jpeg
    annotations/
      {embryo_id}_phases.csv  ← annotation với RUN index mới (re-indexed)
    splits.json               ← train/val/test embryo IDs
    dataset_stats.csv         ← thống kê số frame per stage per embryo

Annotation CSV format (giống raw):
  stage_name, start_run, end_run
  (1-indexed, inclusive — giống embryo_dataset_annotations)
"""

import os, re, csv, json, random, argparse, math, logging, shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import concurrent.futures

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
        # Hỗ trợ vô hạn biến thể cho aug_id >= 16
        np.random.seed(aug_id) # Giữ tính deterministic cho mỗi aug_id
        aug = img.copy()
        
        # Lật ngẫu nhiên
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 1)
        if np.random.rand() > 0.5:
            aug = cv2.flip(aug, 0)
            
        # Xoay ngẫu nhiên (1-359 độ)
        angle = np.random.uniform(1, 360)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Chỉnh sáng/tương phản nhẹ
        alpha = np.random.uniform(0.85, 1.15)
        beta = np.random.randint(-15, 15)
        aug = np.clip(aug * alpha + beta, 0, 255).astype(np.uint8)

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
# Per-embryo stage-balanced frame selection (TRAIN only)
# ===========================================================================

def select_frames_stage_balanced(
    stage_frames: Dict[str, List[Path]],
    active_stages: List[str],
    transition_keep: int = 5,
) -> Dict[str, List[Tuple[Path, int]]]:
    """
    Cân bằng số frame giữa các stage TRONG 1 embryo (chỉ dùng cho train).

    Logic:
      1. quota = mean(len(frames) for all stages in embryo)
      2. Stage >= quota: giữ tất cả frame gốc (aug_id=-1)
         - Luôn đảm bảo transition_keep frame đầu + cuối có mặt
      3. Stage < quota: giữ tất cả gốc + nhân bản bằng aug để đạt quota
         - Ưu tiên aug transition frames (đầu + cuối stage)
         - aug_id cycle 0..15 để đa dạng transform (flip, rotate, brightness...)
         - Không lặp cùng (frame, aug_id) trong 1 stage

    Returns:
        {stage: [(frame_path, aug_id)]}
        aug_id=-1: frame gốc | aug_id=0..15: augmented variant
    """
    present = {s: f for s, f in stage_frames.items() if s in active_stages and len(f) > 0}
    if not present:
        return {}

    lengths = [len(f) for f in present.values()]
    quota = int(sum(lengths) / len(lengths))
    quota = max(quota, transition_keep * 2 + 1)

    result: Dict[str, List[Tuple[Path, int]]] = {}

    for stg, frames in present.items():
        n = len(frames)
        entries: List[Tuple[Path, int]] = []

        # Tất cả frame gốc trước
        for fp in frames:
            entries.append((fp, -1))

        if n < quota:
            # Nhân bản bằng aug để đạt quota
            shortfall = quota - n

            # Pool aug: ưu tiên transition frames (đầu + cuối)
            t_head = frames[:transition_keep]
            t_tail = frames[max(0, n - transition_keep):]
            middle  = frames[transition_keep: max(transition_keep, n - transition_keep)]
            # Xen kẽ: transition trước, middle sau — lặp lại nếu cần
            aug_pool = (t_head + t_tail + middle) * (shortfall // max(n, 1) + 2)

            aug_id_counter = 0
            filled = 0
            while filled < shortfall:
                src = aug_pool[filled % len(aug_pool)]
                aug_id = aug_id_counter
                entries.append((src, aug_id))
                filled += 1
                aug_id_counter += 1

        result[stg] = entries

    return result


def select_frames_full(
    stage_frames: Dict[str, List[Path]],
    active_stages: List[str],
) -> Dict[str, List[Tuple[Path, int]]]:
    """Val/Test: giữ nguyên toàn bộ frame gốc, không cân bằng, aug_id=-1."""
    return {
        s: [(fp, -1) for fp in f]
        for s, f in stage_frames.items()
        if s in active_stages and len(f) > 0
    }


# ===========================================================================
# Main pipeline — timelapse video dataset
# ===========================================================================

def run_pipeline(
    data_root: str,
    ann_root: str,
    embryo_list: Optional[str],
    out_data: str,
    dry_run: bool,
    merge_classes: bool = True,
    seed: int = 42,
    transition_keep: int = 5,
):
    data_root = Path(data_root)
    ann_root  = Path(ann_root)
    out_data  = Path(out_data)

    if not dry_run:
        out_data.mkdir(parents=True, exist_ok=True)

    log_file = out_data / "build_timelapse.log" if not dry_run else Path("dry_run_timelapse.log")
    logging.basicConfig(
        filename=log_file, filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    active_stages = VALID_STAGES_MERGED if merge_classes else VALID_STAGES

    print("=" * 70)
    print(" IVF Timelapse Video Dataset Builder")
    print("=" * 70)
    print(f"Merge classes    : {merge_classes}")
    print(f"Active stages    : {active_stages}")
    print(f"Transition keep  : {transition_keep} frames đầu/cuối mỗi stage")
    print(f"Train balancing  : mean-quota per embryo")
    print(f"Val/Test         : full sequence, no balancing")
    print(f"Output           : {out_data}\n")

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
    print("[1/3] Scanning embryos...")
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

        # Deduplicate while preserving temporal order
        for stg in sfmap:
            seen = set()
            deduped = []
            for p in sfmap[stg]:
                if p not in seen:
                    seen.add(p)
                    deduped.append(p)
            sfmap[stg] = deduped

        if sfmap:
            embryo_info[eid] = {'stage_frames': sfmap, 'all_frames': all_frames}

    print(f"  Valid embryos: {len(embryo_info)}")
    splits_count = {'train': 0, 'val': 0, 'test': 0}
    for eid in embryo_info:
        sp = split_of.get(eid, '')
        if sp in splits_count:
            splits_count[sp] += 1
    print(f"  Train: {splits_count['train']} | Val: {splits_count['val']} | Test: {splits_count['test']}\n")

    # ── 3. Build & write per-embryo sequences ─────────────────────────────────
    print("[2/3] Building sequences...")

    out_images_dir = out_data / "embryo_dataset"
    out_ann_dir    = out_data / "embryo_dataset_annotations"
    if not dry_run:
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_ann_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = [["embryo_id", "split"] + active_stages + ["total_frames", "quota"]]
    detailed_log_rows = [["embryo_id", "split", "stage", "original_frames", "selected_frames", "start_run", "end_run"]]
    written_embryos = {'train': 0, 'val': 0, 'test': 0}
    copy_tasks = []

    for eid in tqdm(embryo_info, desc="Write"):
        sp = split_of.get(eid, '')
        if sp not in ('train', 'val', 'test'):
            continue

        info = embryo_info[eid]
        stage_frames = info['stage_frames']

        # ── Chọn frame theo split ─────────────────────────────────────────
        if sp == 'train':
            selected = select_frames_stage_balanced(
                stage_frames, active_stages, transition_keep=transition_keep)
        else:
            # Val/Test: giữ nguyên toàn bộ
            selected = select_frames_full(stage_frames, active_stages)

        if not selected:
            continue

        # ── Tính quota để log ─────────────────────────────────────────────
        lengths = [len(entries) for entries in selected.values()]
        quota = int(sum(lengths) / len(lengths)) if lengths else 0

        # ── Xây dựng sequence theo thứ tự thời gian ───────────────────────
        # selected[stg] = List[Tuple[Path, aug_id]]
        # Sort theo RUN index của frame gốc
        all_selected: List[Tuple[Path, int]] = []
        for stg in active_stages:
            if stg in selected:
                all_selected.extend(selected[stg])
        all_selected.sort(key=lambda x: _run_sort_key(x[0]))

        if not all_selected:
            continue

        # ── Ghi ảnh (gom task) ───────────────────────────────────────────
        embryo_out_dir = out_images_dir / eid
        if not dry_run:
            embryo_out_dir.mkdir(parents=True, exist_ok=True)
            for fp, aug_id in all_selected:
                if aug_id == -1:
                    dst = embryo_out_dir / fp.name
                    if not dst.exists():
                        copy_tasks.append((str(fp), str(dst), -1))
                else:
                    dst_name = f"{fp.stem}_aug{aug_id}{fp.suffix}"
                    dst = embryo_out_dir / dst_name
                    if not dst.exists():
                        copy_tasks.append((str(fp), str(dst), aug_id))

        # ── Thu thập detailed log ─────────────────────────────────────────
        for stg in active_stages:
            orig_f = info['stage_frames'].get(stg, [])
            sel_entries = selected.get(stg, [])
            if orig_f:
                s_run = _run_sort_key(orig_f[0])
                e_run = _run_sort_key(orig_f[-1])
                detailed_log_rows.append([eid, sp, stg, len(orig_f), len(sel_entries), s_run, e_run])

        # ── Ghi annotation CSV ────────────────────────────────────────────
        # Tạo mapping: (frame_path, aug_id) → stage
        frame_to_stage: Dict[Tuple[Path, int], str] = {}
        for stg, entries in selected.items():
            for fp, aug_id in entries:
                frame_to_stage[(fp, aug_id)] = stg

        ann_entries = []
        if all_selected:
            cur_stage = frame_to_stage.get(all_selected[0])
            cur_start_run = _run_sort_key(all_selected[0][0])
            cur_end_run   = cur_start_run

            for fp, aug_id in all_selected[1:]:
                stg = frame_to_stage.get((fp, aug_id))
                run = _run_sort_key(fp)
                if stg == cur_stage:
                    cur_end_run = run
                else:
                    if cur_stage is not None:
                        ann_entries.append((cur_stage, cur_start_run, cur_end_run))
                    cur_stage = stg
                    cur_start_run = run
                    cur_end_run   = run
            if cur_stage is not None:
                ann_entries.append((cur_stage, cur_start_run, cur_end_run))

        if not dry_run:
            ann_path = out_ann_dir / f"{eid}_phases.csv"
            with open(ann_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for stage_name, s_run, e_run in ann_entries:
                    writer.writerow([stage_name, s_run, e_run])

        # ── Stats row ─────────────────────────────────────────────────────
        stage_counts = {stg: len(selected.get(stg, [])) for stg in active_stages}
        total = sum(stage_counts.values())
        stats_rows.append(
            [eid, sp] + [stage_counts[s] for s in active_stages] + [total, quota]
        )
        written_embryos[sp] += 1
        logging.info(f"{eid} ({sp}): {total} frames, quota={quota}, stages={stage_counts}")

    # ── Copy/aug ảnh song song ───────────────────────────────────────────────
    if not dry_run and copy_tasks:
        print(f"\n[3/4] Writing {len(copy_tasks)} images (parallel)...")

        def _do_write(task):
            src, dst, aug_id = task
            if aug_id == -1:
                shutil.copy2(src, dst)
            else:
                try:
                    arr = np.array(Image.open(src).convert('RGB'))
                    arr = _augment_single(arr, aug_id)
                    Image.fromarray(arr).save(dst, 'JPEG', quality=95)
                except Exception as e:
                    logging.error(f"Aug write error {src} aug{aug_id}: {e}")

        workers = min(32, (os.cpu_count() or 4) * 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            list(tqdm(executor.map(_do_write, copy_tasks), total=len(copy_tasks), desc="Write images"))

    # ── 4. Write splits.json + stats ─────────────────────────────────────────
    print("\n[4/4] Writing metadata...")
    if not dry_run:
        # splits.json
        splits_out = {'train': [], 'val': [], 'test': []}
        for eid in embryo_info:
            sp = split_of.get(eid, '')
            if sp in splits_out:
                splits_out[sp].append(eid)
        with open(out_data / "splits.json", 'w', encoding='utf-8') as f:
            json.dump(splits_out, f, indent=2)

        # dataset_stats.csv
        stats_path = out_data / "dataset_stats.csv"
        with open(stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(stats_rows)
        print(f"  Stats: {stats_path}")

        # detailed_log.csv
        detailed_log_path = out_data / "detailed_log_timelapse.csv"
        with open(detailed_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(detailed_log_rows)
        print(f"  Detailed Log: {detailed_log_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)
    print(f"  Train embryos written: {written_embryos['train']}")
    print(f"  Val   embryos written: {written_embryos['val']}")
    print(f"  Test  embryos written: {written_embryos['test']}")

    # Per-stage stats từ stats_rows
    if len(stats_rows) > 1:
        print(f"\n  {'Split':<6} {'Stage':<12} {'Embryos':>8} {'Total Frames':>13} {'Avg/Embryo':>11}")
        print(f"  {'-'*55}")
        for sp in ['train', 'val', 'test']:
            sp_rows = [r for r in stats_rows[1:] if r[1] == sp]
            if not sp_rows:
                continue
            for si, stg in enumerate(active_stages):
                col_idx = 2 + si
                counts = [r[col_idx] for r in sp_rows if r[col_idx] > 0]
                if counts:
                    total = sum(counts)
                    avg = total / len(counts)
                    print(f"  {sp:<6} {stg:<12} {len(counts):>8} {total:>13,} {avg:>11.1f}")
    print("\nDONE.")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="IVF Timelapse Video Dataset Builder — per-embryo stage-balanced sequences")
    p.add_argument('--data_root',       required=True,  help='Raw embryo image folder')
    p.add_argument('--ann_root',        required=True,  help='Annotation CSV folder')
    p.add_argument('--out_data',        required=True,  help='Output dataset folder')
    p.add_argument('--embryo_list',     required=True,  help='JSON file with train/val/test embryo IDs')
    p.add_argument('--transition_keep', type=int, default=5,
                   help='Số frame đầu/cuối mỗi stage luôn giữ (transition frames, default: 5)')
    p.add_argument('--seed',            type=int, default=42)
    p.add_argument('--merge_classes',   action='store_true', default=True,
                   help='Merge into 7 classes (default: True)')
    p.add_argument('--dry_run',         action='store_true')
    args = p.parse_args()

    run_pipeline(
        data_root=args.data_root,
        ann_root=args.ann_root,
        embryo_list=args.embryo_list,
        out_data=args.out_data,
        dry_run=args.dry_run,
        merge_classes=args.merge_classes,
        seed=args.seed,
        transition_keep=args.transition_keep,
    )

if __name__ == '__main__':
    main()
