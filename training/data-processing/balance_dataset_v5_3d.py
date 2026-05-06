"""
balance_dataset_v5_3d.py — IVF Dataset Builder (5-frame clips, class-balanced)
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
RAW_STAGES = ['tpnf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tm', 'tsb']

# Final stage names with numbering (Standard)
VALID_STAGES = ['1-tpnf', '2-t2', '3-t3', '4-t4', '5-t5', '6-t6', '7-t7', '8-t8', '9-t9+', '10-tm', '11-tsb']

# Final stage names with numbering (Merged/Q1)
VALID_STAGES_MERGED = ['1-tpnf', '2-t2', '3-t3+', '5-t5+', '7-t7+', '9-t9+', '10-tm+']

# Mappings for data processing
MAP_NORMAL = {
    'tpnf': '1-tpnf', 't2': '2-t2', 't3': '3-t3', 't4': '4-t4', 't5': '5-t5',
    't6': '6-t6', 't7': '7-t7', 't8': '8-t8', 't9+': '9-t9+', 'tm': '10-tm', 'tsb': '11-tsb'
}

MAP_MERGED = {
    'tpnf': '1-tpnf', 't2': '2-t2',
    't3': '3-t3+', 't4': '3-t3+',
    't5': '5-t5+', 't6': '5-t5+',
    't7': '7-t7+', 't8': '7-t7+',
    't9+': '9-t9+',
    'tm': '10-tm+', 'tsb': '10-tm+'
}

IMG_EXTS = {'.jpeg', '.jpg', '.png'}

CELL_COUNTS = {
    'tpnf': 1, 't2': 2, 't3': 3, 't4': 4, 't5': 5, 't6': 6,
    't7': 7, 't8': 8, 't9+': 9, 'tm': 10, 'tsb': 11,
    '1-tpnf': 1, '2-t2': 2, '3-t3': 3, '4-t4': 4, '5-t5': 5, '6-t6': 6,
    '7-t7': 7, '8-t8': 8, '9-t9+': 9, '10-tm': 10, '11-tsb': 11,
    '3-t3+': 3, '5-t5+': 5, '7-t7+': 7, '10-tm+': 10,
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
# Parallel clip writing helper
# ===========================================================================
def _write_single_clip(args):
    """Worker function for parallel clip writing."""
    (clip_frames, out_folder, frame_to_cc, stage, embryo_id, clip_name, 
     split, stage_fallback, aug_id, stride_val) = args
    
    try:
        out_folder.mkdir(parents=True, exist_ok=True)
        
        # Load and augment if needed
        if aug_id >= 0:
            aug_imgs = augment_clip_5frames(clip_frames, aug_id)
            if not aug_imgs:
                return None
            imgs = [Image.fromarray(img) for img in aug_imgs]
        else:
            imgs = [Image.open(fp).convert('RGB') for fp in clip_frames]
        
        # Apply CLAHE+Sobel and save
        for frame_idx, img in enumerate(imgs):
            img = apply_clahe_sobel(img)
            img.save(out_folder / f"frame_{frame_idx}.jpeg", 'JPEG', quality=95)
        
        # Cell count
        counts = [frame_to_cc.get(fp, stage_fallback) for fp in clip_frames]
        avg_count = sum(counts) / len(counts)
        
        return {
            'clip_name': clip_name,
            'split': split,
            'stage': stage,
            'cell_count': round(avg_count, 1),
            'embryo_id': embryo_id,
            'is_aug': True if aug_id >= 0 else False,
            'stride': stride_val
        }
    except Exception as e:
        logging.error(f"Error writing {clip_name}: {e}")
        shutil.rmtree(out_folder, ignore_errors=True)
        return None


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
# Sparse clip generation
# ===========================================================================
def generate_sparse_clips(frames: List[Path], num_frames: int, start_stride: int) -> List[Tuple[List[Path], int]]:
    """
    Sinh TẤT CẢ sparse clips từ stride=1 tăng dần đến stride=start_stride.
    Ưu tiên stride nhỏ (dense) trước — clips liên tiếp được chọn trước,
    sau đó mới đến sparse stride lớn hơn.

    VD: num_frames=5, start_stride=10, n=100:
      stride=1:  [0,1,2,3,4], [1,2,3,4,5], ..., [95,96,97,98,99]  → 96 clips
      stride=2:  [0,2,4,6,8], [1,3,5,7,9], ..., [91,93,95,97,99]  → 92 clips
      ...
      stride=10: [0,10,20,30,40], ..., [59,69,79,89,99]            → 60 clips

    Thứ tự: stride ASC → dense clips (stride=1) được ưu tiên trước trong round-robin.
    Aug chỉ được dùng khi toàn bộ pool này không đủ target.

    KHÔNG có clip trùng: mỗi (i, s) cho tập frame khác nhau.
    """
    n = len(frames)
    all_clips = []
    for s in range(1, start_stride + 1):
        clip_span = (num_frames - 1) * s
        if clip_span >= n:
            continue  # stride quá lớn so với n_frames → bỏ qua
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
# Main pipeline
# ===========================================================================
def run_pipeline(
    data_root: str,
    ann_root: str,
    embryo_list: Optional[str],
    out_data: str,
    target_train: int,
    target_val: int,
    target_test: int,
    stride: int,
    dry_run: bool,
    merge_classes: bool = False,
    num_frames: int = 5,
    auto_stride: bool = False,
    dense_ratio: float = 0.0,   # deprecated — giữ để backward compat, không dùng
):
    data_root = Path(data_root)
    ann_root  = Path(ann_root)
    out_data  = Path(out_data)

    if not dry_run:
        out_data.mkdir(parents=True, exist_ok=True)

    log_file = out_data / "balance_dataset.log" if not dry_run else Path("dry_run_balance.log")
    logging.basicConfig(
        filename=log_file, filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    active_stages = VALID_STAGES_MERGED if merge_classes else VALID_STAGES

    print(f"=====================================================================")
    print(f" IVF Dataset Builder v5.3d (5-frame clips, Cell Count Annotated)      ")
    print(f"=====================================================================")
    if merge_classes:
        print("[MERGE] Gộp lớp: t3+t4 → 't3+' | t5+t6 → 't5+' | t7+t8 → 't7+' | tM+tSB → 'tm+'")
    stride_mode = "AUTO (per-stage)" if auto_stride else str(stride)
    print(f"Active stages  : {active_stages}")
    print(f"Num frames/clip: {num_frames}")
    print(f"Stride mode    : {stride_mode} (stride 1→{stride}, ưu tiên dense trước)")
    print(f"Targets        : Train={target_train} clips/class")
    print(f"                 Val={'unlimited (stride=1, lấy hết)'}")
    print(f"                 Test={'unlimited (stride=1, lấy hết)'}")
    print(f"Clip type      : Dense-first (stride 1→{stride}), aug chỉ khi pool không đủ\n")

    # 1. Read embryo list & splits
    if not embryo_list:
        raise ValueError("--embryo_list là bắt buộc!")

    split_of: Dict[str, str] = {}
    with open(embryo_list) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and set(raw.keys()) <= {'train', 'val', 'test'}:
        for sp, ids in raw.items():
            for eid in ids:
                split_of[eid] = sp

    all_ids = [e for e in split_of if e not in BLACKLIST]
    split_of = {e: sp for e, sp in split_of.items() if e not in BLACKLIST}

    # 2. Scan embryos
    print("[1/3] Scanning embryos & frame counts...")
    embryo_info: Dict = {}
    skipped_short = 0

    for eid in tqdm(all_ids, desc="Scan"):
        all_frames = get_all_frames(data_root, eid)
        stage_ranges = parse_csv(ann_root, eid)
        if not all_frames or not stage_ranges:
            continue

        frame_to_cc: Dict[Path, int] = {}
        for stg, (s0, e0) in stage_ranges.items():
            s0 = max(0, min(s0, len(all_frames) - 1))
            e0 = max(0, min(e0, len(all_frames) - 1))
            cc_val = CELL_COUNTS[stg]
            for f in all_frames[s0: e0 + 1]:
                frame_to_cc[f] = cc_val

        sfmap: Dict[str, List[Path]] = {}
        for stg, (s0, e0) in stage_ranges.items():
            s0 = max(0, min(s0, len(all_frames) - 1))
            e0 = max(0, min(e0, len(all_frames) - 1))
            sf = all_frames[s0: e0 + 1]
            if not sf:
                continue

            tgt_stg = (MAP_MERGED if merge_classes else MAP_NORMAL).get(stg, stg)
            if tgt_stg not in sfmap:
                sfmap[tgt_stg] = []
            sfmap[tgt_stg].extend(sf)

        for stg in sfmap:
            sfmap[stg] = sorted(list(set(sfmap[stg])), key=lambda x: all_frames.index(x))

        if sfmap:
            embryo_info[eid] = {
                'stage_frames': sfmap,
                'frame_to_cc': frame_to_cc,
            }

    print(f"  Scanned {len(embryo_info)} embryos with valid stages\n")

    # Compute per-stage strides
    if auto_stride:
        stage_stride_map = compute_per_stage_auto_stride(
            ann_root=ann_root,
            active_stages=active_stages,
            num_frames=num_frames,
            merge_classes=merge_classes,
            min_clips_per_embryo=5,
        )
    else:
        stage_stride_map = {s: stride for s in active_stages}

    # 3. Build & write clips
    splits = ['train', 'val', 'test']
    # Val/Test: 0 = unlimited (lấy hết stride=1)
    # Dùng sys.maxsize để round-robin không bao giờ dừng sớm
    import sys as _sys
    targets_by_split = {
        'train': target_train,
        'val':   target_val if target_val > 0 else _sys.maxsize,
        'test':  target_test if target_test > 0 else _sys.maxsize,
    }
    global_annotations = []
    
    # Q1 Tracking variables
    q1_pool_stats = {sp: {stg: {'embryos': 0, 'pool': 0} for stg in active_stages} for sp in splits}
    clip_counter = defaultdict(int)
    rng = random.Random(42)

    print("[2/3] Processing clips...")

    for sp in splits:
        target_clips = targets_by_split[sp]
        # Val/Test: lấy hết tất cả clips stride=1, không giới hạn
        unlimited = (sp in ('val', 'test'))
        print(f"\n  [{sp.upper()}] Target = {'unlimited (stride=1)' if unlimited else target_clips} clips per class")

        for stage in active_stages:
            cands = [
                eid for eid in embryo_info
                if split_of.get(eid) == sp and stage in embryo_info[eid]['stage_frames']
            ]

            if not cands:
                print(f"    {stage:>6}: 0 embryos -> skipping")
                continue

            rng.shuffle(cands)

            # Lấy stride cho stage này
            effective_stride = stage_stride_map.get(stage, stride)

            # Val/Test: dùng stride=1 (dense) để phản ánh đúng inference
            if sp in ('val', 'test'):
                effective_stride = 1
            # Build sparse clip pool: [i, i+s, i+2s, ..., i+(n-1)*s] for all valid i
            # Tổng clips per embryo = n_frames - (num_frames-1)*stride  (nếu > 0)
            available_clips_by_eid: Dict[str, List[Tuple[List[Path], int]]] = {}
            total_available = 0
            for eid in cands:
                frames = embryo_info[eid]['stage_frames'][stage]
                e_clips = generate_sparse_clips(frames, num_frames, effective_stride)

                if not e_clips:
                    skipped_short += 1
                    continue

                rng.shuffle(e_clips)
                available_clips_by_eid[eid] = e_clips
                total_available += len(e_clips)

            active_cands = [eid for eid in cands if eid in available_clips_by_eid]

            if not active_cands:
                print(f"    {stage:>6}: 0 embryos có >= {num_frames} frames -> skipping")
                continue
                
            q1_pool_stats[sp][stage]['embryos'] = len(active_cands)
            q1_pool_stats[sp][stage]['pool'] = total_available

            # Log: stride info + pool summary
            clips_per_embryo_avg = total_available / len(active_cands) if active_cands else 0
            print(f"    {stage:>6}: stride 1→{effective_stride}  "
                  f"| embryos={len(active_cands):>3}  "
                  f"| pool={total_available:>6} clips  "
                  f"(~{clips_per_embryo_avg:.0f}/embryo)  "
                  f"| target={'unlimited' if unlimited else target_clips}")

            # Round-robin selection
            # Val/Test: lấy hết toàn bộ pool (unlimited)
            # Train: lấy đến target_clips
            selected_clips_by_eid: Dict[str, List[Tuple[List[Path], int]]] = defaultdict(list)
            total_collected = 0

            if unlimited:
                # Lấy hết tất cả clips có sẵn
                for eid in active_cands:
                    for clip_info in available_clips_by_eid[eid]:
                        selected_clips_by_eid[eid].append(clip_info)
                        total_collected += 1
            else:
                while total_collected < target_clips:
                    any_added = False
                    for eid in active_cands:
                        if total_collected >= target_clips:
                            break
                        if available_clips_by_eid[eid]:
                            clip_info = available_clips_by_eid[eid].pop(0)
                            selected_clips_by_eid[eid].append(clip_info)
                            total_collected += 1
                            any_added = True
                    if not any_added:
                        break

            # Write clips (parallel processing) - SKIP existing
            final_total_clips = 0
            stage_fallback = CELL_COUNTS.get(stage, 1)
            out_stage_dir = out_data / sp / stage
            
            # Load existing clips for this stage
            existing_clips_stage = set()
            csv_path = out_data / "annotations.csv"
            if csv_path.exists():
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['split'] == sp and row['stage'] == stage:
                            existing_clips_stage.add(row['clip_name'])
            
            # Prepare tasks for parallel processing
            tasks = []
            for eid, clip_list in selected_clips_by_eid.items():
                frame_to_cc = embryo_info[eid]['frame_to_cc']
                for clip_frames, stride_val in clip_list:
                    if len(clip_frames) != num_frames:
                        continue
                    clip_name = f"{eid}-{clip_counter[eid]}"
                    
                    # SKIP if already exists
                    if clip_name in existing_clips_stage:
                        clip_counter[eid] += 1
                        final_total_clips += 1  # Count existing
                        # MOCK return for stats aggregation
                        global_annotations.append({
                            'clip_name': clip_name, 'split': sp, 'stage': stage, 
                            'cell_count': 0, 'embryo_id': eid, 'is_aug': False, 'stride': stride_val
                        })
                        continue
                    
                    out_folder = out_stage_dir / clip_name
                    tasks.append((
                        clip_frames, out_folder, frame_to_cc, stage, eid,
                        clip_name, sp, stage_fallback, -1, stride_val  # -1 = no augmentation
                    ))
                    clip_counter[eid] += 1
            
            # Process in parallel
            if not dry_run and tasks:
                with Pool(min(cpu_count(), 8)) as pool:
                    results = list(tqdm(
                        pool.imap(_write_single_clip, tasks),
                        total=len(tasks),
                        desc=f"    {sp.upper()} - {stage:>4}",
                        leave=False
                    ))
                    for result in results:
                        if result:
                            global_annotations.append(result)
                            final_total_clips += 1
            else:
                final_total_clips += len(tasks)
                for t_args in tasks:
                    global_annotations.append({
                        'clip_name': t_args[5], 'split': t_args[6], 'stage': t_args[3],
                        'cell_count': 0, 'embryo_id': t_args[4], 'is_aug': False, 'stride': t_args[9]
                    })

            n_embryos = len(selected_clips_by_eid)
            shortfall = 0 if unlimited else (target_clips - final_total_clips)

            # If SHORT, apply augmentation (also parallel) — VÉT TỐI ĐA
            if shortfall > 0:
                print(f"    {stage:>6}: ✗ {final_total_clips:>5}/{target_clips} clips  "
                      f"| stride 1→{effective_stride} exhausted  "
                      f"| embryos={n_embryos}  "
                      f"| SHORT {shortfall} → augmenting...")
                
                aug_tasks = []
                aug_collected = 0
                
                # VÉT: Dùng TẤT CẢ clips gốc, mỗi clip tạo 16 variants
                # Round-robin qua tất cả clips cho đến khi đủ target
                all_clips_pool = []
                for eid, original_clips in selected_clips_by_eid.items():
                    for clip_frames, stride_val in original_clips:
                        all_clips_pool.append((eid, clip_frames, stride_val))
                
                # Shuffle để đa dạng embryo
                rng.shuffle(all_clips_pool)
                
                # Tạo aug cho từng clip, mỗi clip 16 variants
                for eid, clip_frames, stride_val in all_clips_pool:
                    if aug_collected >= shortfall:
                        break
                    
                    frame_to_cc = embryo_info[eid]['frame_to_cc']
                    
                    for aug_id in range(16):
                        if aug_collected >= shortfall:
                            break
                        clip_name = f"{eid}-{clip_counter[eid]}_aug{aug_id}"
                        
                        # SKIP if already exists
                        if clip_name in existing_clips_stage:
                            clip_counter[eid] += 1
                            aug_collected += 1
                            final_total_clips += 1
                            global_annotations.append({
                                'clip_name': clip_name, 'split': sp, 'stage': stage, 
                                'cell_count': 0, 'embryo_id': eid, 'is_aug': True, 'stride': stride_val
                            })
                            continue
                        
                        out_folder = out_stage_dir / clip_name
                        aug_tasks.append((
                            clip_frames, out_folder, frame_to_cc, stage, eid,
                            clip_name, sp, stage_fallback, aug_id, stride_val
                        ))
                        clip_counter[eid] += 1
                        aug_collected += 1
                
                # Process augmentation in parallel
                if not dry_run and aug_tasks:
                    with Pool(min(cpu_count(), 8)) as pool:
                        results = list(tqdm(
                            pool.imap(_write_single_clip, aug_tasks),
                            total=len(aug_tasks),
                            desc=f"    AUG - {stage:>4}",
                            leave=False
                        ))
                        for result in results:
                            if result:
                                global_annotations.append(result)
                                final_total_clips += 1
                elif dry_run and aug_tasks:
                    final_total_clips += len(aug_tasks)
                    for t_args in aug_tasks:
                        global_annotations.append({
                            'clip_name': t_args[5], 'split': t_args[6], 'stage': t_args[3],
                            'cell_count': 0, 'embryo_id': t_args[4], 'is_aug': True, 'stride': t_args[9]
                        })
                
                shortfall = target_clips - final_total_clips
                if shortfall <= 0:
                    print(f"    {stage:>6}: ✓ {final_total_clips:>5}/{target_clips} clips  "
                          f"| stride 1→{effective_stride} + aug filled  "
                          f"| embryos={n_embryos}")
                else:
                    print(f"    {stage:>6}: ⚠ {final_total_clips:>5}/{target_clips} clips  "
                          f"| stride 1→{effective_stride} + aug  "
                          f"| embryos={n_embryos}  "
                          f"| STILL SHORT {shortfall}")
            else:
                label = f"unlimited ({final_total_clips})" if unlimited else f"{final_total_clips}/{target_clips}"
                print(f"    {stage:>6}: ✓ {label} clips  "
                      f"| stride 1→{effective_stride}  "
                      f"| embryos={n_embryos}  "
                      f"| pool={total_available} avail")
            
            logging.info(
                f"Split {sp} Stage {stage}: {final_total_clips} clips "
                f"from {n_embryos} embryos (available={total_available})"
            )

    if skipped_short > 0:
        print(f"\n  ⚠ Skipped {skipped_short} embryo-stage pairs with < {num_frames} frames")

    # 4. Write CSV (append mode - skip existing clips)
    print("\n[3/3] Exporting annotations...")
    if not dry_run:
        csv_path = out_data / "annotations.csv"
        
        # Load existing annotations
        existing_clips = set()
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_clips.add(row['clip_name'])
            print(f"  -> Found {len(existing_clips)} existing clips in {csv_path}")
        
        # Filter out duplicates
        new_annotations = [r for r in global_annotations if r['clip_name'] not in existing_clips]
        
        if new_annotations:
            # Append new clips
            mode = 'a' if csv_path.exists() else 'w'
            with open(csv_path, mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['clip_name', 'split', 'stage', 'cell_count', 'embryo_id'],
                    extrasaction='ignore'
                )
                if mode == 'w':
                    writer.writeheader()
                for row in new_annotations:
                    writer.writerow(row)
            print(f"  -> Added {len(new_annotations)} new clips to {csv_path}")
        else:
            print(f"  -> No new clips to add (all {len(global_annotations)} already exist)")
        
        total_clips = len(existing_clips) + len(new_annotations)
        print(f"  -> Total clips in dataset: {total_clips}")

        # Verify: count unique clip_names
        names = [r['clip_name'] for r in global_annotations]
        if len(names) != len(set(names)):
            dupes = len(names) - len(set(names))
            print(f"  ⚠ WARNING: {dupes} duplicate clip_names detected!")
        else:
            print(f"  ✓ All {len(names)} clip_names are unique")

    # TỰ ĐỘNG XUẤT BÁO CÁO Q1 DYNAMICALLY
    export_q1_statistics(global_annotations, q1_pool_stats, splits, active_stages, targets_by_split, out_data)
    print("\nDONE.")

def export_q1_statistics(global_annotations, q1_pool_stats, splits, active_stages, targets_by_split, out_data):
    """
    Tự động tổng hợp và sinh ra bảng báo cáo thống kê chi tiết chuẩn Q1 Paper,
    dựa trên Tracking của từng stride và aug.
    """
    print("\n[4/4] Generating Q1 Statistical Report...")
    
    # Aggregation
    stats = {
        sp: {
            stg: {f"s{i}": 0 for i in range(1, 11)} | {'aug': 0, 'total': 0}
            for stg in active_stages
        } for sp in splits
    }
    
    for row in global_annotations:
        sp = row['split']
        stg = row['stage']
        is_aug = row.get('is_aug', False)
        stride = row.get('stride', 1)
        
        if is_aug:
            stats[sp][stg]['aug'] += 1
        else:
            s_key = stride if stride <= 10 else 10
            key = f"s{s_key}"
            if key in stats[sp][stg]:
                stats[sp][stg][key] += 1
        stats[sp][stg]['total'] += 1
        
    # Terminal Display (Markdown)
    print("\n" + "="*80)
    print(" >>> Q1 PAPER STATISTICAL REPORT (DYNAMICALLY GENERATED) <<<")
    print("="*80)
    
    csv_rows = []
    headers = [
        "Split", "Stage", "Target", "Embryos", "Available Pool",
        "S:10", "S:9", "S:8", "S:7", "S:6", "S:5", "S:4", "S:3", "S:2", "S:1", 
        "Augmented", "Total Clips", "Total Frames"
    ]
    csv_rows.append(headers)
    
    for sp in splits:
        print(f"\n### Split: {sp.upper()} (Target: {targets_by_split[sp]} per class)")
        # Shortened header for terminal
        print(f"{'Stage':<10} | {'Emb':<4} | {'S:10-6':<15} | {'S:5':<4} | {'S:4':<4} | {'S:3':<4} | {'S:2':<4} | {'S:1':<4} | {'Aug':<5} | {'Total':<6}")
        print("-" * 100)
        for stg in active_stages:
            d = stats[sp][stg]
            p = q1_pool_stats[sp][stg]
            total_frames = d['total'] * 5
            
            s10_6 = f"{d['s10']},{d['s9']},{d['s8']},{d['s7']},{d['s6']}"
            print(f"{stg:<10} | {p['embryos']:<4} | {s10_6:<15} | "
                  f"{d['s5']:<4} | {d['s4']:<4} | {d['s3']:<4} | {d['s2']:<4} | {d['s1']:<4} | "
                  f"{d['aug']:<5} | {d['total']:<6}")
                  
            csv_rows.append([
                sp.upper(), stg, targets_by_split[sp], p['embryos'], p['pool'],
                d['s10'], d['s9'], d['s8'], d['s7'], d['s6'],
                d['s5'], d['s4'], d['s3'], d['s2'], d['s1'],
                d['aug'], d['total'], total_frames
            ])
            
    # Write to CSV
    try:
        if not out_data.exists():
            out_data.mkdir(parents=True, exist_ok=True)
            
        csv_path = out_data / "dataset_q1_statistics.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\n[INFO] Detailed Q1 statistics successfully exported to: {csv_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to write Q1 statistics CSV: {e}")

def main():
    p = argparse.ArgumentParser(
        description="IVF Dataset Builder v5.3d — Sparse Sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CÁCH HOẠT ĐỘNG CỦA STRIDE:
  Stride = khoảng cách giữa các FRAME TRONG một clip (sparse sampling).
  VD: num_frames=5, stride=15:
    Clip 0: [frame_0,  frame_15, frame_30, frame_45, frame_60]
    Clip 1: [frame_1,  frame_16, frame_31, frame_46, frame_61]
    ...
  Thay đổi so với v5.2: stride KHÔNG còn là bước nhảy giữa 2 clip liên tiếp.

MODE AUTO STRIDE:
  --auto_stride: Tự tính stride tối ưu per-stage từ annotation CSVs,
  sao cho mỗi embryo có ít nhất 5 clips khả dụng trên median segment length.
  --stride bị bỏ qua khi --auto_stride được bật.
"""
    )
    p.add_argument('--data_root',    required=True,
                   help='Thư mục chứa ảnh phôi')
    p.add_argument('--ann_root',     required=True,
                   help='Thư mục chứa CSV annotation')
    p.add_argument('--out_data',     required=True,
                   help='Thư mục output dataset')
    p.add_argument('--embryo_list',  required=True,
                   help='File JSON train/val/test split')
    p.add_argument('--target_train', type=int, default=3500,
                   help='Số clips mục tiêu mỗi class (train). Default 3500.')
    p.add_argument('--target_val',   type=int, default=0,
                   help='Số clips mục tiêu mỗi class (val). 0 = lấy hết (unlimited, stride=1).')
    p.add_argument('--target_test',  type=int, default=0,
                   help='Số clips mục tiêu mỗi class (test). 0 = lấy hết (unlimited, stride=1).')
    p.add_argument('--dense_ratio',  type=float, default=0.0,
                   help='[deprecated] Không dùng nữa. Stride 1→max đã bao gồm dense clips.')
    p.add_argument('--stride',       type=int, default=10,
                   help='Stride tối đa trong clip (default: 10). '
                        'Clips được sinh từ stride=1 đến stride=max. '
                        'Bị bỏ qua nếu --auto_stride được bật.')
    p.add_argument('--auto_stride',  action='store_true',
                   help='Tự tính stride tối ưu per-stage từ annotation CSVs. '
                        'Stride lớn nhất để mỗi embryo có >= 5 clips (dựa trên median segment length).')
    p.add_argument('--num_frames',   type=int, default=5,
                   help='Số frame mỗi clip (default: 5)')
    p.add_argument('--merge_classes', action='store_true',
                   help='Gộp lớp: t3+t4→t3+, t5+t6→t5+, t7+t8→t7+ và tM+tSB→tm+ (7 classes)')
    p.add_argument('--dry_run',      action='store_true',
                   help='Chạy thử, không ghi file')
    args = p.parse_args()

    run_pipeline(
        data_root=args.data_root,
        ann_root=args.ann_root,
        embryo_list=args.embryo_list,
        out_data=args.out_data,
        target_train=args.target_train,
        target_val=args.target_val,
        target_test=args.target_test,
        stride=args.stride,
        dry_run=args.dry_run,
        merge_classes=args.merge_classes,
        num_frames=args.num_frames,
        auto_stride=args.auto_stride,
        dense_ratio=args.dense_ratio,
    )

if __name__ == '__main__':
    main()
