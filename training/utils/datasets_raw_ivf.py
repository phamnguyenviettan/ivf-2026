"""
datasets_raw_ivf.py — IVF Raw Dataset Loader (online preprocessing)
====================================================================
Loads raw embryo images directly from disk, applies CLAHE+Sobel preprocessing
online, and generates sequential sliding-window clips for training/validation.

Label = stage of the LAST frame in each clip.
No pre-built clip folders required — works directly from the raw dataset
and annotation CSVs.

7 classes:
  0: tpnf  — tPNf
  1: t2    — t2
  2: t3+   — t3, t4
  3: t5+   — t5, t6
  4: t7+   — t7, t8
  5: t9+   — t9+
  6: tM+   — tM, tSB
"""

import re
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All raw stage labels that appear in CSV (lowercase for matching)
# tPB2, tPNa are pre-fertilisation — excluded from training
# tB, tEB are blastocyst — excluded
RAW_STAGES_VALID = {
    'tpnf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tm', 'tsb'
}

# 7-class output
VALID_STAGES_MERGED = ['tpnf', 't2', 't3+', 't5+', 't7+', 't9+', 'tM+']

# Merge map: raw CSV stage (lowercase) → output class
MERGE_MAP = {
    'tpnf': 'tpnf',
    't2':   't2',
    't3':   't3+',
    't4':   't3+',
    't5':   't5+',
    't6':   't5+',
    't7':   't7+',
    't8':   't7+',
    't9+':  't9+',
    'tm':   'tM+',
    'tsb':  'tM+',
}

IMG_EXTS = {'.jpeg', '.jpg', '.png'}

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

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _run_sort_key(p: Path) -> int:
    m = re.search(r'RUN(\d+)', p.stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+', p.stem)
    return int(nums[-1]) if nums else 0


def apply_clahe_sobel(img: Image.Image) -> Image.Image:
    """Áp dụng CLAHE + Sobel Edge để làm rõ ranh giới màng tế bào phôi."""
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


def _parse_phases_csv(csv_path: Path) -> List[Tuple[str, int, int]]:
    """
    Read {embryo_id}_phases.csv — format: stage,start_frame,end_frame (1-indexed).
    Returns list of (raw_stage_lower, start, end) for stages in RAW_STAGES_VALID.
    """
    if not csv_path.exists():
        return []
    phases = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                stage = row[0].strip().lower()
                if stage not in RAW_STAGES_VALID:
                    continue
                try:
                    start_frame = int(row[1].strip())
                    end_frame = int(row[2].strip())
                except ValueError:
                    continue
                phases.append((stage, start_frame, end_frame))
    except Exception:
        return []
    return phases


def _get_embryo_frames(data_root: Path, embryo_id: str) -> List[Path]:
    """List all image files in data_root/embryo_id/, sorted by RUN number."""
    embryo_dir = data_root / embryo_id
    if not embryo_dir.exists():
        return []
    frames = [p for p in embryo_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    frames.sort(key=_run_sort_key)
    return frames


def _assign_labels_to_frames(
    frames: List[Path],
    phases: List[Tuple[str, int, int]],
) -> List[Optional[str]]:
    """
    For each frame (0-based index i), assign merged class label based on
    which phase interval contains frame_num = i+1 (1-indexed).
    Returns None for frames with no annotation or excluded stages.
    """
    labels: List[Optional[str]] = []
    for i in range(len(frames)):
        frame_num = i + 1
        assigned = None
        for stage, start_frame, end_frame in phases:
            if start_frame <= frame_num <= end_frame:
                merged = MERGE_MAP.get(stage)
                if merged is not None:
                    assigned = merged
                break
        labels.append(assigned)
    return labels


def _build_sequential_clips(
    frames: List[Path],
    labels: List[Optional[str]],
    num_frames: int,
    stage_to_idx: Dict[str, int],
) -> List[Tuple[List[Path], int]]:
    """
    Sliding window stride=1 over ALL frames of an embryo.
    Label = merged class of the LAST frame in the clip.
    Only include clips where the last frame has a valid label.
    """
    n = len(frames)
    clips = []
    for end in range(num_frames - 1, n):
        last_label = labels[end]
        if last_label is None:
            continue
        class_idx = stage_to_idx.get(last_label)
        if class_idx is None:
            continue
        clip = frames[end - num_frames + 1: end + 1]
        clips.append((clip, class_idx))
    return clips


# ---------------------------------------------------------------------------
# IVFRawDataset
# ---------------------------------------------------------------------------

class IVFRawDataset(data.Dataset):
    """
    Raw IVF embryo dataset — sequential sliding-window clips, stride=1.

    Label = stage of the last frame in each clip (7-class merged).
    Applies CLAHE+Sobel preprocessing online.
    """

    def __init__(
        self,
        data_root,
        ann_root,
        splits_json,
        split,
        num_frames: int = 5,
        img_size: int = 224,
        is_training: bool = False,
        **kwargs,
    ):
        self.data_root = Path(data_root)
        self.ann_root = Path(ann_root)
        self.splits_json = Path(splits_json)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_training = is_training

        self.STAGE_TO_IDX: Dict[str, int] = {
            s: i for i, s in enumerate(VALID_STAGES_MERGED)
        }

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.samples: List[Tuple[List[Path], int]] = []
        self._build_samples()

    def _build_samples(self):
        with open(self.splits_json, 'r', encoding='utf-8') as f:
            splits_data = json.load(f)
        embryo_ids: List[str] = splits_data.get(self.split, [])
        embryo_ids = [e for e in embryo_ids if e not in BLACKLIST]

        skipped_no_frames = 0
        skipped_no_phases = 0
        total_clips = 0

        for embryo_id in embryo_ids:
            frames = _get_embryo_frames(self.data_root, embryo_id)
            if not frames:
                skipped_no_frames += 1
                continue

            csv_path = self.ann_root / f"{embryo_id}_phases.csv"
            phases = _parse_phases_csv(csv_path)
            if not phases:
                skipped_no_phases += 1
                continue

            labels = _assign_labels_to_frames(frames, phases)
            clips = _build_sequential_clips(
                frames, labels, self.num_frames, self.STAGE_TO_IDX
            )
            self.samples.extend(clips)
            total_clips += len(clips)

        if skipped_no_frames > 0:
            print(f"[IVFRawDataset] Skipped {skipped_no_frames} embryos (no frames)")
        if skipped_no_phases > 0:
            print(f"[IVFRawDataset] Skipped {skipped_no_phases} embryos (no phases)")

        print(
            f"[IVFRawDataset] split={self.split}, embryos={len(embryo_ids)}, "
            f"clips={total_clips}, training={self.is_training}"
        )

        from collections import Counter
        dist = Counter(label for _, label in self.samples)
        for idx, stage in enumerate(VALID_STAGES_MERGED):
            count = dist.get(idx, 0)
            pct = 100.0 * count / max(len(self.samples), 1)
            print(f"  {stage:<6}: {count:>6} ({pct:>5.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        frame_paths, label = self.samples[index]

        imgs = [Image.open(fp).convert('RGB') for fp in frame_paths]
        imgs = [apply_clahe_sobel(img) for img in imgs]

        if self.is_training:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                imgs[0], scale=(0.7, 1.0), ratio=(0.9, 1.1)
            )
            do_hflip = random.random() < 0.5
            do_vflip = random.random() < 0.2
            angle = random.uniform(-15, 15)

            frames = []
            for img in imgs:
                img = TF.resized_crop(img, i, j, h, w, [self.img_size, self.img_size])
                if do_hflip:
                    img = TF.hflip(img)
                if do_vflip:
                    img = TF.vflip(img)
                img = TF.rotate(img, angle)

                if random.random() < 0.8:
                    img = transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02
                    )(img)

                t = TF.to_tensor(img)
                t = self.normalize(t)

                if random.random() < 0.15:
                    t = transforms.RandomErasing(p=1.0, scale=(0.02, 0.1))(t)

                frames.append(t)
        else:
            frames = []
            for img in imgs:
                img = TF.resize(img, [self.img_size, self.img_size])
                t = TF.to_tensor(img)
                t = self.normalize(t)
                frames.append(t)

        clip = torch.stack(frames, dim=0)  # (T, 3, H, W)
        return clip, label


# ---------------------------------------------------------------------------
# get_raw_ivf_loader
# ---------------------------------------------------------------------------

def get_raw_ivf_loader(args, is_training: bool, split: str = None):
    """
    Create a DataLoader for the raw IVF dataset.
    Returns (loader, class_distribution_dict).
    """
    if split is None:
        split = 'train' if is_training else 'val'

    data_root = Path(args.data_dir)
    ann_root = Path(args.ann_dir)

    splits_json = getattr(args, 'splits_json', None)
    if splits_json:
        splits_json = Path(splits_json)
    if not splits_json or not splits_json.exists():
        fallback = data_root.parent / 'processdata' / 'splits.json'
        if fallback.exists():
            splits_json = fallback
        else:
            raise FileNotFoundError(
                f"splits_json not found. Tried: {getattr(args, 'splits_json', None)} "
                f"and fallback {fallback}"
            )

    num_frames = getattr(args, 'num_frames', 5)
    img_size = getattr(args, 'img_size', 224) or 224

    dataset = IVFRawDataset(
        data_root=data_root,
        ann_root=ann_root,
        splits_json=splits_json,
        split=split,
        num_frames=num_frames,
        img_size=img_size,
        is_training=is_training,
    )

    batch_size = args.batch_size if is_training else (
        getattr(args, 'validation_batch_size', None) or args.batch_size
    )

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=getattr(args, 'workers', 4),
        pin_memory=getattr(args, 'pin_mem', False),
        drop_last=is_training,
    )

    from collections import Counter
    dist = Counter(label for _, label in dataset.samples)
    class_distribution = {
        stage: {'total': dist.get(idx, 0)}
        for idx, stage in enumerate(VALID_STAGES_MERGED)
    }

    return loader, class_distribution
