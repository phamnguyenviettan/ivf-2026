"""3D IVF Dataset loader — with temporally-consistent online augmentation.

All frames in a clip share the SAME spatial transform (crop, flip, rotation)
to preserve temporal consistency. Only per-frame color jitter is independent.

NOTE on preprocessing:
  - Dataset được tạo bởi balance_dataset_v5_3d_timelapevideo.py đã apply
    CLAHE+Sobel khi ghi file (_do_write). DataLoader KHÔNG cần apply lại.
  - frames_preprocessed flag được giữ lại để tương thích ngược nếu cần
    load raw frames (set False), nhưng mặc định True.
"""

import os
import math
import random
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

IVF_7_CLASSES = ['1-tpnf', '2-t2', '3-t3+', '5-t5+', '7-t7+', '9-t9+', '10-tm+']
IVF_9_CLASSES = ['tpnf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+']


class IVF3DDataset(data.Dataset):
    """
    3D IVF Dataset — loads T-frame clips with temporally-consistent augmentation.

    Structure: {root}/{stage}/{embryo_id}-{k}/frame_{0..T-1}.jpeg
    Returns: clip (T, 3, H, W) tensor, target (int)

    Frames đã được apply CLAHE+Sobel khi tạo dataset (balance_dataset_v5_3d).
    DataLoader chỉ cần resize + normalize, không cần CLAHE nữa.
    """

    NO_PREV = 7  # Special token index for "no previous stage"

    def __init__(self, root, class_map=None, img_size=224,
                 is_training=False, num_frames: int = 5,
                 phase2_mode: bool = False,
                 frames_preprocessed: bool = True,
                 **kwargs):
        self.root = root
        self.img_size = img_size
        self.is_training = is_training
        self.num_frames = num_frames
        self.phase2_mode = phase2_mode
        self.frames_preprocessed = frames_preprocessed  # kept for backward compat
        self.samples = []

        # Normalization — ImageNet stats, applied after augmentation
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if class_map is not None:
            skipped_empty = 0
            for stage_dir, class_idx in class_map.items():
                stage_path = os.path.join(root, stage_dir)
                if not os.path.isdir(stage_path):
                    continue

                for clip_folder in sorted(os.listdir(stage_path)):
                    clip_path = os.path.join(stage_path, clip_folder)
                    if not os.path.isdir(clip_path):
                        continue

                    available_frames = []
                    for i in range(num_frames):
                        found = False
                        for ext in ['.jpeg', '.jpg', '.png', '.bmp']:
                            fp = os.path.join(clip_path, f"frame_{i}{ext}")
                            if os.path.exists(fp):
                                available_frames.append(fp)
                                found = True
                                break
                        if not found:
                            break

                    if len(available_frames) == 0:
                        skipped_empty += 1
                        continue

                    # Pad nếu thiếu frame (repeat frame cuối)
                    while len(available_frames) < num_frames:
                        available_frames.append(available_frames[-1])

                    self.samples.append((available_frames, class_idx))

            if skipped_empty > 0:
                print(f"[IVF3DDataset] Skipped {skipped_empty} empty clips")

        if not self.samples:
            raise RuntimeError(f"No valid clips found in {root}")

        print(f"[IVF3DDataset] Loaded {len(self.samples)} clips from {root} "
              f"(T={num_frames}, training={is_training})")
        self.idx_to_stage = {v: k for k, v in class_map.items()} if class_map else {}

    def _get_prev_stage(self, target):
        """Teacher forcing: derive prev_stage from ground truth target."""
        if target == 0:
            return self.NO_PREV
        prev = target - 1
        if self.is_training and random.random() < 0.2:
            return self.NO_PREV
        return prev

    def __getitem__(self, index):
        available_frames, target = self.samples[index]

        # Load frames — CLAHE+Sobel đã được apply khi tạo dataset
        imgs = [Image.open(fp).convert('RGB') for fp in available_frames]

        if self.is_training:
            # ── Temporally-consistent augmentation ───────────────────────
            # TẤT CẢ T frames dùng CÙNG params cho mọi transform
            # (spatial + color + erasing) để giữ temporal coherence

            # 1. RandomResizedCrop params (shared)
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                imgs[0], scale=(0.7, 1.0), ratio=(0.9, 1.1))

            # 2. Flip decisions (shared)
            do_hflip = random.random() < 0.5
            do_vflip = random.random() < 0.2

            # 3. Rotation angle (shared)
            angle = random.uniform(-15, 15)

            # 4. ColorJitter params (shared) — quan trọng cho temporal consistency
            # Nếu frame 1 sáng hơn frame 2 do jitter khác nhau → nhiễu temporal signal
            do_jitter = random.random() < 0.8
            if do_jitter:
                jitter_fn = transforms.ColorJitter(
                    brightness=0.3, contrast=0.3,
                    saturation=0.1, hue=0.02)
                # Lấy params 1 lần, dùng cho tất cả frames
                jitter_params = transforms.ColorJitter.get_params(
                    jitter_fn.brightness, jitter_fn.contrast,
                    jitter_fn.saturation, jitter_fn.hue)

            # 5. RandomErasing params (shared) — cùng region cho tất cả frames
            do_erase = random.random() < 0.15
            erase_params = None
            if do_erase:
                # Tính erase region thủ công để tránh version dependency
                erase_ratio = random.uniform(0.02, 0.1)
                erase_aspect = random.uniform(0.3, 3.3)
                erase_h = int(round(math.sqrt(
                    self.img_size * self.img_size * erase_ratio * erase_aspect)))
                erase_w = int(round(math.sqrt(
                    self.img_size * self.img_size * erase_ratio / erase_aspect)))
                erase_h = min(erase_h, self.img_size)
                erase_w = min(erase_w, self.img_size)
                erase_i = random.randint(0, self.img_size - erase_h)
                erase_j = random.randint(0, self.img_size - erase_w)
                erase_v = torch.zeros(3, erase_h, erase_w)  # black erase
                erase_params = (erase_i, erase_j, erase_h, erase_w, erase_v)

            frames = []
            for img in imgs:
                # Shared spatial transforms
                img = TF.resized_crop(img, i, j, h, w,
                                      [self.img_size, self.img_size])
                if do_hflip:
                    img = TF.hflip(img)
                if do_vflip:
                    img = TF.vflip(img)
                img = TF.rotate(img, angle)

                # Shared color jitter — cùng params cho tất cả frames
                if do_jitter:
                    fn_idx, brightness_factor, contrast_factor, \
                        saturation_factor, hue_factor = jitter_params
                    for fn_id in fn_idx:
                        if fn_id == 0:
                            img = TF.adjust_brightness(img, brightness_factor)
                        elif fn_id == 1:
                            img = TF.adjust_contrast(img, contrast_factor)
                        elif fn_id == 2:
                            img = TF.adjust_saturation(img, saturation_factor)
                        elif fn_id == 3:
                            img = TF.adjust_hue(img, hue_factor)

                t = TF.to_tensor(img)
                t = self.normalize(t)

                # Shared random erasing — cùng region cho tất cả frames
                if do_erase and erase_params is not None:
                    i_e, j_e, h_e, w_e, v = erase_params
                    t = TF.erase(t, i_e, j_e, h_e, w_e, v)

                frames.append(t)
        else:
            # ── Val/Test: deterministic resize + normalize only ──────────
            frames = []
            for img in imgs:
                img = TF.resize(img, [self.img_size, self.img_size])
                t = TF.to_tensor(img)
                t = self.normalize(t)
                frames.append(t)

        clip = torch.stack(frames, dim=0)  # (T, 3, H, W)

        if self.phase2_mode:
            prev_stage = self._get_prev_stage(target)
            return clip, target, prev_stage

        return clip, target

    def __len__(self):
        return len(self.samples)

    def get_class_distribution(self) -> dict:
        counts = {}
        for frames, cls_idx in self.samples:
            if cls_idx not in counts:
                counts[cls_idx] = {'total': 0, 'real': 0}
            counts[cls_idx]['total'] += 1
            folder_name = os.path.basename(os.path.dirname(frames[0]))
            if '_aug' not in folder_name:
                counts[cls_idx]['real'] += 1
        return {
            self.idx_to_stage.get(idx, f'class_{idx}'): info
            for idx, info in sorted(counts.items())
        }


def get_3d_ivf_loader(args, is_training=True, split=None):
    """Create 3D IVF dataloader.

    Args:
        split: Override split name ('train', 'val', 'test').
               If None, inferred from is_training.
    """
    if split is None:
        split = 'train' if is_training else 'val'
    data_dir = os.path.join(args.data_dir, split)

    if not os.path.exists(data_dir):
        if split == 'val':
            alt_dir = os.path.join(args.data_dir, 'validation')
            if os.path.exists(alt_dir):
                data_dir = alt_dir
            else:
                raise FileNotFoundError(
                    f"Cannot find val/validation in {args.data_dir}")
        elif split == 'test':
            raise FileNotFoundError(
                f"Cannot find test split in {args.data_dir}")

    base_classes = IVF_7_CLASSES if args.num_classes == 7 else IVF_9_CLASSES

    class_map = {}
    if os.path.exists(data_dir):
        for d in sorted(os.listdir(data_dir)):
            full_path = os.path.join(data_dir, d)
            if not os.path.isdir(full_path):
                continue
            clean_d = d.strip()
            if args.num_classes == 7:
                if clean_d in ['t5', 't6', 't5+', 't6+', 't5_t6']:
                    class_map[d] = 4
                elif clean_d in ['t7', 't8', 't7+', 't8+', 't7_t8']:
                    class_map[d] = 5
                elif clean_d in base_classes:
                    class_map[d] = base_classes.index(clean_d)
            else:
                if clean_d in base_classes:
                    class_map[d] = base_classes.index(clean_d)

    if not class_map:
        print(f"[Warning] Empty class_map — using defaults")
        class_map = {c: i for i, c in enumerate(base_classes)}

    print(f"[get_3d_ivf_loader] class_map ({split}): {class_map}")

    img_size = getattr(args, 'img_size', 224) or 224
    num_frames = getattr(args, 'num_frames', 5)
    phase2 = getattr(args, 'phase2', False)

    dataset = IVF3DDataset(
        data_dir,
        class_map=class_map,
        img_size=img_size,
        is_training=is_training,
        num_frames=num_frames,
        phase2_mode=phase2,
        frames_preprocessed=True,  # dataset luôn đã có CLAHE+Sobel
    )

    if phase2:
        print(f"[get_3d_ivf_loader] Phase 2 mode: prev_stage enabled (teacher forcing)")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size if is_training else (
            getattr(args, 'validation_batch_size', None) or args.batch_size),
        shuffle=is_training,
        num_workers=args.workers,
        pin_memory=getattr(args, 'pin_mem', False),
        drop_last=is_training,
    )

    dist = dataset.get_class_distribution()
    if is_training:
        print(f"[get_3d_ivf_loader] Distribution ({split}, total={len(dataset)}):")
        for stage, info in dist.items():
            total = info['total']
            real = info['real']
            pct = 100.0 * total / len(dataset)
            print(f"  {stage:<8}: {total:>5} clips ({pct:>5.1f}%) [Real: {real:>5}]")

    return loader, dist
