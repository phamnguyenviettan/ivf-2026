"""
datasets_2d_ivf.py — 2D IVF Dataset loader for EmbryoMambaNet (Phase 1)

Dataset structure (built by balance_dataset_v5_2d.py):
    {root}/{split}/{stage}/{embryo_id}-{original_filename}.jpeg
    e.g. dataset_balanced_2d/train/1-tPNf/AA83-7-D2013.01.28_S0717_I132_WELL7_RUN42.jpeg

Returns: image (3, H, W) tensor, label (int)
"""

import os
import random
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# 7 merged classes — must match balance_dataset_v5_2d.py VALID_STAGES_MERGED
IVF_7_CLASSES = ['1-tPNf', '2-t2', '3-t3+', '5-t5+', '7-t7+', '9-t9+', '10-tM+']


class IVF2DDataset(data.Dataset):
    """
    2D IVF Dataset — loads single images with online augmentation.

    Structure: {root}/{stage}/{image_file}.jpeg
    Returns: image (3, H, W) tensor, target (int)
    """

    def __init__(self, root: str, class_map: dict = None,
                 img_size: int = 224, is_training: bool = False, **kwargs):
        self.root        = root
        self.img_size    = img_size
        self.is_training = is_training
        self.samples     = []   # list of (filepath, class_idx)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        if class_map is None:
            raise ValueError("class_map is required for IVF2DDataset")

        skipped = 0
        for stage_dir, class_idx in class_map.items():
            stage_path = os.path.join(root, stage_dir)
            if not os.path.isdir(stage_path):
                continue
            for fname in sorted(os.listdir(stage_path)):
                if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    continue
                fp = os.path.join(stage_path, fname)
                self.samples.append((fp, class_idx))

        if not self.samples:
            raise RuntimeError(f"No valid images found in {root} with class_map={class_map}")

        self.idx_to_stage = {v: k for k, v in class_map.items()}
        print(f"[IVF2DDataset] Loaded {len(self.samples):,} images from {root} "
              f"(training={is_training})")

    def __getitem__(self, index: int):
        fp, target = self.samples[index]
        img = Image.open(fp).convert('RGB')

        if self.is_training:
            # ── Online augmentation ───────────────────────────────────────────
            # RandomResizedCrop
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.7, 1.0), ratio=(0.9, 1.1))
            img = TF.resized_crop(img, i, j, h, w, [self.img_size, self.img_size])

            # Flip
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.2:
                img = TF.vflip(img)

            # Rotation
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle)

            # Color jitter
            if random.random() < 0.8:
                img = transforms.ColorJitter(
                    brightness=0.3, contrast=0.3,
                    saturation=0.1, hue=0.02)(img)

            # To tensor + normalize
            t = TF.to_tensor(img)
            t = self.normalize(t)

            # Random erasing
            if random.random() < 0.15:
                t = transforms.RandomErasing(p=1.0, scale=(0.02, 0.1))(t)

        else:
            # ── Validation/Test: deterministic resize ─────────────────────────
            img = TF.resize(img, [self.img_size, self.img_size])
            t = TF.to_tensor(img)
            t = self.normalize(t)

        return t, target

    def __len__(self) -> int:
        return len(self.samples)

    def get_class_distribution(self) -> dict:
        """Returns {stage_name: {'total': N, 'real': N_real}} for logging."""
        counts = {}
        for fp, cls_idx in self.samples:
            stage = self.idx_to_stage.get(cls_idx, f'class_{cls_idx}')
            if stage not in counts:
                counts[stage] = {'total': 0, 'real': 0}
            counts[stage]['total'] += 1
            fname = os.path.basename(fp)
            if '_aug' not in fname:
                counts[stage]['real'] += 1
        return counts


def get_2d_ivf_loader(args, is_training: bool = True, split: str = None):
    """
    Create 2D IVF DataLoader.

    Args:
        args: training args (needs data_dir, num_classes, img_size, batch_size, workers, pin_mem)
        is_training: True for train split, False for val
        split: override split name ('train', 'val', 'test')

    Returns:
        loader, class_distribution_dict
    """
    if split is None:
        split = 'train' if is_training else 'val'

    data_dir = os.path.join(args.data_dir, split)

    # Fallback: val → validation
    if not os.path.exists(data_dir) and split == 'val':
        alt = os.path.join(args.data_dir, 'validation')
        if os.path.exists(alt):
            data_dir = alt
        else:
            raise FileNotFoundError(f"Cannot find val/validation in {args.data_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Split directory not found: {data_dir}")

    # Build class_map from directories present on disk
    num_classes = getattr(args, 'num_classes', 7) or 7
    base_classes = IVF_7_CLASSES if num_classes == 7 else IVF_7_CLASSES

    class_map = {}
    for d in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, d)
        if not os.path.isdir(full_path):
            continue
        clean_d = d.strip()
        if clean_d in base_classes:
            class_map[d] = base_classes.index(clean_d)

    if not class_map:
        print(f"[Warning] Empty class_map for {data_dir} — using defaults")
        class_map = {c: i for i, c in enumerate(base_classes)}

    print(f"[get_2d_ivf_loader] class_map ({split}): {class_map}")

    img_size = getattr(args, 'img_size', 224) or 224

    dataset = IVF2DDataset(
        root=data_dir,
        class_map=class_map,
        img_size=img_size,
        is_training=is_training,
    )

    batch_size = args.batch_size if is_training else (
        getattr(args, 'validation_batch_size', None) or args.batch_size)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=args.workers,
        pin_memory=getattr(args, 'pin_mem', False),
        drop_last=is_training,
    )

    dist = dataset.get_class_distribution()
    if is_training:
        print(f"[get_2d_ivf_loader] Distribution ({split}, total={len(dataset):,}):")
        for stage, info in sorted(dist.items()):
            total = info['total']
            real  = info['real']
            pct   = 100.0 * total / len(dataset)
            print(f"  {stage:<12}: {total:>6,} ({pct:>5.1f}%) [Real: {real:>6,}]")

    return loader, dist
