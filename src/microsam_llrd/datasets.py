import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import random_split

from torch_em import get_data_loader
from torch_em.data import ConcatDataset, MinInstanceSampler, datasets
from micro_sam.training.util import ResizeRawTrafo, ResizeLabelTrafo


def _to_tt_split(split: str) -> str:
    return "test" if split in ("val", "valid", "validation") else split


def _split_dataset(ds, val_fraction: float = 0.1, seed: int = 0):
    n_total = len(ds)
    n_val = max(1, int(val_fraction * n_total))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=gen)


def _common_transforms(patch_shape: Tuple[int, int, int]):
    raw_tf = ResizeRawTrafo(patch_shape[1:], do_rescaling=False, ensure_rgb=False)
    lab_tf = ResizeLabelTrafo(patch_shape[1:])
    return raw_tf, lab_tf


def _set_sampling_attempts(ds, n: int = 5000):
    if hasattr(ds, "max_sampling_attempts"):
        ds.max_sampling_attempts = n
    if hasattr(ds, "dataset") and hasattr(ds.dataset, "max_sampling_attempts"):
        ds.dataset.max_sampling_attempts = n


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy(), True, x.dtype
    return np.asarray(x), False, None


def _from_numpy(x, was_tensor, dtype):
    x = np.ascontiguousarray(x)
    if was_tensor:
        return torch.from_numpy(x).to(dtype=dtype)
    return x


def augment_em(raw, label):
    raw_np, raw_was_tensor, raw_dtype = _to_numpy(raw)
    lab_np, lab_was_tensor, lab_dtype = _to_numpy(label)

    if random.random() < 0.5:
        raw_np = np.flip(raw_np, axis=-1)
        lab_np = np.flip(lab_np, axis=-1)

    if random.random() < 0.5:
        raw_np = np.flip(raw_np, axis=-2)
        lab_np = np.flip(lab_np, axis=-2)

    k = random.randint(0, 3)
    if k > 0:
        raw_np = np.rot90(raw_np, k, axes=(-2, -1))
        lab_np = np.rot90(lab_np, k, axes=(-2, -1))

    if random.random() < 0.3:
        factor = 0.9 + 0.2 * random.random()
        raw_np = raw_np * factor

    if random.random() < 0.3:
        bias = np.random.uniform(-0.05, 0.05)
        raw_np = raw_np + bias

    if random.random() < 0.3:
        sigma = np.random.uniform(0.005, 0.02)
        raw_np = raw_np + np.random.normal(0.0, sigma, size=raw_np.shape)

    raw_out = _from_numpy(raw_np, raw_was_tensor, raw_dtype)
    lab_out = _from_numpy(lab_np, lab_was_tensor, lab_dtype)
    return raw_out, lab_out


class AugmentedUpsampledDataset(torch.utils.data.Dataset):
    """Upsample a dataset to target_size by cycling samples and applying augmentation."""

    def __init__(self, dataset, target_size: int):
        self.dataset = dataset
        self.target_size = int(target_size)
        self.ndim = getattr(dataset, "ndim", None)
        self.raw_channels = getattr(dataset, "raw_channels", None)
        self.n_samples = self.target_size
        if hasattr(dataset, "max_sampling_attempts"):
            self.max_sampling_attempts = dataset.max_sampling_attempts

    def __len__(self):
        return self.target_size

    def __getitem__(self, idx):
        base_idx = idx % len(self.dataset)
        sample = self.dataset[base_idx]

        if isinstance(sample, dict):
            sample = dict(sample)
            raw_key = "raw" if "raw" in sample else None
            label_key = "label" if "label" in sample else ("labels" if "labels" in sample else None)
            if raw_key is None or label_key is None:
                raise KeyError(
                    f"Expected sample dict to contain 'raw' and 'label'/'labels', got keys: {list(sample.keys())}"
                )
            raw_aug, label_aug = augment_em(sample[raw_key], sample[label_key])
            sample[raw_key] = raw_aug
            sample[label_key] = label_aug
            return sample

        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            raw_aug, label_aug = augment_em(sample[0], sample[1])
            rest = list(sample[2:])
            return type(sample)((raw_aug, label_aug, *rest)) if isinstance(sample, tuple) else [raw_aug, label_aug, *rest]

        raise TypeError(f"Unsupported sample type from dataset: {type(sample)}")


def _get_lucchi(input_path, patch_shape, split):
    split = _to_tt_split(split)
    path = os.path.join(input_path, "lucchi")
    raw_tf, lab_tf = _common_transforms(patch_shape)
    return datasets.get_lucchi_dataset(
        path=path,
        patch_shape=patch_shape,
        download=True,
        split=split,
        ndim=2,
        sampler=MinInstanceSampler(min_num_instances=2),
        raw_transform=raw_tf,
        label_transform=lab_tf,
    )


def _get_urocell_full(input_path, patch_shape, target="mito"):
    path = os.path.join(input_path, "urocell")
    raw_tf, lab_tf = _common_transforms(patch_shape)
    return datasets.get_uro_cell_dataset(
        path=path,
        target=target,
        patch_shape=patch_shape,
        download=True,
        ndim=2,
        sampler=MinInstanceSampler(min_num_instances=2),
        raw_transform=raw_tf,
        label_transform=lab_tf,
    )


def _get_kasthuri(input_path, patch_shape, split):
    split = _to_tt_split(split)
    path = os.path.join(input_path, "kasthuri")
    raw_tf, lab_tf = _common_transforms(patch_shape)
    return datasets.get_kasthuri_dataset(
        path=path,
        split=split,
        patch_shape=patch_shape,
        download=True,
        ndim=2,
        sampler=MinInstanceSampler(min_num_instances=2),
        raw_transform=raw_tf,
        label_transform=lab_tf,
    )


def _get_vnc_try_split(input_path, patch_shape, split):
    split = _to_tt_split(split)
    path = os.path.join(input_path, "vnc")
    raw_tf, lab_tf = _common_transforms(patch_shape)
    return datasets.get_vnc_mito_dataset(
        path=path,
        split=split,
        patch_shape=patch_shape,
        download=True,
        ndim=2,
        sampler=MinInstanceSampler(min_num_instances=2),
        raw_transform=raw_tf,
        label_transform=lab_tf,
    )


def _get_vnc_train_val(input_path, patch_shape, val_fraction: float = 0.1, seed: int = 0):
    try:
        vnc_train = _get_vnc_try_split(input_path, patch_shape, split="train")
        vnc_val = _get_vnc_try_split(input_path, patch_shape, split="val")
        return vnc_train, vnc_val
    except TypeError:
        path = os.path.join(input_path, "vnc")
        raw_tf, lab_tf = _common_transforms(patch_shape)
        vnc_full = datasets.get_vnc_mito_dataset(
            path=path,
            patch_shape=patch_shape,
            download=True,
            ndim=2,
            sampler=MinInstanceSampler(min_num_instances=2),
            raw_transform=raw_tf,
            label_transform=lab_tf,
        )
        vnc_train, vnc_val = _split_dataset(vnc_full, val_fraction=val_fraction, seed=seed)
        return vnc_train, vnc_val


def get_generalist_lucchi_urocell_kasthuri_vnc_loaders(
    input_path,
    patch_shape,
    num_workers: int = 0,
    uro_target: str = "mito",
    uro_val_fraction: float = 0.1,
    vnc_val_fraction: float = 0.1,
    split_seed: int = 0,
    batch_size: int = 1,
    augment_to_max_size: bool = True,
):
    lucchi_train = _get_lucchi(input_path, patch_shape, split="train")
    lucchi_val = _get_lucchi(input_path, patch_shape, split="val")

    uro_full = _get_urocell_full(input_path, patch_shape, target=uro_target)
    uro_train, uro_val = _split_dataset(uro_full, val_fraction=uro_val_fraction, seed=split_seed)

    kas_train = _get_kasthuri(input_path, patch_shape, split="train")
    kas_val = _get_kasthuri(input_path, patch_shape, split="val")

    vnc_train, vnc_val = _get_vnc_train_val(input_path, patch_shape, val_fraction=vnc_val_fraction, seed=split_seed)

    for ds in [lucchi_train, lucchi_val, kas_train, kas_val, uro_full, vnc_train, vnc_val]:
        _set_sampling_attempts(ds, 5000)

    if augment_to_max_size:
        target_size = max(len(lucchi_train), len(uro_train), len(kas_train), len(vnc_train))
        if len(lucchi_train) < target_size:
            lucchi_train = AugmentedUpsampledDataset(lucchi_train, target_size)
        if len(uro_train) < target_size:
            uro_train = AugmentedUpsampledDataset(uro_train, target_size)
        if len(kas_train) < target_size:
            kas_train = AugmentedUpsampledDataset(kas_train, target_size)
        if len(vnc_train) < target_size:
            vnc_train = AugmentedUpsampledDataset(vnc_train, target_size)

    train_ds = ConcatDataset(lucchi_train, uro_train, kas_train, vnc_train)
    val_ds = ConcatDataset(lucchi_val, uro_val, kas_val, vnc_val)

    train_loader = get_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = get_data_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def get_specialist_loaders(
    dataset,
    input_path,
    patch_shape,
    num_workers: int = 0,
    uro_target: str = "mito",
    uro_val_fraction: float = 0.1,
    vnc_val_fraction: float = 0.1,
    split_seed: int = 0,
    batch_size: int = 1,
):
    if dataset == "lucchi":
        train_ds = _get_lucchi(input_path, patch_shape, split="train")
        val_ds = _get_lucchi(input_path, patch_shape, split="val")
    elif dataset == "urocell":
        full_ds = _get_urocell_full(input_path, patch_shape, target=uro_target)
        train_ds, val_ds = _split_dataset(full_ds, val_fraction=uro_val_fraction, seed=split_seed)
    elif dataset == "kasthuri":
        train_ds = _get_kasthuri(input_path, patch_shape, split="train")
        val_ds = _get_kasthuri(input_path, patch_shape, split="val")
    elif dataset == "vnc":
        train_ds, val_ds = _get_vnc_train_val(input_path, patch_shape, val_fraction=vnc_val_fraction, seed=split_seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    for ds in [train_ds, val_ds]:
        _set_sampling_attempts(ds, 5000)

    train_loader = get_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = get_data_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def get_eval_loaders(
    input_path,
    patch_shape,
    num_workers: int = 0,
    uro_target: str = "mito",
    uro_val_fraction: float = 0.1,
    vnc_val_fraction: float = 0.1,
    split_seed: int = 0,
    batch_size: int = 1,
):
    lucchi_test = _get_lucchi(input_path, patch_shape, split="test")
    _set_sampling_attempts(lucchi_test, 5000)

    uro_full = _get_urocell_full(input_path, patch_shape, target=uro_target)
    _, uro_val = _split_dataset(uro_full, val_fraction=uro_val_fraction, seed=split_seed)
    _set_sampling_attempts(uro_full, 5000)

    kas_test = _get_kasthuri(input_path, patch_shape, split="test")
    _set_sampling_attempts(kas_test, 5000)

    vnc_train, vnc_val = _get_vnc_train_val(input_path, patch_shape, val_fraction=vnc_val_fraction, seed=split_seed)
    _set_sampling_attempts(vnc_train, 5000)
    _set_sampling_attempts(vnc_val, 5000)

    lucchi_test_loader = get_data_loader(lucchi_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    urocell_val_loader = get_data_loader(uro_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    kasthuri_test_loader = get_data_loader(kas_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    vnc_val_loader = get_data_loader(vnc_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return lucchi_test_loader, urocell_val_loader, kasthuri_test_loader, vnc_val_loader
