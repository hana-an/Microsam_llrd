import argparse
import os

import imageio.v2 as imageio
import numpy as np

from microsam_llrd.datasets import get_eval_loaders


def ensure_dirs(base):
    for s in ("val", "test"):
        os.makedirs(os.path.join(base, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(base, s, "labels"), exist_ok=True)


def to_uint8(x):
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (255 * x).astype(np.uint8)


def to_2d(a):
    a = np.asarray(a)
    a = np.squeeze(a)
    while a.ndim > 2:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"Expected 2D, got {a.shape}")
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="/content/em_data")
    parser.add_argument("--dataset", required=True, choices=["lucchi", "urocell", "kasthuri", "vnc"])
    parser.add_argument("--inst_ch", type=int, default=0)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--out_root", default="/content/microsam_eval_data")
    parser.add_argument("--patch", nargs=3, type=int, default=[1, 512, 512])
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    patch = tuple(args.patch)
    loaders = get_eval_loaders(
        input_path=args.input_path,
        patch_shape=patch,
        num_workers=args.num_workers,
        uro_target="mito",
        uro_val_fraction=0.1,
        vnc_val_fraction=0.1,
        split_seed=0,
        batch_size=1,
    )

    lucchi_test_loader, urocell_val_loader, kasthuri_test_loader, vnc_val_loader = loaders
    loader_map = {
        "lucchi": lucchi_test_loader,
        "urocell": urocell_val_loader,
        "kasthuri": kasthuri_test_loader,
        "vnc": vnc_val_loader,
    }
    loader = loader_map[args.dataset]

    base = os.path.join(args.out_root, args.dataset)
    ensure_dirs(base)

    def dump(split, n):
        img_dir = os.path.join(base, split, "images")
        lab_dir = os.path.join(base, split, "labels")
        for i, (x, y) in enumerate(loader):
            if i >= n:
                break
            img = to_uint8(to_2d(x[0].numpy()))
            inst = to_2d(y[0, args.inst_ch].numpy()).astype(np.uint16)
            imageio.imwrite(f"{img_dir}/{args.dataset}_{split}_{i:04d}.png", img)
            imageio.imwrite(f"{lab_dir}/{args.dataset}_{split}_{i:04d}.png", inst)

    dump("val", args.n_val)
    dump("test", args.n_test)
    print(f"Exported evaluation data to: {base}")


if __name__ == "__main__":
    main()
