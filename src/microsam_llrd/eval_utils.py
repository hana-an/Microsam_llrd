import os
from glob import glob


def get_paths(dataset: str, split: str = "test", eval_root: str = "/content/microsam_eval_data"):
    root = os.path.join(eval_root, dataset, split)
    image_paths = sorted(glob(os.path.join(root, "images", "*")))
    gt_paths = sorted(glob(os.path.join(root, "labels", "*")))
    return image_paths, gt_paths
