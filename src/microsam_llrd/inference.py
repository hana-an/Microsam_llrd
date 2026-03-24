import os

import numpy as np

from micro_sam.evaluation.inference import run_instance_segmentation_with_decoder
import micro_sam.evaluation.instance_segmentation as instseg

from .eval_utils import get_paths


try:
    import tifffile
except ImportError:
    tifffile = None


def safe_imwrite(path, arr, **kwargs):
    arr = np.asarray(arr).squeeze()
    base, _ = os.path.splitext(path)
    path = base + ".tif"
    if tifffile is not None:
        tifffile.imwrite(path, arr)
    else:
        import imageio.v3 as iio

        iio.imwrite(path, arr)
    return path


def patch_micro_sam_writer():
    instseg.imageio.imwrite = safe_imwrite


def run_decoder_inference(dataset: str, checkpoint: str, model_type: str, experiment_folder: str, eval_root: str):
    patch_micro_sam_writer()
    os.makedirs(experiment_folder, exist_ok=True)

    val_imgs, val_gts = get_paths(dataset, split="val", eval_root=eval_root)
    test_imgs, _ = get_paths(dataset, split="test", eval_root=eval_root)

    pred_dir = run_instance_segmentation_with_decoder(
        checkpoint=checkpoint,
        model_type=model_type,
        experiment_folder=experiment_folder,
        val_image_paths=val_imgs,
        val_gt_paths=val_gts,
        test_image_paths=test_imgs,
    )
    return pred_dir
