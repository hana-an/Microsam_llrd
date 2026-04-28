import argparse
import time
from pathlib import Path
import nibabel as nib

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from micro_sam.instance_segmentation import (
    get_predictor_and_decoder,
    InstanceSegmentationWithDecoder,
)


def load_image(path: str) -> np.ndarray:
    path = Path(path)

    if path.name.endswith(".nii") or path.name.endswith(".nii.gz"):
        img = nib.load(str(path))
        arr = img.get_fdata()
        arr = np.asarray(arr)

        # Convert NIfTI from (H, W, Z) to (Z, H, W)
        # because the script expects slice-first 3D volumes.
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))

        return np.squeeze(arr)

    arr = iio.imread(path)
    arr = np.asarray(arr)
    return np.squeeze(arr)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image -= image.min()
    if image.max() > 0:
        image /= image.max()
    image = (image * 255).astype(np.uint8)
    return image


def prepare_image_for_microsam(image: np.ndarray) -> np.ndarray:
    image_u8 = normalize_to_uint8(image)
    if image_u8.ndim == 2:
        image_u8 = np.stack([image_u8] * 3, axis=-1)
    return image_u8


def synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def compute_overlap_metrics(pred: np.ndarray, gt: np.ndarray):
    pred_bin = pred > 0
    gt_bin = gt > 0

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    vs = 1.0 - (abs(fn - fp) / (2 * tp + fp + fn + 1e-8))

    return dice, iou, precision, recall, vs


def _get_surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    structure = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    surface = mask ^ eroded
    return surface


def compute_surface_distance_metrics(pred: np.ndarray, gt: np.ndarray):
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)

    if not np.any(pred_bin) and not np.any(gt_bin):
        return 0.0, 0.0

    if not np.any(pred_bin) or not np.any(gt_bin):
        return np.inf, np.inf

    pred_surface = _get_surface(pred_bin)
    gt_surface = _get_surface(gt_bin)

    if not np.any(pred_surface) or not np.any(gt_surface):
        return np.inf, np.inf

    dt_to_gt = distance_transform_edt(~gt_surface)
    dt_to_pred = distance_transform_edt(~pred_surface)

    pred_to_gt = dt_to_gt[pred_surface]
    gt_to_pred = dt_to_pred[gt_surface]

    all_surface_distances = np.concatenate([pred_to_gt, gt_to_pred])

    hd95 = np.percentile(all_surface_distances, 95)
    asd = all_surface_distances.mean()

    return float(hd95), float(asd)


def compute_volume_metrics(pred: np.ndarray, gt: np.ndarray):
    dice, iou, precision, recall, vs = compute_overlap_metrics(pred, gt)
    hd95, asd = compute_surface_distance_metrics(pred, gt)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "vs": vs,
        "hd95": hd95,
        "asd": asd,
    }


def compute_pr_curve(score_map: np.ndarray, gt: np.ndarray):
    gt_bin = (gt > 0).astype(np.uint8).flatten()
    scores = score_map.astype(np.float32).flatten()

    precision, recall, _ = precision_recall_curve(gt_bin, scores)
    pr_auc = auc(recall, precision)
    return precision, recall, float(pr_auc)


def compute_roc_curve(score_map: np.ndarray, gt: np.ndarray):
    gt_bin = (gt > 0).astype(np.uint8).flatten()
    scores = score_map.astype(np.float32).flatten()

    fpr, tpr, _ = roc_curve(gt_bin, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, float(roc_auc)


def plot_pr_curve(precision, recall, pr_auc, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved PR curve to {save_path}")

    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved ROC curve to {save_path}")

    plt.show()


def get_score_map_from_prediction(pred: np.ndarray) -> np.ndarray:
    # Fallback: use binary mask as a coarse score map.
    # Replace this later with true confidence / probability map if available.
    return (pred > 0).astype(np.float32)


def predict_2d_with_timing(image: np.ndarray, predictor, decoder):
    t0 = time.perf_counter()
    image_u8 = prepare_image_for_microsam(image)
    t1 = time.perf_counter()
    preprocessing_time = t1 - t0

    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)

    synchronize_if_needed()
    t2 = time.perf_counter()
    segmenter.initialize(image_u8)
    instances = segmenter.generate()
    synchronize_if_needed()
    t3 = time.perf_counter()
    inference_time = t3 - t2

    return instances, preprocessing_time, inference_time


def predict_3d_with_timing_and_metrics(
    volume: np.ndarray,
    predictor,
    decoder,
    gt: np.ndarray | None = None,
):
    preds = []
    preprocessing_times = []
    inference_times = []

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    vs_scores = []

    for z in range(volume.shape[0]):
        pred_slice, pre_time, inf_time = predict_2d_with_timing(volume[z], predictor, decoder)

        preds.append(pred_slice)
        preprocessing_times.append(pre_time)
        inference_times.append(inf_time)

        msg = (
            f"Slice {z + 1}/{volume.shape[0]} | "
            f"pre={pre_time:.4f}s | inf={inf_time:.4f}s"
        )

        if gt is not None:
            d, i, p, r, vs = compute_overlap_metrics(pred_slice, gt[z])
            dice_scores.append(d)
            iou_scores.append(i)
            precision_scores.append(p)
            recall_scores.append(r)
            vs_scores.append(vs)

            msg += (
                f" | Dice={d:.4f}"
                f" IoU={i:.4f}"
                f" P={p:.4f}"
                f" R={r:.4f}"
                f" VS={vs:.4f}"
            )

        print(msg)

    result = {
        "pred": np.stack(preds, axis=0),
        "pre_times": np.array(preprocessing_times),
        "inf_times": np.array(inference_times),
    }

    if gt is not None:
        result.update({
            "dice": np.array(dice_scores),
            "iou": np.array(iou_scores),
            "precision": np.array(precision_scores),
            "recall": np.array(recall_scores),
            "vs": np.array(vs_scores),
        })

    return result


def save_prediction(pred: np.ndarray, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pred.ndim == 2:
        iio.imwrite(output_path, pred.astype(np.uint16))
    elif pred.ndim == 3:
        tifffile.imwrite(output_path, pred.astype(np.uint16))
    else:
        raise ValueError(f"Unsupported prediction ndim: {pred.ndim}")


def visualize_2d(
    image: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray | None = None,
    save_fig: str | None = None
) -> None:
    if gt is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        pred_bin = pred > 0
        gt_bin = gt > 0

        tp = pred_bin & gt_bin
        fp = pred_bin & ~gt_bin
        fn = ~pred_bin & gt_bin

        error_map = np.zeros((*image.shape, 3), dtype=np.float32)
        error_map[..., 1] = tp.astype(np.float32)
        error_map[..., 0] = fp.astype(np.float32)
        error_map[..., 2] = fn.astype(np.float32)

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(gt_bin, alpha=0.4, cmap="Reds")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(pred_bin, alpha=0.4, cmap="Greens")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        axes[3].imshow(image, cmap="gray")
        axes[3].imshow(error_map, alpha=0.5)
        axes[3].set_title("Error Map\nGreen=TP, Red=FP, Blue=FN")
        axes[3].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")

        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(pred > 0, alpha=0.4, cmap="Greens")
        axes[1].set_title("Prediction")
        axes[1].axis("off")

    plt.tight_layout()
    if save_fig:
        save_fig = Path(save_fig)
        save_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_fig}")
    plt.show()


def visualize_3d(
    volume: np.ndarray,
    pred: np.ndarray,
    slice_index: int,
    gt: np.ndarray | None = None,
    save_fig: str | None = None
) -> None:
    slice_index = max(0, min(slice_index, volume.shape[0] - 1))

    image = volume[slice_index]
    pred_slice = pred[slice_index]

    if gt is not None:
        gt_slice = gt[slice_index]

        pred_bin = pred_slice > 0
        gt_bin = gt_slice > 0

        tp = pred_bin & gt_bin
        fp = pred_bin & ~gt_bin
        fn = ~pred_bin & gt_bin

        error_map = np.zeros((*image.shape, 3), dtype=np.float32)
        error_map[..., 1] = tp.astype(np.float32)
        error_map[..., 0] = fp.astype(np.float32)
        error_map[..., 2] = fn.astype(np.float32)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title(f"Input Slice {slice_index}")
        axes[0].axis("off")

        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(gt_bin, alpha=0.4, cmap="Reds")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(image, cmap="gray")
        axes[2].imshow(pred_bin, alpha=0.4, cmap="Greens")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        axes[3].imshow(image, cmap="gray")
        axes[3].imshow(error_map, alpha=0.5)
        axes[3].set_title("Error Map\nGreen=TP, Red=FP, Blue=FN")
        axes[3].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title(f"Input Slice {slice_index}")
        axes[0].axis("off")

        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(pred_slice > 0, alpha=0.4, cmap="Greens")
        axes[1].set_title("Prediction")
        axes[1].axis("off")

    plt.tight_layout()
    if save_fig:
        save_fig = Path(save_fig)
        save_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_fig}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run 2D or 3D inference with timing, overlap metrics, VS, HD95, ASD, PR and ROC curves."
    )
    parser.add_argument("--image_path", required=True, help="Path to input .png or .tif")
    parser.add_argument("--gt_path", default=None, help="Optional path to ground-truth mask")
    parser.add_argument("--checkpoint", help="Path to checkpoint")
    parser.add_argument("--model_type", default="vit_b_em_organelles")
    parser.add_argument("--output_path", default="prediction.tif", help="Where to save prediction")
    parser.add_argument("--save_fig", default=None, help="Optional PNG for visualization")
    parser.add_argument("--slice_index", type=int, default=0, help="Slice index to visualize for 3D")
    parser.add_argument("--save_pr_curve", default=None, help="Optional path to save PR curve")
    parser.add_argument("--save_roc_curve", default=None, help="Optional path to save ROC curve")
    args = parser.parse_args()

    t0 = time.perf_counter()
    image = load_image(args.image_path)
    t1 = time.perf_counter()
    print(f"Loaded image shape: {image.shape}")
    print(f"Image loading time: {t1 - t0:.4f} sec")

    gt = None
    if args.gt_path is not None:
        gt = load_image(args.gt_path)
        print(f"Loaded ground truth shape: {gt.shape}")
        if gt.shape != image.shape:
            raise ValueError(f"Ground truth shape {gt.shape} does not match input image shape {image.shape}")

    synchronize_if_needed()
    t2 = time.perf_counter()
    predictor, decoder = get_predictor_and_decoder(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
    )
    synchronize_if_needed()
    t3 = time.perf_counter()
    print(f"Model loading time: {t3 - t2:.4f} sec")

    if image.ndim == 2:
        print("Detected 2D image")

        pred, pre_time, inf_time = predict_2d_with_timing(image, predictor, decoder)
        print(f"Preprocessing time: {pre_time:.4f} sec")
        print(f"Inference time: {inf_time:.4f} sec")

        if gt is not None:
            dice, iou, precision, recall, vs = compute_overlap_metrics(pred, gt)
            hd95, asd = compute_surface_distance_metrics(pred, gt)

            print("\n--- 2D Performance Summary ---")
            print(f"Dice:      {dice:.4f}")
            print(f"IoU:       {iou:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"VS:        {vs:.4f}")
            print(f"HD95:      {hd95:.4f}")
            print(f"ASD:       {asd:.4f}")

            score_map = get_score_map_from_prediction(pred)

            pr_precision, pr_recall, pr_auc = compute_pr_curve(score_map, gt)
            roc_fpr, roc_tpr, roc_auc = compute_roc_curve(score_map, gt)

            print("\n--- Curve-based Metrics ---")
            print(f"PR AUC / AP: {pr_auc:.4f}")
            print(f"ROC AUC:     {roc_auc:.4f}")

            plot_pr_curve(pr_precision, pr_recall, pr_auc, save_path=args.save_pr_curve)
            plot_roc_curve(roc_fpr, roc_tpr, roc_auc, save_path=args.save_roc_curve)

    elif image.ndim == 3:
        print("Detected 3D volume; running slice-wise 2D inference")

        result = predict_3d_with_timing_and_metrics(image, predictor, decoder, gt)
        pred = result["pred"]

        print("\n--- Timing Summary ---")
        print(f"Preprocessing Mean: {result['pre_times'].mean():.4f} sec")
        print(f"Preprocessing Std:  {result['pre_times'].std():.4f} sec")
        print(f"Inference Mean:     {result['inf_times'].mean():.4f} sec")
        print(f"Inference Std:      {result['inf_times'].std():.4f} sec")
        print(f"Inference Min:      {result['inf_times'].min():.4f} sec")
        print(f"Inference Max:      {result['inf_times'].max():.4f} sec")

        if gt is not None:
            print("\n--- Slice-wise Performance Summary (mean ± std) ---")
            print(f"Dice:      {result['dice'].mean():.4f} ± {result['dice'].std():.4f}")
            print(f"IoU:       {result['iou'].mean():.4f} ± {result['iou'].std():.4f}")
            print(f"Precision: {result['precision'].mean():.4f} ± {result['precision'].std():.4f}")
            print(f"Recall:    {result['recall'].mean():.4f} ± {result['recall'].std():.4f}")
            print(f"VS:        {result['vs'].mean():.4f} ± {result['vs'].std():.4f}")

            volume_metrics = compute_volume_metrics(pred, gt)

            print("\n--- Volume-level Performance Summary ---")
            print(f"Dice:      {volume_metrics['dice']:.4f}")
            print(f"IoU:       {volume_metrics['iou']:.4f}")
            print(f"Precision: {volume_metrics['precision']:.4f}")
            print(f"Recall:    {volume_metrics['recall']:.4f}")
            print(f"VS:        {volume_metrics['vs']:.4f}")
            print(f"HD95:      {volume_metrics['hd95']:.4f}")
            print(f"ASD:       {volume_metrics['asd']:.4f}")

            score_map = get_score_map_from_prediction(pred)

            pr_precision, pr_recall, pr_auc = compute_pr_curve(score_map, gt)
            roc_fpr, roc_tpr, roc_auc = compute_roc_curve(score_map, gt)

            print("\n--- Curve-based Metrics (Volume-level) ---")
            print(f"PR AUC / AP: {pr_auc:.4f}")
            print(f"ROC AUC:     {roc_auc:.4f}")

            plot_pr_curve(pr_precision, pr_recall, pr_auc, save_path=args.save_pr_curve)
            plot_roc_curve(roc_fpr, roc_tpr, roc_auc, save_path=args.save_roc_curve)

    else:
        raise ValueError(f"Unsupported input shape: {image.shape}")

    t4 = time.perf_counter()
    save_prediction(pred, args.output_path)
    t5 = time.perf_counter()
    print(f"Saving time: {t5 - t4:.4f} sec")
    print(f"Saved prediction to {args.output_path}")

    if image.ndim == 2:
        visualize_2d(image, pred, gt=gt, save_fig=args.save_fig)
    else:
        visualize_3d(image, pred, args.slice_index, gt=gt, save_fig=args.save_fig)


if __name__ == "__main__":
    main()