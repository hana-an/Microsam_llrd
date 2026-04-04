import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from micro_sam.instance_segmentation import (
    get_predictor_and_decoder,
    InstanceSegmentationWithDecoder,
)


def load_image(path: str) -> np.ndarray:
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


def predict_2d(image: np.ndarray, predictor, decoder) -> np.ndarray:
    image_u8 = prepare_image_for_microsam(image)

    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)
    segmenter.initialize(image_u8)
    instances = segmenter.generate()
    return instances


def predict_3d(volume: np.ndarray, predictor, decoder) -> np.ndarray:
    preds = []
    for z in range(volume.shape[0]):
        pred = predict_2d(volume[z], predictor, decoder)
        preds.append(pred)
        print(f"Processed slice {z + 1}/{volume.shape[0]}")
    return np.stack(preds, axis=0)


def save_prediction(pred: np.ndarray, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pred.ndim == 2:
        iio.imwrite(output_path, pred.astype(np.uint16))
    elif pred.ndim == 3:
        tifffile.imwrite(output_path, pred.astype(np.uint16))
    else:
        raise ValueError(f"Unsupported prediction ndim: {pred.ndim}")


def visualize_2d(image: np.ndarray, pred: np.ndarray, save_fig: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(pred > 0, alpha=0.35)
    axes[1].set_title("Prediction Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_fig}")
    plt.show()


def visualize_3d(volume: np.ndarray, pred: np.ndarray, slice_index: int, save_fig: str | None = None) -> None:
    slice_index = max(0, min(slice_index, volume.shape[0] - 1))
    image = volume[slice_index]
    mask = pred[slice_index]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Input Slice {slice_index}")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(mask > 0, alpha=0.35)
    axes[1].set_title(f"Prediction Overlay Slice {slice_index}")
    axes[1].axis("off")

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_fig}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run 2D or 3D inference with a repo-clean checkpoint.")
    parser.add_argument("--image_path", required=True, help="Path to input .png or .tif")
    parser.add_argument("--checkpoint", required=True, help="Path to repo-clean merged checkpoint")
    parser.add_argument("--model_type", default="vit_b_em_organelles")
    parser.add_argument("--output_path", default="prediction.tif", help="Where to save prediction")
    parser.add_argument("--save_fig", default=None, help="Optional PNG for visualization")
    parser.add_argument("--slice_index", type=int, default=0, help="Slice index to visualize for 3D")
    args = parser.parse_args()

    image = load_image(args.image_path)
    print(f"Loaded image shape: {image.shape}")

    predictor, decoder = get_predictor_and_decoder(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
    )

    if image.ndim == 2:
        print("Detected 2D image")
        pred = predict_2d(image, predictor, decoder)
        save_prediction(pred, args.output_path)
        visualize_2d(image, pred, args.save_fig)

    elif image.ndim == 3:
        print("Detected 3D volume; running slice-wise 2D inference")
        pred = predict_3d(image, predictor, decoder)
        save_prediction(pred, args.output_path)
        visualize_3d(image, pred, args.slice_index, args.save_fig)

    else:
        raise ValueError(f"Unsupported input shape: {image.shape}")

    print(f"Saved prediction to {args.output_path}")


if __name__ == "__main__":
    main()