import argparse
import os
from glob import glob

from micro_sam.evaluation.evaluation import run_evaluation

from microsam_llrd.eval_utils import get_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["lucchi", "urocell", "kasthuri", "vnc"])
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--experiment_dir", required=True)
    args = parser.parse_args()

    pred_dir = os.path.join(args.experiment_dir, "instance_segmentation_with_decoder", "inference")
    _, gt_paths = get_paths(args.dataset, split="test", eval_root=args.eval_root)
    pred_paths = sorted(glob(os.path.join(pred_dir, "*.tif")))

    print("GT:", len(gt_paths))
    print("Pred:", len(pred_paths))
    print("Pred dir:", pred_dir)
    print("First pred:", pred_paths[0] if pred_paths else None)
    print("First gt:", gt_paths[0] if gt_paths else None)

    if len(gt_paths) != len(pred_paths):
        raise RuntimeError("Counts mismatch; filenames might not align.")

    out_csv = os.path.join(args.experiment_dir, "results", "instance_segmentation_with_decoder.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res = run_evaluation(gt_paths, pred_paths, save_path=out_csv)
    print(f"Saved: {out_csv}")
    print(res)


if __name__ == "__main__":
    main()
