import argparse
import os

from microsam_llrd.inference import run_decoder_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["lucchi", "urocell", "kasthuri", "vnc"])
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--model_type", default="vit_b_em_organelles")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    pred_dir = run_decoder_inference(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        experiment_folder=args.experiment_dir,
        eval_root=args.eval_root,
    )
    print(f"Predictions written to: {pred_dir}")


if __name__ == "__main__":
    main()
