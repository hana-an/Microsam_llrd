import argparse

from microsam_llrd.merge_lora import merge_lora_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-pt", required=True)
    parser.add_argument("--out-pt", required=True)
    parser.add_argument("--model-type", default="vit_b_em_organelles")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-start", type=int, default=6)
    args = parser.parse_args()

    out = merge_lora_checkpoint(
        best_pt=args.best_pt,
        out_pt=args.out_pt,
        model_type=args.model_type,
        rank=args.rank,
        lora_start=args.lora_start,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
