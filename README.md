# microsam-llrd

LoRA + layer-wise learning rate decay (LLRD) training utilities for MicroSAM on EM mitochondria datasets.


## What is included

- dataset loading for Lucchi, UroCell, Kasthuri, and VNC
- augmentation-based upsampling to reduce dataset imbalance
- LoRA-only LLRD optimizer grouping
- generalist training entry point
- LoRA checkpoint merge utility
- evaluation data export utility
- instance segmentation inference utility
- evaluation utility

## Repository layout

```text
microsam-llrd/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/
│   └── microsam_llrd/
│       ├── __init__.py
│       ├── datasets.py
│       ├── llrd.py
│       ├── train.py
│       ├── merge_lora.py
│       ├── eval_utils.py
│       └── inference.py
├── scripts/
│   ├── train_generalist.py
│   ├── merge_checkpoint.py
│   ├── export_eval_data.py
│   ├── run_instance_segmentation.py
│   └── evaluate_predictions.py
└── notebooks/
    └── 20Kaugmentation_exp1_LLRD_LORA_FFT_Training_luchii_urocell_kasturi_vnc.ipynb
```

## Installation

Create your environment first, then install this repo in editable mode.

```bash
git clone https://github.com/YOUR_USERNAME/microsam-llrd.git
cd microsam-llrd
pip install -e .
```

You will also need a working install of:

- `torch`
- `torch-em`
- `micro_sam`
- dataset IO dependencies used by your environment

Because `micro_sam` and `torch-em` setups differ across CUDA, Colab, Linux, and ARM systems, they are left as normal requirements rather than hard-pinned here.

## Dataset layout

Expected root:

```text
em_data/
├── lucchi/
├── urocell/
├── kasthuri/
└── vnc/
```

## Training

Example:

```bash
python scripts/train_generalist.py   --input_path /path/to/em_data   --save_root /path/to/microsam_runs   --iterations 8500   --lora_rank 8   --lora_start_block 6   --lr 1e-4   --layer_decay 0.95   --lr_other_factor 0.1   --lr_unetr_factor 0.1   --augment_to_max_size   --n_objects 10
```

## Merge LoRA checkpoint

```bash
python scripts/merge_checkpoint.py   --best-pt /path/to/best.pt   --out-pt /path/to/mergedbest.pt   --model-type vit_b_em_organelles   --rank 8   --lora-start 6
```

## Export evaluation data

```bash
python scripts/export_eval_data.py   --input_path /path/to/em_data   --dataset kasthuri   --out_root /path/to/microsam_eval_data
```

## Run inference

```bash
python scripts/run_instance_segmentation.py   --dataset lucchi   --eval_root /path/to/microsam_eval_data   --checkpoint /path/to/mergedbest.pt   --experiment_dir /path/to/microsam_eval_runs/lucchi_ais   --model_type vit_b_em_organelles
```

## Evaluate predictions

```bash
python scripts/evaluate_predictions.py   --dataset lucchi   --eval_root /path/to/microsam_eval_data   --experiment_dir /path/to/microsam_eval_runs/lucchi_ais
```

## Notes

- Colab notebook is provided for easy execution
  
