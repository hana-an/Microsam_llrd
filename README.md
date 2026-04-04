# Microsam-LORA+LLRD

LoRA + layer-wise learning rate decay (LLRD) training utilities for MicroSAM on EM mitochondria datasets.


## What is included

- Dataset loading for Lucchi, UroCell, Kasthuri, and VNC
- Augmentation-based upsampling to reduce dataset imbalance
- LoRA-only LLRD optimizer grouping
- Generalist training entry point
- LoRA checkpoint merge utility
- Evaluation data export utility
- Instance segmentation inference utility
- Evaluation utility

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
│   ├── evaluate_predictions.py
│   └── test.py
└── notebooks/
    └── 20Kaugmentation_exp1_LLRD_LORA_FFT_Training_luchii_urocell_kasturi_vnc.ipynb
```

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/hana-an/Microsam_llrd.git
```

---

### Step 2: Setup Environment (Recommended)

This project depends on `torch_em` and `micro_sam`, which require `python-elf` (`elf.io`).  
A standard pip setup may fail for these dependencies.

It is recommended to use micromamba / conda.

#### Install micromamba

```bash
wget -qO micromamba.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest
tar -xjf micromamba.tar.bz2
export MAMBA_ROOT_PREFIX=/content/micromamba
./bin/micromamba shell init -s bash -r $MAMBA_ROOT_PREFIX
source ~/.bashrc
```
```bash
eval "$(/content/bin/micromamba shell hook -s bash)"
micromamba config append channels conda-forge
micromamba config set channel_priority strict
```

#### Create environment

```bash
micromamba create -y -n microsam_llrd \
  "python=3.12" \
  "pip" \
  "python-elf>=0.7.1" \
  "pytorch>=2.5" \
  "torchvision" \
  "torch_em>=0.8"
```

#### Activate environment

```bash
micromamba activate microsam_llrd
cd /content/Microsam_llrd
```

---

### Step 3: Install Dependencies

```bash
pip install timm tqdm xarray zarr==2.18.4 numcodecs==0.12.1 \
natsort pooch imagecodecs xxhash nibabel scipy scikit-image \
matplotlib imageio tifffile tensorboard netcdf4 segment-anything

pip install git+https://github.com/computational-cell-analytics/micro-sam.git
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

---

### Step 4: Install This Repository

```bash
pip install -e .
```

---

### Important

If you encounter:

```bash
ModuleNotFoundError: No module named 'elf'
```

Ensure that `python-elf` is installed via the conda/micromamba environment.

---

### Verification

```bash
python -c "import elf.io; print('elf.io OK')"
python -c "import torch_em; print('torch_em OK')"

python -c "import microsam_llrd; print('package OK')"
python -c "import microsam_llrd.llrd; print('llrd OK')"
python -c "import microsam_llrd.merge_lora; print('merge_lora OK')"
python -c "import microsam_llrd.inference; print('inference OK')"

python scripts/train_generalist.py --help
```

## Weights and Data

- **Pretrained Model (LoRA + LLRD merged)**  
  [Download Checkpoint for github_repo](https://drive.google.com/file/d/1XbfX4yiOwpgsHBuO3G-oUkSmyYleJzQA/view?usp=sharing)
  [Download Checkpoint for notebook execution](https://drive.google.com/file/d/1z1_U7h5Yfco9xxSzYauLm6KgIliZBHYP/view?usp=sharing)

- **Training Data (torch_em format)**  
  [Download em_data](https://drive.google.com/drive/folders/1gRXC9uEVipiJD49SRMK6zvDlCgV_HK_w?usp=sharing)

- **Evaluation Data**  
  [Download microsam_eval_data](https://drive.google.com/drive/folders/1Vy0Ryt8xDylkVJu_rQ0KcqLCWYyVMrBV?usp=sharing)


## Expected Dataset Layout

```bash
/path/to/em_data/
├── lucchi/
├── urocell/
├── kasthuri/
└── vnc/

/path/to/microsam_eval_data/
├── lucchi/
├── urocell/
├── kasthuri/
└── vnc/
```

## Training



```bash
python scripts/train_generalist.py   --input_path /path/to/em_data   --save_root /path/to/microsam_runs   --iterations 20000   --lora_rank 8   --lora_start_block 6   --lr 1e-4   --layer_decay 0.95   --lr_other_factor 0.1   --lr_unetr_factor 0.1   --augment_to_max_size   --n_objects 10
```

Example:
```bash
(microsam_llrd) /content/Microsam_llrd# python scripts/train_generalist.py \
  --input_path /content/drive/MyDrive/microsam_llrd_generalist_Aug20k/Aug_20k/em_data \
  --save_root /content/microsam_runs_test \
  --iterations 10 \
  --lora_rank 8 \
  --lora_start_block 6 \
  --lr 1e-4 \
  --layer_decay 0.95 \
  --lr_other_factor 0.1 \
  --lr_unetr_factor 0.1 \
  --n_objects 10 \
  --num_workers 0
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
Example:
```bash
python scripts/run_instance_segmentation.py \
  --dataset lucchi \
  --eval_root /content/drive/MyDrive/microsam_llrd_generalist_Aug20k/Aug_20k/microsam_eval_data \
  --checkpoint /content/drive/MyDrive/microsam_llrd_generalist_Aug20k/Aug_20k/microsam_runs/checkpoints/vit_b_em_organelles/repo_mergedbest.pt \
  --experiment_dir /content/microsam_inference_clean/lucchi \
  --model_type vit_b_em_organelles
```

## Evaluate predictions

```bash
python scripts/evaluate_predictions.py   --dataset lucchi   --eval_root /path/to/microsam_eval_data   --experiment_dir /path/to/microsam_eval_runs/lucchi_ais
```
Example:
```bash
python scripts/evaluate_predictions.py \
  --dataset lucchi \
  --eval_root /content/drive/MyDrive/microsam_llrd_generalist_Aug20k/Aug_20k/microsam_eval_data \
  --experiment_dir /content/microsam_inference_test/lucchi
```
## Evaluate predictions (2D/3D)
Download the small test data for quick evaluation and visualization
- **Evaluation Data**  
  [Small Sample data](https://drive.google.com/drive/folders/16qbHDoMAkDS0OmyPuD1UafufZpr0CIRT)
Example:
```bash
python scripts/test.py \
  --image_path test_data/3d/lucchi_stack.tif \
  --checkpoint checkpoints/repo_mergedbest.pt \
  --output_path outputs/lucchi_stack_pred.tif \
  --slice_index 5 \
  --save_fig outputs/lucchi_slice5_vis.png
```
## Notes

- Colab notebook MicroSAM_Generalist_LoRa_LLRD_Augm.ipynb is provided for easy execution
