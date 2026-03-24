import argparse
import os
import time

import torch
import torch_em

from micro_sam.training import joint_sam_trainer as joint_trainers
from micro_sam.training.util import ConvertToSamInputs, get_trainable_sam_model
from micro_sam.util import export_custom_sam_model, get_device

from .datasets import get_generalist_lucchi_urocell_kasthuri_vnc_loaders
from .llrd import build_llrd_groups_lora_only


def train_sam_lora_llrd(
    name,
    model_type,
    train_loader,
    val_loader,
    n_objects_per_batch,
    n_sub_iteration,
    checkpoint_path,
    device,
    lr,
    n_iterations,
    save_root,
    scheduler_kwargs,
    box_distortion_factor,
    peft_kwargs,
    layer_decay,
    lora_start_block,
    lr_other_factor,
    lr_unetr_factor,
    weight_decay=1e-4,
):
    device = get_device(device)

    model, state = get_trainable_sam_model(
        model_type=model_type,
        device=device,
        freeze=None,
        checkpoint_path=checkpoint_path,
        return_state=True,
        peft_kwargs=peft_kwargs,
    )

    n_trainable = sum(p.requires_grad for p in model.parameters())
    n_lora_named = sum(p.requires_grad and ("lora" in n.lower()) for n, p in model.named_parameters())
    print("Trainable params:", n_trainable, "| LoRA-named trainables:", n_lora_named)

    convert_inputs = ConvertToSamInputs(transform=model.transform, box_distortion_factor=box_distortion_factor)

    from micro_sam.instance_segmentation import get_unetr

    unetr = get_unetr(
        image_encoder=model.sam.image_encoder,
        decoder_state=state.get("decoder_state", None),
        device=device,
    )

    lr_other = lr * lr_other_factor
    lr_unetr = lr * lr_unetr_factor

    param_groups = build_llrd_groups_lora_only(
        model=model,
        base_lr=lr,
        layer_decay=layer_decay,
        start_block=lora_start_block,
        lr_other=lr_other,
    )

    unetr_decoder_params = []
    for pname, p in unetr.named_parameters():
        if p.requires_grad and not pname.startswith("encoder"):
            unetr_decoder_params.append(p)
    if unetr_decoder_params:
        param_groups.append({"params": unetr_decoder_params, "lr": lr_unetr})

    group_lrs = [g["lr"] for g in param_groups]
    print("Optimizer groups:", len(param_groups))
    print("Group LRs:", group_lrs)
    print("Group LRs (min..max):", min(group_lrs), "..", max(group_lrs))
    print("LR other factor:", lr_other_factor, "=>", lr_other)
    print("LR unetr factor:", lr_unetr_factor, "=>", lr_unetr)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    if scheduler_kwargs is None:
        scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 15}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **scheduler_kwargs)

    instance_seg_loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=True)

    trainer = joint_trainers.JointSamTrainer(
        name=name,
        save_root=save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        logger=joint_trainers.JointSamLogger,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=n_sub_iteration,
        compile_model=False,
        unetr=unetr,
        instance_loss=instance_seg_loss,
        instance_metric=instance_seg_loss,
        early_stopping=None,
        mask_prob=0.5,
    )

    start_time = time.time()
    trainer.fit(iterations=n_iterations, overwrite_training=True)
    end_time = time.time()
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes ({(end_time - start_time) / 3600:.2f} hours)")


def finetune_mito_nuc_em_generalist(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = args.model_type

    checkpoint_path = args.checkpoint_path
    if checkpoint_path in (None, "", "none", "None"):
        checkpoint_path = None
    if checkpoint_path is not None and not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    patch_shape = (1, 384, 384)
    n_objects_per_batch = args.n_objects

    checkpoint_name = (
        f"{args.model_type}/mito_nuc_em_generalist_sam_lora_llrd_r{args.lora_rank}"
        f"_d{args.layer_decay}_lr{args.lr}_of{args.lr_other_factor}_uf{args.lr_unetr_factor}"
        f"_aug{int(args.augment_to_max_size)}"
    )

    train_loader, val_loader = get_generalist_lucchi_urocell_kasthuri_vnc_loaders(
        input_path=args.input_path,
        patch_shape=patch_shape,
        num_workers=args.num_workers,
        uro_target="mito",
        uro_val_fraction=args.uro_val_fraction,
        vnc_val_fraction=args.vnc_val_fraction,
        split_seed=args.split_seed,
        batch_size=1,
        augment_to_max_size=args.augment_to_max_size,
    )

    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 15}
    peft_kwargs = {
        "rank": args.lora_rank,
        "update_matrices": ["q", "v"],
        "attention_layers_to_update": list(range(args.lora_start_block, 12)),
    }

    print("Starting generalist training (LoRA + LLRD)")
    print("Model type:", model_type)
    print("Checkpoint path:", checkpoint_path)
    print("Iterations:", args.iterations)
    print("Objects per batch:", n_objects_per_batch)
    print("Device:", device)
    print("Decoder enabled:", True)
    print("Augment to max size:", args.augment_to_max_size)
    print("LoRA rank:", args.lora_rank)
    print("LoRA blocks:", peft_kwargs["attention_layers_to_update"])
    print("LoRA matrices:", peft_kwargs["update_matrices"])
    print("Top LR:", args.lr)
    print("LLRD:", args.layer_decay)
    print("Other LR factor:", args.lr_other_factor)
    print("UNETR LR factor:", args.lr_unetr_factor)

    train_sam_lora_llrd(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=4,
        checkpoint_path=checkpoint_path,
        device=device,
        lr=args.lr,
        n_iterations=args.iterations,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        box_distortion_factor=0.05,
        peft_kwargs=peft_kwargs,
        layer_decay=args.layer_decay,
        lora_start_block=args.lora_start_block,
        lr_other_factor=args.lr_other_factor,
        lr_unetr_factor=args.lr_unetr_factor,
        weight_decay=1e-4,
    )

    if args.export_path is not None:
        best_ckpt = os.path.join(
            "" if args.save_root is None else args.save_root,
            "checkpoints",
            checkpoint_name,
            "best.pt",
        )
        export_custom_sam_model(
            checkpoint_path=best_ckpt,
            model_type=model_type,
            save_path=args.export_path,
        )


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", default="/content/em_data")
    parser.add_argument("--model_type", "-m", default="vit_b_em_organelles")
    parser.add_argument("--save_root", "-s", default="/content/microsam_runs")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--export_path", "-e", default=None)
    parser.add_argument("--n_objects", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--uro_val_fraction", type=float, default=0.1)
    parser.add_argument("--vnc_val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", "-c", default=None)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_start_block", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer_decay", type=float, default=0.95)
    parser.add_argument("--lr_other_factor", type=float, default=0.1)
    parser.add_argument("--lr_unetr_factor", type=float, default=0.1)
    parser.add_argument(
        "--augment_to_max_size",
        action="store_true",
        help="Augment and upsample smaller training datasets to the largest dataset size",
    )
    return parser
