import torch
from micro_sam.training.util import get_trainable_sam_model


def merge_lora_checkpoint(best_pt: str, out_pt: str, model_type: str, rank: int = 8, lora_start: int = 6):
    lora_layers = list(range(lora_start, 12))
    update = ["q", "v"]

    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError("Expected micro-sam checkpoint dict.")

    if "model_state" in ckpt:
        model_key = "model_state"
    elif "state_dict" in ckpt:
        model_key = "state_dict"
    else:
        raise RuntimeError(f"Cannot find model weights key. Keys: {list(ckpt.keys())}")

    decoder_keys = [k for k in ckpt.keys() if "decoder" in k.lower()]
    if not decoder_keys:
        raise RuntimeError(f"No decoder state found in ckpt keys: {list(ckpt.keys())}")

    trainable = get_trainable_sam_model(
        model_type=model_type,
        device="cpu",
        peft_kwargs={
            "rank": rank,
            "update_matrices": update,
            "attention_layers_to_update": lora_layers,
        },
    )

    trainable.load_state_dict(ckpt[model_key], strict=True)
    sam = trainable.sam

    for i, blk in enumerate(sam.image_encoder.blocks):
        if i not in lora_layers:
            continue
        qkv = blk.attn.qkv
        base = qkv.qkv_proj
        W = base.weight.data
        d = base.in_features

        if hasattr(qkv, "w_a_linear_q"):
            W[0:d, :] += (qkv.w_b_linear_q.weight @ qkv.w_a_linear_q.weight) * qkv.alpha
        if hasattr(qkv, "w_a_linear_v"):
            W[2 * d : 3 * d, :] += (qkv.w_b_linear_v.weight @ qkv.w_a_linear_v.weight) * qkv.alpha

        blk.attn.qkv = base

    merged = dict(ckpt)
    merged[model_key] = sam.state_dict()
    torch.save(merged, out_pt)
    return out_pt
