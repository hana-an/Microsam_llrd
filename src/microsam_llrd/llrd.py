def _freeze_block_params(block):
    for p in block.parameters():
        p.requires_grad = False


def _collect_block_lora_params(block):
    params = []
    attn = getattr(block, "attn", None)
    qkv = getattr(attn, "qkv", None) if attn is not None else None
    if qkv is None:
        return params

    for attr in ("w_a_linear_q", "w_b_linear_q", "w_a_linear_v", "w_b_linear_v"):
        m = getattr(qkv, attr, None)
        if m is not None:
            params.extend([p for p in m.parameters() if p.requires_grad])
    return params


def build_llrd_groups_lora_only(model, base_lr: float, layer_decay: float, start_block: int, lr_other: float):
    """
    Param groups:
      - blocks < start_block: frozen
      - LoRA params in blocks >= start_block: LLRD
      - everything else trainable: lr_other
    """
    assert hasattr(model.sam.image_encoder, "blocks"), "Expected ViT-style encoder with .blocks"
    blocks = model.sam.image_encoder.blocks
    n_blocks = len(blocks)

    for i in range(0, min(start_block, n_blocks)):
        _freeze_block_params(blocks[i])

    param_groups = []
    lora_ids = set()

    for i in range(start_block, n_blocks):
        lr_i = base_lr * (layer_decay ** (n_blocks - i - 1))
        lora_params = _collect_block_lora_params(blocks[i])
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lr_i})
            for p in lora_params:
                lora_ids.add(id(p))

    other = [p for p in model.parameters() if p.requires_grad and id(p) not in lora_ids]
    if other:
        param_groups.append({"params": other, "lr": lr_other})

    if not param_groups:
        raise RuntimeError("No optimizer param groups built. LoRA not found / not trainable.")

    return param_groups
