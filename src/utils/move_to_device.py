import torch

def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(x, device) for x in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(x, device) for x in batch)
    else:
        return batch