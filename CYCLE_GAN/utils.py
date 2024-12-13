import os
import torch

def save_checkpoint(model, optimizer, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer=None, lr=None):
    checkpoint = torch.load(filename, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
    print(f"Checkpoint loaded from {filename}")
