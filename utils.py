import random
import torch
import os
import numpy as np
import config


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """Saves the model and optimizer state."""
    print("=> Saving checkpoint")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None):
    """Loads the checkpoint into the model and optionally the optimizer."""
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file '{checkpoint_file}' not found. Skipping checkpoint loading.")
        return
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE, weights_only=True)

    # Load model weights
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if available and optimizer is provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Ensure optimizer uses the correct learning rate
        if lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        print(f"Optimizer learning rate set to {lr}.")

    print("Checkpoint loaded.")


def seed_everything(seed=42):
    """Sets the seed for reproducibility across different libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
