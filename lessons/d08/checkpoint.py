import torch
import torch.nn as nn

from pathlib import Path

def save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        generator,
        completed_epochs,
        global_step,
        history,
        batch_order,
        config
):
    checkpoint = {
        "model_state" : model.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "completed_epochs" : completed_epochs,
        "global_step" : global_step,
        "torch_rng_state" : torch.get_rng_state(),
        "cuda_rng_state_all" : torch.cuda.get_rng_state_all(),
        "loader_rng_state" : generator.get_state(),
        "history" : history,
        "batch_order" : batch_order,
        "config" : config
    }

    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, generator):
    checkpoint = torch.load(
        checkpoint_path,
        map_location = "cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    generator.set_state(checkpoint["loader_rng_state"])
    completed_epochs = checkpoint["completed_epochs"]
    global_step = checkpoint["global_step"]
    history = checkpoint["history"]
    batch_order = checkpoint["batch_order"]

    return completed_epochs, global_step, history, batch_order
