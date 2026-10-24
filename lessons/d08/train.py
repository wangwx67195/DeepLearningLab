import torch
import json

# Import configs
from config import (
    MODE,
    DATA_PATH,
    OUTPUT_PATH,
    JSON_PATH,
    DATA_SEED,
    MODEL_SEED,
    LOADER_SEED,
    TOTAL_EPOCHS,
    INTERRUPT_EPOCHS
) 
if not (MODE == "baseline" or MODE == "interrupt"):
    from config import CKPT_PATH
from config import (
    BATCH_SIZE,
    WEIGHT_DECAY,
    LEARNING_RATE
)

from model import TinyResNet
from loss import create_loss_fn
from data import generate_data, load_data, create_train_loader
from engine import train_one_epoch, evaluate
from checkpoint import save_checkpoint, load_checkpoint

device = "cuda"

def main():
    # Create dataset if MODE is baseline or interrupt
    if MODE == "baseline":
        torch.manual_seed(DATA_SEED)
        data = generate_data() # Data is generated
        torch.save(data, DATA_PATH) # Save the generated data
    elif MODE == "interrupt":
        torch.manual_seed(DATA_SEED)
        data = generate_data() # Data is generated
        torch.save(data, DATA_PATH) # Save the generated data
    else:
        data = load_data(DATA_PATH) # No need to generate data, load from existing data

    # Create Model, use model_seed
    torch.manual_seed(MODEL_SEED)
    model = TinyResNet().to(device)
    loss_fn = create_loss_fn() # Create loss function

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    ) 

    # Create generator
    generator = torch.Generator(device = "cpu")
    generator.manual_seed(LOADER_SEED)
    # Create dataloader, use generator and loader_seed
    train_loader = create_train_loader(data, generator, BATCH_SIZE)

    completed_epochs, global_step, history, batch_order = 0, 0, [], [] # Mode: baseline & interrupt initial states

    # Create dataset, use data_seed
    if MODE == "baseline": # Mode baseline: Training from scratch
        target_epochs = TOTAL_EPOCHS
        
    elif MODE == "interrupt": # Mode interrupt: Training from baseline's checkpoint
        if not DATA_PATH.exists():
            raise FileNotFoundError(DATA_PATH)
        target_epochs = INTERRUPT_EPOCHS

    elif MODE == "resume": # Mode resume: Training from interrupt's checkpoint
        if not DATA_PATH.exists():
            raise FileNotFoundError(DATA_PATH)
        target_epochs = TOTAL_EPOCHS
        completed_epochs, global_step, history, batch_order = load_checkpoint(CKPT_PATH, model, optimizer, generator) # Load interrupt's checkpoint

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    # Training loops, train and save
    for epoch_index in range(completed_epochs, target_epochs):
        global_step = train_one_epoch(model, train_loader, optimizer, loss_fn, device, global_step, batch_order)

        train_loss_value, train_acc_value = evaluate(model, data["train_data"], data["train_target"], loss_fn, device)
        val_loss_value, val_acc_value = evaluate(model, data["val_data"], data["val_target"], loss_fn, device)

        completed_epochs = epoch_index + 1
        # assert global_step == completed_epochs * len(train_loader)

        record = {
            "epochs": epoch_index,
            "completed_epochs": completed_epochs,
            "global_step": global_step,
            "train_loss": train_loss_value,
            "train_acc": train_acc_value,
            "val_loss": val_loss_value,
            "val_acc": val_acc_value
        }
        history.append(record)
        assert len(history) == completed_epochs
        assert len(batch_order) == global_step
        with open(JSON_PATH, "w", encoding = "utf-8") as fp:
            json.dump(history, fp, indent = 2, ensure_ascii = False)

    config = {
        "model_seed": MODEL_SEED,
        "loader_seed": LOADER_SEED,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        # "batch_size": train_loader.batch_size,
        "mode": MODE
    } # Create a config dict

    print("MODE:", MODE)
    print("completed_epochs:", completed_epochs)
    print("global_step:", global_step)
    print("history length:", len(history))
    print("batch_order length:", len(batch_order))
    

    save_checkpoint(OUTPUT_PATH, model, optimizer, generator, completed_epochs, global_step, history, batch_order, config) # save checkpoint file

if __name__ == "__main__":
    main()