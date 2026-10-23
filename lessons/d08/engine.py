import torch

# Evaluate the loss, accuracy of the model
def evaluate(model, images, labels, loss_fn, device):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        labels = labels.to(device)

        pred_logits = model(images)
        pred_labels = pred_logits.argmax(dim = 1)

        pred_correct = (pred_labels == labels).float().sum()
        pred_acc = (pred_labels == labels).float().mean()
        pred_CE_loss = loss_fn(pred_logits, labels).float()

        loss_value = pred_CE_loss.item()
        acc_value = pred_acc.item()

        return loss_value, acc_value

# Training one epoch, load model/loader/optimizer/loss_fn from outer
def train_one_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        device,
        global_step,
        batch_order
):
    model.train()
    for idx, batch_data in enumerate(loader):
        optimizer.zero_grad()

        batch_input, batch_target, ids = batch_data
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        batch_pred = model(batch_input)

        loss = loss_fn(batch_pred, batch_target)
        loss.backward()

        optimizer.step()
        global_step += 1

        batch_order.append(ids.tolist())

    return global_step


        
