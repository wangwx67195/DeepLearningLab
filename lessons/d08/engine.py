import torch

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

        return (pred_CE_loss, pred_acc)

def train_one_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        device,
        global_step,
        batch_order
):
    for idx, batch_data in enumerate(loader):
        model.to(device)
        optimizer.zero_grad()

        batch_input, batch_target, ids = batch_data
        batch_pred = model(batch_input)

        loss = loss_fn(batch_pred, batch_target)
        loss.backward()

        optimizer.step()
        global_step += 1

        batch_order.append(ids.tolist())

        return global_step


        
