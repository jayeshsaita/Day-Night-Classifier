import torch


def accuracy(out, y):
    """
    out: shape (batch_size x 2) logits
    y: shape (batch_size) class indices
    """
    bs = out.shape[0]
    out = torch.softmax(out, dim=1).argmax(dim=1).view(bs) # Applying softmax over prediction logits
    y = y.view(bs)
    corrects = (out == y).sum().float()
    acc = corrects/bs
    return acc


# Simple function to perform one Training epoch
def train_epoch(model, dl, criterion, optimizer, scheduler, device='cuda:0'):
    model.train() # putting model in training mode
    losses = [] # tracking running losses
    accuracies = [] # training running accuracy

    for x,y in dl:
        # Transferring batch to GPU
        x = x.to(device=device)
        y = y.to(device=device)

        optimizer.zero_grad()

        out = model(x)
        bs_acc = accuracy(out, y) # Train accuracy for current batch
        loss = criterion(out, y) # Train loss for current batch
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item() * x.shape[0]) # running loss
        accuracies.append(bs_acc * x.shape[0]) # running accuracy

    epoch_loss = sum(losses) / len(dl.dataset) # epoch loss
    epoch_acc = sum(accuracies) / len(dl.dataset) # epoch accuracy
    return epoch_loss, epoch_acc


# Simple function to perform one Validation epoch
def valid_epoch(model, dl, criterion, device='cuda:0'):
    model.eval() # putting model in evaluation mode
    losses = [] # tracking running losses
    accuracies = [] # tracking running accuracy

    for x,y in dl:
        # Transferring batch to GPU
        x = x.to(device=device)
        y = y.to(device=device)

        # Using torch.no_grad() to prevent calculation of gradients hence saving memory as gradients are not required during validation phase
        with torch.no_grad():
            out = model(x)
            bs_acc = accuracy(out, y) # Validation accuracy for current batch
            loss = criterion(out, y) # Validation loss for current batch

        losses.append(loss.item() * x.shape[0]) # running loss
        accuracies.append(bs_acc * x.shape[0]) # running accuracy

    epoch_loss = sum(losses) / len(dl.dataset) # epoch loss
    epoch_acc = sum(accuracies) / len(dl.dataset) # epoch accuracy
    return epoch_loss, epoch_acc