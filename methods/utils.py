import torch.nn.functional as F

def task_loss(model, batch, lam):
    price, news, tf, target = batch
    pos, al = model(price, news, tf)
    pred = F.mse_loss(pos, target)
    return pred + lam * al, pred.item(), al.item()

def collate(dataset, indices, batch_size):
    """Collate dataset items into list of (price, news, time, target) batches."""
    batches = []
    import torch
    for i in range(0, len(indices), batch_size):
        chunk = [dataset[j] for j in indices[i:i + batch_size]]
        batches.append(tuple(torch.stack([c[k] for c in chunk]) for k in range(4)))
    return batches
