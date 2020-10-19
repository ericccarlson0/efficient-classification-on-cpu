import time
import torch
import sys

def train(model: torch.nn.Module, criterion, optimizer, scheduler, loaders, sizes, writer,
          num_epochs: int = 16,
          log_freq: int = 32,
          device: str = 'cpu'):
    """
    A standard helper method for training a CNN.
    :param log_freq:    the number of batches before logging is triggered
    :param writer:      a tensorboard.SummaryWriter
    :param model:       the model to train
    :param criterion:   the loss criterion to optimize the model to
    :param optimizer:   the strategy used to optimize to the loss criterion
    :param scheduler:   the learning rate scheduler
    :param loaders:     a dict of DataLoader objects (one for 'train', one for 'val')
    :param sizes:       a dict of Dataset sizes (one for 'train', one for 'val')
    :param num_epochs:  the amount of times to iterate through the dataset during training
    :param device:      either CPU or CUDA
    :return:            the trained model
    """
    assert device in ['cpu', 'cuda']

    start = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}\n")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            losses = 0.0
            corrects = 0.0
            batch_count = 0

            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.long())
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                batch_count += 1
                if batch_count % log_freq == 0:
                    print(f"{batch_count} batches with {losses: .4f} losses...")
                    writer.add_scalar("TRAINING LOSS",
                                      losses / log_freq,
                                      epoch * sizes[phase] + batch_count)
                    losses = 0

                losses += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            losses = losses / sizes[phase]
            accuracy = corrects / sizes[phase]

            print(f"{phase.capitalize()} phase:\n")
            print(f"LOSS: {losses: .4f}, ACCURACY: {accuracy * 100: .2f}")

        print("...")

    duration = time.time() - start
    print(f"{num_epochs} epochs were completed in...")
    print(f"{duration // 3600: .0f}H, {duration // 60: .0f}M, {duration % 60: .of}s ")

    return model


def train_epoch(model: torch.nn.Module, criterion, optimizer, loader, writer,
                max_batches: int = sys.maxsize,
                log_freq: int = 32,
                epoch: int = 0):

    model.train()
    count = 0
    losses = 0

    for inputs, labels in loader:
        if count >= max_batches:
            break
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        count += 1
        losses += loss.item()

        if count % log_freq == 0:
            print(f"{count} batches with {losses} losses...")
            writer.add_scalar("TRAINING LOSS", losses / log_freq, epoch * len(loader) + count)
            losses = 0
