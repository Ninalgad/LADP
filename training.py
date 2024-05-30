import gc
import torch
from evaluation import evaluate
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from data import ImageDataset, DenosingDataset
from augment import get_transform
from loss import loss_func


def train_denoise(model, device, output_name, training_ids, data_dir, noising_transform=None,
                  inp_size=256, num_epochs=60, batch_size=16, lr=1e-4,
                  debug=False):
    if debug:
        batch_size = 2
        training_ids = training_ids[:2]

    train_loader = DataLoader(
        DenosingDataset(training_ids, data_dir, inp_size,
                        transform=get_transform(inp_size),
                        noising_transform=noising_transform),
        batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=lr)

    def denoise_train_step(batch):
        inp, tar = batch['image'].to(device).float(), batch['label'].to(device).float()
        optimizer.zero_grad()

        outputs = model(inp)

        mask = (tar > -6).float()
        loss_ = (tar - outputs) ** 2
        loss_ = torch.sum(loss_ * mask) / (torch.sum(mask) + 1)

        loss_.backward()
        optimizer.step()
        return loss_.detach().cpu().numpy()

    if debug:
        num_epochs = 1

    for epoch in range(num_epochs):

        train_epoch(model, train_loader, denoise_train_step, debug)

        torch.save({
            'model_state_dict': model.encoder.state_dict()
        }, f'{output_name}-pt.pt')


def train_finetune(model, device, output_name, training_ids, validation_ids, data_dir,
                   inp_size=256, num_epochs=60, batch_size=16, lr=1e-4, debug=False):
    if debug:
        batch_size = 2
        training_ids = training_ids[:2]
        validation_ids = validation_ids[:2]

    train_loader = DataLoader(ImageDataset(training_ids, data_dir, inp_size,
                                           transform=get_transform(inp_size)),
                              batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=lr)

    def train_step(batch):
        inp, tar = batch['image'].to(device), batch['label'].to(device).float()
        optimizer.zero_grad()

        outputs = model(inp)
        loss_ = loss_func(outputs, tar)

        loss_.backward()
        optimizer.step()
        return loss_.detach().cpu().numpy()

    best_val = -1

    if debug:
        num_epochs = 1

    for epoch in range(num_epochs):

        train_epoch(model, train_loader, train_step, debug)

        gc.collect()
        with torch.no_grad():
            val, thresh = evaluate(model, validation_ids, data_dir, inp_size, device)

        if val > best_val:
            best_val = val

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val': val,
                'best_thresh': thresh,
            }, f'{output_name}.pt')

    return best_val


def train_epoch(model, train_loader, train_step_fn, debug=False):
    model.train()
    train_loss = []

    for batch in tqdm(train_loader, total=len(train_loader)):
        loss = train_step_fn(batch)

        train_loss.append(loss)
        if debug:
            break
    return sum(train_loss)/len(train_loss)
