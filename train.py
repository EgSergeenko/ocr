import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils import Decoder


def train_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    decoder: Decoder,
    device: torch.device,
) -> tuple[float, list[str], list[str]]:
    optimizer.zero_grad()

    x, targets, x_lengths, target_lengths = batch
    x, targets = x.to(device), targets.to(device)

    output = torch.nn.functional.log_softmax(model(x), dim=2)

    loss = criterion(
        output.permute(1, 0, 2),
        targets,
        x_lengths,
        target_lengths,
    )

    predictions = output.argmax(dim=2)

    y_pred, y_true = decoder.decode_batch(
        (predictions, targets, x_lengths, target_lengths),
    )

    loss.backward()
    optimizer.step()

    return loss.item(), y_pred, y_true


def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    decoder: Decoder,
    device: torch.device,
) -> tuple[float, list[str], list[str]]:
    model.train()
    epoch_loss = 0
    y_pred, y_true = [], []
    for batch in dataloader:
        step_loss, y_pred_batch, y_true_batch = train_step(
            model=model,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            decoder=decoder,
            device=device,
        )
        epoch_loss += step_loss
        y_pred.extend(y_pred_batch)
        y_true.extend(y_true_batch)
    return epoch_loss / len(dataloader), y_pred, y_true


@torch.no_grad()
def eval_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    criterion: torch.nn.Module,
    decoder: Decoder,
    device: torch.device,
):
    x, targets, x_lengths, target_lengths = batch
    x, targets = x.to(device), targets.to(device)

    output = torch.nn.functional.log_softmax(model(x), dim=2)

    loss = criterion(
        output.permute(1, 0, 2),
        targets,
        x_lengths,
        target_lengths,
    )

    predictions = output.argmax(dim=2)

    y_pred, y_true = decoder.decode_batch(
        (predictions, targets, x_lengths, target_lengths),
    )

    return loss.item(), y_pred, y_true


def eval_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    decoder: Decoder,
    device: torch.device,
) -> tuple[float, list[str], list[str]]:
    model.eval()
    epoch_loss = 0
    y_pred, y_true = [], []
    for batch in dataloader:
        step_loss, y_pred_batch, y_true_batch = eval_step(
            model=model,
            batch=batch,
            criterion=criterion,
            decoder=decoder,
            device=device,
        )
        epoch_loss += step_loss
        y_pred.extend(y_pred_batch)
        y_true.extend(y_true_batch)
    return epoch_loss / len(dataloader), y_pred, y_true
