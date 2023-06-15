import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Decoder(object):
    def __init__(self, labels: list[str], blank_idx: str) -> None:
        self.label_2_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_2_label = {idx: label for label, idx in self.label_2_idx.items()}
        self.blank_idx = blank_idx

    def decode_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[list[str], list[str]]:
        y_pred, y_true = [], []
        for sample in list(zip(*batch)):
            x_decoded, y_decoded = self.decode_sample(sample)
            y_pred.append(x_decoded)
            y_true.append(y_decoded)
        return y_pred, y_true

    def decode_sample(
        self,
        sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[str, str]:
        x, y, x_length, y_length = sample
        x, y = x[:x_length], y[:y_length]
        x = torch.unique_consecutive(x)
        return self.decode_sequence(x), self.decode_sequence(y)

    def decode_sequence(self, sequence: torch.Tensor) -> str:
        sequence_symbols = []
        for label_idx in sequence:
            if label_idx.item() == self.blank_idx:
                continue
            label = self.idx_2_label[label_idx.item()]
            sequence_symbols.append(label)
        return ''.join(sequence_symbols)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    mode: str,
    num_workers: int,
) -> DataLoader:
    shuffle, drop_last = True, True
    if mode == 'val':
        shuffle, drop_last = False, False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=num_workers,
    )


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    dataset: Dataset,
    decoder: Decoder,
    device: torch.device,
) -> tuple[np.ndarray, str, str]:
    model.eval()

    sample_idx = random.randint(0, len(dataset) - 1)
    sample = dataset[sample_idx]
    image, label, x_length, y_length = sample

    x, y = torch.Tensor(image).unsqueeze(0).to(device), torch.Tensor(label)
    x_length, y_length = torch.LongTensor([x_length]), torch.LongTensor([y_length])

    output = torch.nn.functional.log_softmax(model(x), dim=2)
    prediction = output.argmax(dim=2)

    y_pred, y_true = decoder.decode_sample(
        (prediction.squeeze(), y, x_length, y_length),
    )

    return image.squeeze(), y_pred, y_true

