import torch


class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # conv layers
        self.conv_1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv_3 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv_4 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv_5 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv_6 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv_7 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(2, 2),
            stride=1,
            padding=0,
        )

        # pooling types
        self.pooling_1 = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=2,
        )
        self.pooling_2 = torch.nn.MaxPool2d(
            kernel_size=(2, 1), stride=(2, 1),
        )

        # batch norm
        self.batch_norm = torch.nn.BatchNorm2d(num_features=512)

        # activation
        self.relu = torch.nn.ReLU()

        # whole model
        self.model = torch.nn.Sequential(
            self.conv_1,
            self.pooling_1,
            self.relu,
            self.conv_2,
            self.pooling_1,
            self.relu,
            self.conv_3,
            self.relu,
            self.conv_4,
            self.pooling_2,
            self.relu,
            self.conv_5,
            self.batch_norm,
            self.relu,
            self.conv_6,
            self.batch_norm,
            self.pooling_2,
            self.relu,
            self.conv_7,
            self.relu,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze().permute(0, 2, 1)


class RNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = torch.nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[0]


class CRNN(torch.nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()

        self.cnn = CNN()

        self.map_to_sequence = torch.nn.Linear(
            in_features=512, out_features=256,
        )

        self.rnn = RNN()

        fc_in_features = self.rnn.model.hidden_size
        if self.rnn.model.bidirectional:
            fc_in_features *= 2
        self.fc = torch.nn.Linear(
            in_features=fc_in_features,
            out_features=n_classes,
        )

        self.model = torch.nn.Sequential(
            self.cnn, self.map_to_sequence, self.rnn, self.fc,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == '__main__':
    n_classes = 10
    model = CRNN(n_classes=n_classes)
    batch = torch.zeros((20, 1, 32, 100))  # batch x channels x 32 x width
    output = model(batch)  # batch x frames x n_classes
