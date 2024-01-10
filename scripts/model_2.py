import torch
from torch import nn

class TinyVGG(nn.Module):
    """
    TinyVGG architecture.
    
    Inspired by the TinyVGG architecture from the CNN explainer website.
        See: https://poloclub.github.io/cnn-explainer/
    
    Args:
        input_shape:  An integer indicating the number of input channels
        hidden_units_1: An integer indicating the number of hidden units in block 1
        hidden_units_2: An integer indicating the number of hidden units in block 2
        output_shape: An integer indicating the number of output units
    """

    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_units_1: int,
                 hidden_units_2: int,
                 ) -> None:

        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units_1,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units_1,
                      out_channels=hidden_units_1,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_1,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units_2,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Assuming input images of size 32x32
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units_2 * 13 * 13,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        # Using operator fusion
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

class TinyVGGDropout(nn.Module):
    """
    TinyVGG architecture.
    
    Inspired by the TinyVGG architecture from the CNN explainer website.
        See: https://poloclub.github.io/cnn-explainer/
    
    Args:
        input_shape:  An integer indicating the number of input channels
        hidden_units_1: An integer indicating the number of hidden units in block 1
        hidden_units_2: An integer indicating the number of hidden units in block 2
        output_shape: An integer indicating the number of output units
    """

    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_units_1: int,
                 hidden_units_2: int,
                 ) -> None:

        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units_1,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=hidden_units_1,
                      out_channels=hidden_units_1,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),   
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_1,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=hidden_units_2,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5),
        )

        # Assuming input images of size 32x32
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_units_2 * 13 * 13,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        # Using operator fusion
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
