"""
PyTorch model code for TinyVGG model.
"""

# Libraries
import torch
from torch import nn

# Vanilla TinyVGG model
class TinyVGG(nn.Module):
    """
    TinyVGG architecture.
    
    Inspired by the TinyVGG architecture from the CNN explainer website.
        See: https://poloclub.github.io/cnn-explainer/
    
    Args:
        input_shape:  An integer indicating the number of input channels.
        hidden_units: An integer indicating the number of hidden units between layers.
        output_shape: An integer indicating the number of output units
    """
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
                                          nn.Conv2d(in_channels=input_shape, 
                                                    out_channels=hidden_units, 
                                                    kernel_size=3,
                                                    stride=1, 
                                                    padding=0),  
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=hidden_units, 
                                                    out_channels=hidden_units,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2,
                                                       stride=2)
        )
        
        self.conv_block_2 = nn.Sequential(
                                          nn.Conv2d(hidden_units,
                                                    hidden_units,
                                                    kernel_size=3,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.Conv2d(hidden_units,
                                                    hidden_units,
                                                    kernel_size=3,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
                                        nn.Flatten(),
                                        # Fully connected layer (fc), aka dense layer
                                        nn.Linear(in_features=hidden_units*13*13, # each layer of our network compresses and changes the shape of our inputs data.
                                                  out_features=output_shape)
                                        # where each element in the flattened input is connected to each element in the output, forming a fully connected (or densely connected) structure.
        )
    
    def forward(self, x: torch.Tensor):
        #x = self.conv_block_1(x)
        #x = self.conv_block_2(x)
        #x = self.classifier(x)
        #return x
        
        # Using operator fusion (faster)
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))



# TinyVGG model with multiple dropouts, spatialdropout (Dropout2d)
class TinyVGG_MultiDropout(TinyVGG):
    """
    Subclassed TinyVGG model, adding dropout 
    """
    
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int,
                 dropout: float = 0.5) -> None:
        
        super().__init__(input_shape, hidden_units, output_shape)
        
        self.conv_block_1 = nn.Sequential(
                                          nn.Conv2d(in_channels=input_shape, 
                                                    out_channels=hidden_units, 
                                                    kernel_size=3,
                                                    stride=1, 
                                                    padding=0),  
                                          nn.ReLU(),
                                          nn.Dropout2d(p=dropout),
                                          nn.Conv2d(in_channels=hidden_units, 
                                                    out_channels=hidden_units,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2,
                                                       stride=2)
        )
        
        self.conv_block_2 = nn.Sequential(
                                          nn.Conv2d(hidden_units,
                                                    hidden_units,
                                                    kernel_size=3,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.Dropout2d(p=dropout),
                                          nn.Conv2d(hidden_units,
                                                    hidden_units,
                                                    kernel_size=3,
                                                    padding=0),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2)
        )

