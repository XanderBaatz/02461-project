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
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        #x = self.conv_block_1(x)
        #x = self.conv_block_2(x)
        #x = self.classifier(x)
        #return x
        
        # Using operator fusion (faster)
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

class TinyVGG_test(nn.Module):
    """
    TinyVGG architecture.
    
    Inspired by the TinyVGG architecture from the CNN explainer website.
        See: https://poloclub.github.io/cnn-explainer/
    
    Args:
        input_shape:  An integer indicating the number of input channels.
        hidden_units: An integer indicating the number of hidden units between layers.
        output_shape: An integer indicating the number of output units
    
    Args (exclusive to convolution layers):
        padding:      An integer to help adjust when the kernel extends beyond the activation map. 
        kernel_size:  An integer indicating the dimensions of the sliding window over the input.
        stride:       An integer indicating the number of pixels the kernel should be shifted over at a time.
    """
    
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_units_1: int, # Hidden units for block 1
                 hidden_units_2: int, # Hidden units for block 2
                 ) -> None:
        
        # Inherit all the methods and properties from the parent class (nn.Module)
        super().__init__() 
        
        self.convolutional_block_1 = nn.Sequential(
            # conv_1_1
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units_1,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            
            # relu_1_1
            nn.ReLU(),
            
            # conv_1_2
            nn.Conv2d(in_channels=hidden_units_1,
                      out_channels=hidden_units_1,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            
            # relu_1_2
            nn.ReLU(),
            
            # max_pool_1
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        
        self.convolutional_block_2 = nn.Sequential(
            # conv_2_1
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            
            # relu_2_1
            nn.ReLU(),
            
            # conv_2_1
            nn.Conv2d(in_channels=hidden_units_2,
                      out_channels=hidden_units_2,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            
            # relu_2_2
            nn.ReLU(),
            
            # max_pool_1
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            # Flatten layer: converts a three-dimensional layer in the network into a one-dimensional
            # vector to fit the input of a fully-connected layer for classification.
            nn.Flatten(),
            
            # Handles input and output shapes
            nn.Linear(in_features=hidden_units_2*13*13,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        #x = self.convolutional_block_1(x)
        #x = self.convolutional_block_2(x)
        #x = self.classifier(x)
        #return x
        
        # Using operator fusion
        return self.classifier(
                               self.convolutional_block_2(
                                                          self.convolutional_block_1(x))
               )

class TinyVGG_test_2(nn.Module):
    """
    TinyVGG architecture.
    
    Inspired by the TinyVGG architecture from the CNN explainer website.
        See: https://poloclub.github.io/cnn-explainer/
    
    Args:
        input_shape:  An integer indicating the number of input channels.
        hidden_units: An integer indicating the number of hidden units between layers.
        output_shape: An integer indicating the number of output units
    
    Args (exclusive to convolution layers):
        padding:      An integer to help adjust when the kernel extends beyond the activation map. 
        kernel_size:  An integer indicating the dimensions of the sliding window over the input.
        stride:       An integer indicating the number of pixels the kernel should be shifted over at a time.
    """
    
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_units_1: int, # Hidden units for block 1
                 hidden_units_2: int, # Hidden units for block 2
                 padding1:int=0,      # Padding for block 1
                 padding2:int=0,      # Padding for block 2
                 ksize1:int=3,        # Kernel size for block 1
                 ksize2:int=3,        # Kernel size for block 2
                 stride1:int=1,       # Stride for block 1
                 stride2:int=1        # Stride for block 2
                 ) -> None:
        
        # Inherit all the methods and properties from the parent class (nn.Module)
        super().__init__() 
        
        self.convolutional_block_1 = nn.Sequential(
            # conv_1_1
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units_1,
                      kernel_size=ksize1,
                      stride=stride1,
                      padding=padding1),
            
            # relu_1_1
            nn.ReLU(),
            
            # conv_1_2
            nn.Conv2d(in_channels=hidden_units_1,
                      out_channels=hidden_units_1,
                      kernel_size=ksize1,
                      stride=stride1,
                      padding=padding1),
            
            # relu_1_2
            nn.ReLU(),
            
            # max_pool_1
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        
        self.convolutional_block_2 = nn.Sequential(
            # conv_2_1
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units_2,
                      kernel_size=ksize2,
                      stride=stride2,
                      padding=padding2),
            
            # relu_2_1
            nn.ReLU(),
            
            # conv_2_1
            nn.Conv2d(in_channels=hidden_units_2,
                      out_channels=hidden_units_2,
                      kernel_size=ksize2,
                      stride=stride2,
                      padding=padding2),
            
            # relu_2_2
            nn.ReLU(),
            
            # max_pool_1
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            # Flatten layer: converts a three-dimensional layer in the network into a one-dimensional
            # vector to fit the input of a fully-connected layer for classification.
            nn.Flatten(),
            
            # Handles input and output shapes
            nn.Linear(in_features=hidden_units_2*13*13,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.convolutional_block_1(x)
        x = self.convolutional_block_2(x)
        x = self.classifier(x)
        return x
        
        # Using operator fusion
        return self.classifier(
                               self.convolutional_block_2(
                                                          self.convolutional_block_1(x))
               )