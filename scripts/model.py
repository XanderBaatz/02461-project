"""
PyTorch model code for TinyVGG model.
"""

# Libraries
import torch
from torch import nn
from dropblock import DropBlock2D

# Vanilla TinyVGG model .test
class TinyVGG(nn.Module):
    """
    Inspired by the TinyVGG architecture from the CNN explainer website.
        See: https://poloclub.github.io/cnn-explainer/
    
    Args:
        input_shape:  An integer indicating the number of input channels.
        hidden_units: An integer indicating the number of hidden units between layers.
        output_shape: An integer indicating the number of output units
    """
    def __init__(self,
                 input_shape: int,
                 output_shape: int,
                 hidden_units: int=10,
                 ) -> None:
        
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
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        
        #print(x.size())
        
        x = self.classifier(x)
        return x



# Custom CNN model, inspired by TinyVGG
class AJ_CNN(nn.Module):
    """
    Custom made CNN inspired by TinyVGG.
    """
    
    def __init__(self,
                 input_shape:int,
                 output_shape:int,
                 hidden_units:int=12,
                 dropout:float=0.,
                 ) -> None:
        
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            # Convolution layer, model hyperparameters
            nn.Conv2d(in_channels=input_shape,     # input shape, RGB, 3
                      out_channels=hidden_units, # Filters
                      kernel_size=(3,3),           # 3x3 pixels
                      stride=1,                    # meaning it moves 1 pixel at a time
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(hidden_units),
            
            # Relu actvation function, simple and computationally efficient
            ## essentially, it reshapes the output, and therefore the function
            nn.ReLU(), # converts any negative values to zero, while positive values remain unchanged
            
            # Max pooling function
            ## returns the max value after the activation function
            ## (2,2) indicates that it'll take the max value out of a 2x2 region
            ## will further condense the info
            nn.MaxPool2d(kernel_size=(2,2)),
            
            # DropBlock function
            #DropBlock(block_size=7, p=dropout)
            #nn.Dropout2d(p=dropout)
        )
        
        self.dropblock = DropBlock2D(drop_prob=dropout, block_size=5)
        
        # x2 the size of the first convolution block
        self.conv_block_2 = nn.Sequential(
            # Convolution layer, model hyperparameters
            nn.Conv2d(in_channels=hidden_units,     
                      out_channels=hidden_units*2, 
                      kernel_size=(3,3),           
                      stride=1,                    
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(hidden_units*2),
            
            # Relu actvation function, simple and computationally efficient
            nn.ReLU(),
            
            # Max pooling function
            nn.MaxPool2d(kernel_size=(2,2)),
            
            # DropBlock function
            #DropBlock(block_size=7, p=dropout)
            #nn.Dropout2d(p=dropout)
        )
        
        self.conv_block_3 = nn.Sequential(
            # Convolution layer, model hyperparameters
            nn.Conv2d(in_channels=hidden_units*2,     
                      out_channels=hidden_units*2, 
                      kernel_size=(3,3),           
                      stride=1,
                      padding=0,   # intentionally disable padding to lose info in hopes of generalization                 
                      bias=False),
            nn.BatchNorm2d(hidden_units*2),
            
            # Relu actvation function, simple and computationally efficient
            nn.ReLU(),
            
            # Max pooling function
            nn.MaxPool2d(kernel_size=(2,2)),
            
            # DropBlock function
            #DropBlock(block_size=7, p=dropout)
            #nn.Dropout2d(p=dropout)
        )
        
        self.classifier = nn.Sequential(
                                        nn.Flatten(),
                                        nn.Dropout(p=dropout),
                                        # Fully connected layers (fc), aka dense layer
                                        #nn.Linear(in_features=256, out_features=128),
                                        #nn.Linear(in_features=128, out_features=64),
                                        nn.Linear(in_features=hidden_units*2*7*7, # 64
                                                  out_features=output_shape)
        )
        
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.dropblock(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        #print(x.size())
        x = self.classifier(x)
        return x
