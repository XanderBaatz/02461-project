"""
PyTorch model code for TinyVGG model.
"""

# Libraries
import torch
from torch import nn
from lib.dropblock import DropBlock
from torchinfo import summary

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
                                        # where each element in the flattened input is connected to each element in the output, forming a fully connected (or densely connected) structure.
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        
        #print(x.size())
        
        x = self.classifier(x)
        return x
        
        # Using operator fusion (faster)
        #return self.classifier(self.conv_block_2(self.conv_block_1(x)))



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



# Custom CNN model
class AJ_CNN(nn.Module):
    """
    Custom made CNN inspired by TinyVGG.
    """
    
    def __init__(self,
                 input_shape:int,
                 output_shape:int,
                 hidden_units_1:int=16,
                 hidden_units_2:int=32
                 ) -> None:
        
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            # Convolution layer, model hyperparameters
            nn.Conv2d(in_channels=input_shape,     # input shape, RGB, 3
                      out_channels=hidden_units_1, # Filters
                      kernel_size=(3,3),           # 3x3 pixels
                      stride=1,                    # meaning it moves 1 pixel at a time
                      ),
            
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
        
        # x2 the size of the first convolution block
        self.conv_block_2 = nn.Sequential(
            # Convolution layer, model hyperparameters
            nn.Conv2d(in_channels=hidden_units_1,     
                      out_channels=hidden_units_2, 
                      kernel_size=(3,3),           
                      stride=1,                    
                      ),
            
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
            nn.Conv2d(in_channels=hidden_units_2,     
                      out_channels=hidden_units_2, 
                      kernel_size=(3,3),           
                      stride=1,                    
                      ),
            
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
                                        # Fully connected layers (fc), aka dense layer
                                        #nn.Linear(in_features=256, out_features=128),
                                        #nn.Linear(in_features=128, out_features=64),
                                        nn.Linear(in_features=hidden_units_2*6*6, # 64
                                                  out_features=output_shape)
        )
        
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        #print(x.size())
        x = self.classifier(x)
        return x
    
    
    
# Custom CNN with dropout between each block
## Test accuracy, cifar-10, 100 epochs, no data aug: 73.9%
class AJ_CNN_DropoutEnd(AJ_CNN):
    def __init__(self,
                input_shape:int,
                output_shape:int,
                hidden_units_1:int=16,
                hidden_units_2:int=32,
                dropout:float=0.2
                ) -> None:
        
        super().__init__(input_shape, output_shape, hidden_units_1, hidden_units_2)
        
        # Spatial dropout layer
        self.dropout_layer = nn.Dropout(p=dropout)
        
        # Add dropout layer to each convolution block
        self.conv_block_1.add_module('dropout_layer', self.dropout_layer)
        self.conv_block_2.add_module('dropout_layer', self.dropout_layer)
        self.conv_block_3.add_module('dropout_layer', self.dropout_layer)



# Custom CNN with spatial dropout after activation function in each convolution block
## Worse: 70% accuracy, longer training time
class AJ_CNN_DropoutMid(AJ_CNN):
    def __init__(self,
                input_shape:int,
                output_shape:int,
                hidden_units_1:int=16,
                hidden_units_2:int=32,
                dropout:float=0.1
                ) -> None:
        
        super().__init__(input_shape, output_shape, hidden_units_1, hidden_units_2)
        
        # Spatial dropout layer
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,     
                      out_channels=hidden_units_1, 
                      kernel_size=(3,3),           
                      stride=1,                    
                      ),
            nn.ReLU(),
            nn.Dropout2d(p=dropout), # Spatial dropout layer
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_1,     
                      out_channels=hidden_units_2, 
                      kernel_size=(3,3),           
                      stride=1,                    
                      ),
            nn.ReLU(),
            nn.Dropout2d(p=dropout), # Spatial dropout layer
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_2,     
                      out_channels=hidden_units_2, 
                      kernel_size=(3,3),           
                      stride=1,                    
                      ),
            nn.ReLU(),
            nn.Dropout2d(p=dropout), # Spatial dropout layer
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        


# Custom CNN with dropblock tech between each block
## Test accuracy, cifar-10, 100 epochs, no data aug: 73.9%
class AJ_CNN_DropBlockEnd(AJ_CNN):
    def __init__(self,
                input_shape:int,
                output_shape:int,
                hidden_units_1:int=16,
                hidden_units_2:int=32,
                dropout:float=0.2
                ) -> None:
        
        super().__init__(input_shape, output_shape, hidden_units_1, hidden_units_2)
        
        # Spatial dropout layer
        self.dropout_layer = DropBlock(block_size=3, p=dropout)
        
        # Add dropout layer to each convolution block
        self.conv_block_1.add_module('dropout_layer', self.dropout_layer)
        self.conv_block_2.add_module('dropout_layer', self.dropout_layer)
        self.conv_block_3.add_module('dropout_layer', self.dropout_layer)
        


# Custom CNN with DropBlock after activation function in each convolution block
## Worse: 70% accuracy, longer training time
class AJ_CNN_DropBlockMid(AJ_CNN):
    def __init__(self,
                input_shape:int,
                output_shape:int,
                hidden_units_1:int=16,
                hidden_units_2:int=32,
                dropout:float=0.9,
                dbbs:int=7,
                ) -> None:
        
        super().__init__(input_shape, output_shape, hidden_units_1, hidden_units_2)
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,     
                      out_channels=hidden_units_1, 
                      kernel_size=(3,3),           
                      stride=1,                    
                      ),
            nn.ReLU(),
            DropBlock(block_size=dbbs, p=dropout),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_1,     
                      out_channels=hidden_units_2, 
                      kernel_size=(3,3),           
                      stride=1,                    
                      ),
            nn.ReLU(),
            DropBlock(block_size=dbbs, p=dropout),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_2,     
                      out_channels=hidden_units_2, 
                      kernel_size=(3,3),           
                      stride=1,                    
                      ),
            nn.ReLU(),
            DropBlock(block_size=dbbs, p=dropout),
            nn.MaxPool2d(kernel_size=(2,2)),
        )



# Custom CNN with dropout in classifier and batch normalizations
class AJ_CNN_BNN_Dropout_Giga(AJ_CNN):
    def __init__(self,
                input_shape:int,
                output_shape:int,
                hidden_units_1:int=16,
                hidden_units_2:int=32,
                dropout:float=0.2
                ) -> None:
        
        super().__init__(input_shape, output_shape, hidden_units_1, hidden_units_2)
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,     
                      out_channels=hidden_units_1, 
                      kernel_size=(3,3),           
                      stride=1,
                      padding=1                    
                      ),
            nn.BatchNorm2d(num_features=hidden_units_1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        ) # output: ([1, 16, 32, 32])
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_1,     
                      out_channels=hidden_units_2, 
                      kernel_size=(3,3),           
                      stride=1,
                      padding=1                    
                      ),
            nn.BatchNorm2d(num_features=hidden_units_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        ) # output: ([1, 32, 16, 16])
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units_2,     
                      out_channels=hidden_units_2, 
                      kernel_size=(3,3),           
                      stride=1,
                      padding=1                    
                      ),
            nn.BatchNorm2d(num_features=hidden_units_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        ) # output: ([1, 32, 8, 8])
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_units_2*8*8,
                        out_features=hidden_units_2*8*2),
            nn.BatchNorm1d(num_features=hidden_units_2*8*2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units_2*8*2,
                        out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        # Operator fusion (faster)
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))



# Custom CNN with dropout in classifier and batch normalizations
class AJ_CNN_BNorm(nn.Module):
    def __init__(self,
                input_shape:int,
                output_shape:int,
                hidden_units:int=10,
                dropout:float=0.2
                ) -> None:
        
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,     
                      out_channels=hidden_units, 
                      kernel_size=(3,3),           
                      stride=1,
                      padding=1                    
                      ),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,     
                      out_channels=hidden_units*2, 
                      kernel_size=(3,3),           
                      stride=1,
                      padding=1                    
                      ),
            nn.BatchNorm2d(num_features=hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*2,     
                      out_channels=hidden_units*2, 
                      kernel_size=(3,3),           
                      stride=1,
                      padding=1                    
                      ),
            nn.BatchNorm2d(num_features=hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_units*2*8*8,
                        out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        #x = self.conv_block_1(x)
        #print(x.size())
        #x = self.conv_block_2(x)
        #print(x.size())
        #x = self.conv_block_3(x)
        #print(x.size())
        #x = self.classifier(x)
        #return x

        # Operator fusion (faster)
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))



if __name__ == "__main__":
    model = AJ_CNN_BNorm(
                            input_shape=3,
                            output_shape=10,
                            )
    
    print(summary(model=model,
                      input_size=(1, 3, 64, 64),
                      col_names=["input_size", "output_size", "num_params", "trainable"],
                      col_width=2,
                      row_settings=["var_names"]))
    print("19,920")