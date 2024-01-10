"""
Trains a PyTorch image classification model using device-agnostic code.
"""

# Libraries
import os
import argparse
import torch
from torchvision import transforms
import data_setup, engine#, model, utils
import utils
import model_builder
import model_2
import numpy as np

#################
### ARGUMENTS ###
#################

# Parser
parser = argparse.ArgumentParser(description="Hyperparameters and other properties.")

# Arg: model
parser.add_argument("--model",
                    default="TinyVGG",
                    type=str,
                    help="the model to train")

# Arg: model
parser.add_argument("--mn_append",
                    default="std",
                    type=str,
                    help="append to model name output")



### HYPERPARAMETERS FOR NETWORK STRUCTURE

# Arg: hidden units for block 1
parser.add_argument("--hidden_units_1",
                    default=10,
                    type=int,
                    help="number of hidden units in hidden layers for block 1")

# Arg: hidden units for block 2
parser.add_argument("--hidden_units_2",
                    default=10,
                    type=int,
                    help="number of hidden units in hidden layers for block 2")

parser.add_argument('--padding1', type=int, default=0, help='Padding for block 1')
parser.add_argument('--padding2', type=int, default=0, help='Padding for block 2')
parser.add_argument('--ksize1', type=int, default=3, help='Kernel size for block 1')
parser.add_argument('--ksize2', type=int, default=3, help='Kernel size for block 2')
parser.add_argument('--stride1', type=int, default=1, help='Stride for block 1')
parser.add_argument('--stride2', type=int, default=1, help='Stride for block 2')


### HYPERPARAMETERS FOR NETWORK TRAINING

# Arg: number of epochs
parser.add_argument("--num_epochs",
                    default=10,
                    type=int,
                    help="the number of epochs to train for")

# Arg: batch_size
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="number of samples per batch")

# Arg: learning_rate
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning rate to use for model")

# Arg: Gradient algorithm momentum
#parser.add_argument("--momentum",
#                    default=0.001,
#                    type=float,
#                    help="learning rate to use for model")



### DATA DIRECTORIES

# Arg: training directory
parser.add_argument("--train_dir",
                    default="data/pizza_steak_sushi/train",
                    type=str,
                    help="directory file path to training data in standard image classification format")

# Arg: test directory
parser.add_argument("--test_dir",
                    default="data/pizza_steak_sushi/test",
                    type=str,
                    help="directory file path to testing data in standard image classification format")



### DATA TRANSFORMS

# Arg: training data transforms 
parser.add_argument("--train_transform",
                    default=transforms.Compose([
                                                transforms.Resize((64, 64)),
                                                transforms.ToTensor()
                                                ]),
                    type=transforms.Compose,
                    help="training data transforms")

# Arg: testing data transforms 
parser.add_argument("--test_transform",
                    default=transforms.Compose([
                                                transforms.Resize((64, 64)),
                                                transforms.ToTensor()
                                                ]),
                    type=transforms.Compose,
                    help="testing data transforms")

#################

# Derive arguments from the parser
args = parser.parse_args()

# Setup model
model_name = args.model
mn_append = args.mn_append

# Setup hyperparameters for network structure
HIDDEN_UNITS_1 = args.hidden_units_1
HIDDEN_UNITS_2 = args.hidden_units_2
PADDING_1 = args.padding1
PADDING_2 = args.padding2
KSIZE_1 = args.ksize1 
KSIZE_2 = args.ksize2 
STRIDE_1 = args.stride1
STRIDE_2 = args.stride2

# Setup hyperparameters for network training
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
#print(f"[INFO] Training:\n    Model: {model_name}\n    Epochs: {NUM_EPOCHS}\n    Batch size: {BATCH_SIZE}\n    Hidden units: {HIDDEN_UNITS_1}\n    Learning rate: {LEARNING_RATE}")

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}")
print(f"[INFO] Testing data file: {test_dir}")

# Setup transforms
train_transform = args.train_transform
test_transform = args.test_transform
print(f"[INFO] Training transforms: {train_transform}")
print(f"[INFO] Testing transforms: {test_transform}")

#################

def train_model(model_name=model_name,
                NUM_EPOCHS=NUM_EPOCHS,
                BATCH_SIZE=BATCH_SIZE,
                LEARNING_RATE=LEARNING_RATE,
                HIDDEN_UNITS_1=HIDDEN_UNITS_1,
                HIDDEN_UNITS_2=HIDDEN_UNITS_2,
                mn_append=mn_append,
                train_transform=train_transform
                ):

    print("[INFO] Training:"
        f"\n    Model:            {model_name}"
        f"\n    Epochs:           {NUM_EPOCHS}"
        f"\n    Batch size:       {BATCH_SIZE}"
        f"\n    Hidden units (1): {HIDDEN_UNITS_1}"
        f"\n    Hidden units (2): {HIDDEN_UNITS_2}"
    #    f"\n    Padding (1):      {PADDING_1}"
    #    f"\n    Padding (2):      {PADDING_2}"
    #    f"\n    Kernel size (1):  {KSIZE_1}"
    #    f"\n    Kernel size (2):  {KSIZE_2}"
    #    f"\n    Stride (1):       {STRIDE_1}"
    #    f"\n    Stride (2):       {STRIDE_2}"
    )

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #train_transform = transforms.Compose([
    #    transforms.Resize((64, 64)),
    #    #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Randomly adjust color
    #    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
    #    #transforms.RandomRotation(degrees=90),  # Randomly rotate the image up to 30 degrees
    #    #transforms.GaussianBlur(kernel_size=3, sigma=0.1),
    #    #transforms.TrivialAugmentWide(num_magnitude_bins=31),
    #    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    #    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image pixels
    #])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model.py
    model_name = "TinyVGG"
    model_class = getattr(model_2, model_name) # Assume model_name contains the string name of the class

    model = model_class(
        input_shape=3, # RGB
        output_shape=len(class_names),
        hidden_units_1=HIDDEN_UNITS_1,
        hidden_units_2=HIDDEN_UNITS_2,
        #hu_b1=20,
        #hu_b2=20,
        #ksize_b1=3,
        #ksize_b2=3,
        #padding1=PADDING_1,     
        #padding2=PADDING_2,     
        #ksize1=KSIZE_1,       
        #ksize2=KSIZE_2,       
        #stride1=STRIDE_1,      
        #stride2=STRIDE_2       
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    # Start training with help from engine.py
    results = engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS,
                        device=device)

    # Save the model with help from utils.py
    train_duration = sum(results.get("epoch_duration"))
    test_acc = results.get("test_acc")[-1]

    # Quantiles
    test_acc_list = results.get("test_acc")
    q_01 = np.quantile(a=test_acc_list, q=0.01)
    q_25 = np.quantile(a=test_acc_list, q=0.25)
    q_50 = np.quantile(a=test_acc_list, q=0.50)
    q_75 = np.quantile(a=test_acc_list, q=0.75)
    q_100 = np.quantile(a=test_acc_list, q=1)

    print(f"Q01: {q_01:.2f}\nQ25: {q_25:.2f}\nQ50: {q_50:.2f}\nQ75: {q_75:.2f}\nQ100: {q_100:.2f}")

    m_name = f"tf{mn_append}_{model_name}_{train_duration:.2f}s_{test_acc:.4f}test-acc_q01_{q_01:.2f}_q100_{q_100:.2f}.pth"

    utils.save_model(model=model,
                    target_dir="models",
                    model_name=m_name)