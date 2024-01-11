import os
import subprocess
import zipfile
import random
import shutil

def download_kaggle_dataset(dataset_name: str):
    """
    Downloads a Kaggle dataset using the Kaggle API and extracts the contents to a 'data' subfolder.

    Args:
        dataset_name: The Kaggle dataset name in the format 'username/dataset-name'.
    """
    # Create a 'data' directory if it doesn't exist
    data_directory = 'data'
    os.makedirs(data_directory, exist_ok=True)

    # Command to download the dataset using Kaggle API
    command = f"kaggle datasets download -d {dataset_name} -p {data_directory}"

    # Run the command in the terminal
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Print the output
    if output:
        print(output.decode("utf-8"))

    # Print the error if there is any
    if error:
        print(error.decode("utf-8"))

    # Extract the contents of the downloaded zip file
    zip_file_path = os.path.join(data_directory, f"{dataset_name.split('/')[1]}.zip")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_directory)
        print(f"Extracted contents to '{data_directory}' folder.")

    # Remove the downloaded zip file after extraction
    os.remove(zip_file_path)



def train_test_split(source_folder: str, destination_folder: str, split_ratio: float = 0.8):
    """
    Splits folders into training and test data directories.

    Args:
        source_folder: Path to the source folder containing subfolders of data classes.
        destination_folder: Path to the destination folder where 'train' and 'test' folders will be created.
        split_ratio: Ratio for splitting the data into training and test sets. (default: 0.8)
    """
    # Create 'train' and 'test' folders in the destination folder
    train_folder = os.path.join(destination_folder, 'train')
    test_folder = os.path.join(destination_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Iterate through each class folder in the source directory
    for class_folder in os.listdir(source_folder):
        class_path = os.path.join(source_folder, class_folder)
        if os.path.isdir(class_path):
            files = os.listdir(class_path)
            random.shuffle(files)  # Shuffle files to randomize split

            # Calculate split indices
            split_index = int(len(files) * split_ratio)

            # Split files into train and test sets
            train_files = files[:split_index]
            test_files = files[split_index:]

            # Create train and test class folders in the destination 'train' and 'test' folders
            train_class_folder = os.path.join(train_folder, class_folder)
            test_class_folder = os.path.join(test_folder, class_folder)
            os.makedirs(train_class_folder, exist_ok=True)
            os.makedirs(test_class_folder, exist_ok=True)

            # Copy train files to train folder
            for file in train_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(train_class_folder, file)
                shutil.copy(src, dst)

            # Copy test files to test folder
            for file in test_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(test_class_folder, file)
                shutil.copy(src, dst)