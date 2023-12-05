import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from model_simple import MyModel as model_simple2
import numpy as np

def pkl_to_train_loader(file_path, lowest_output_index, Embedding=None, batch_size=64):
    """
    Load a pickle file, extract training input and output, and create a DataLoader.

    Args:
        file_path (str): Path to the pickle file.
        lowest_output_index (int): Index separating input and output in each vector.
        Embedding (callable, optional): Embedding function to apply to input and output vectors.
        batch_size (int, optional): Batch size for the DataLoader.

    Returns:
        DataLoader: PyTorch DataLoader containing the training data.

    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    training_input = []
    training_output = []

    for vector in loaded_data:
        if Embedding is not None:
            training_input.append(Embedding(vector[:lowest_output_index]))
            training_output.append(Embedding(vector[lowest_output_index:]))
        else:
            training_input.append(vector[:lowest_output_index])
            training_output.append(vector[lowest_output_index:])

    return train_loader(list(training_input), list(training_output), batch_size)


def train_loader(input_data, target_data, batch_size):
    """Creating Pytorch DataLoader for training"""
    dataset = TensorDataset(torch.tensor(input_data), torch.tensor(target_data))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader


def train_model(model, data_loader, optimizer, criterion, epochs=1000, patience=11):
    """
    Train a PyTorch model using the provided DataLoader.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        data_loader (DataLoader): PyTorch DataLoader containing training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (torch.nn.Module): The loss criterion.
        epochs (int, optional): Number of training epochs. Default is 1000.
        patience (int, optional): Patience for early stopping. Default is 11.

    Returns:
        None

    """
    best_loss = float('inf')
    current_patience = 0

    for epoch in range(epochs):
        total_loss = 0.0

        for inputs, targets in data_loader:
            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                raise(ValueError, "Input data contains NaN values")
                
            optimizer.zero_grad()
            outputs = model(inputs)
            # Ensure consistent data types
            if outputs.dtype != targets.dtype:
                # Convert model's output to the same data type as targets
                outputs = outputs.to(targets.dtype)

            # Now both outputs and targets have the same data type
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2) # gradient clipping
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss}")

        # Early stopping
        if average_loss < best_loss:
            best_loss = average_loss
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                print(f"Early stopping at epoch {epoch}")
                break



def training_model(name, Embedding=None):
    """
    Train a PyTorch model using data loaded from a pickle file and save the trained model.

    Args:
        name (str): Name of the file (without extension) to save the trained model.
        Embedding (callable): Embedding function to apply to input and output vectors.

    Returns:
        None

    """
    # Load data from a pickle file and create a DataLoader
    data_loader = pkl_to_train_loader("Polyformer/99999_training_sample_sin_cos_creator.pkl", 256, Embedding)

    # Create the model
    our_model = model_simple2()

    # Set up optimizer and loss criterion
    optimizer = optim.Adam(our_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)

    # Train the model
    train_model(our_model, data_loader, optimizer, criterion, epochs=1000)

    # Save the trained model in the same folder as the code
    model_path = name + '.pth'
    torch.save(our_model.state_dict(), model_path)


if __name__== "__main__":
    training_model("model1")