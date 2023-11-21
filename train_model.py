import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from model import MyModel as model

def pkl_to_train_loader(file_path, lowest_output_index):
    """Loading pickle file and creating training_input/output list"""
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    training_input, training_output = zip(*((vector[:lowest_output_index], vector[lowest_output_index:]) for vector in loaded_data))
    return train_loader(list(training_input), list(training_output))

def train_loader(input_data, target_data, batch_size=64):
    """Creating Pytorch DataLoader for training"""
    dataset = TensorDataset(torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader

def train_model(model, data_loader, optimizer, criterion, epochs=1000):
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss}")

def first_try_training_model():
    data_loader = pkl_to_train_loader("90000_training_sample_sin_cos_creator6.pkl", 99)
    our_model = model(input_size=99)
    optimizer = optim.SGD(our_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    train_model(our_model , data_loader, optimizer, criterion, epochs=1000)
    torch.save(our_model.state_dict(), 'trained_model1')

if __name__== "__main__":
    first_try_training_model()