import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from model import MyModel as model1
from  model2 import MyModel as model2
from Functions.Embeddings import DCT, DWT


def pkl_to_train_loader(file_path, lowest_output_index, Embedding=None, batch_size=64):
    """Loading pickle file and creating training_input/output list"""
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
    dataset = TensorDataset(torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader


def train_model(model, data_loader, optimizer, criterion, epochs=1000, patience=11):
    torch.autograd.set_detect_anomaly(True)
    best_loss = float('inf')
    current_patience = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            #print("hallp")
           # print(outputs)
            #print("t: " + str(targets))
            loss = criterion(outputs, targets)
            loss.backward()
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


def training_model(name, Embedding):
    data_loader = pkl_to_train_loader("90000_training_sample_sin_cos_creator6.pkl", 99, Embedding)
    our_model = model2(input_size=99)
    #optimizer = optim.SGD(our_model.parameters(), lr=0.01)
    optimizer = optim.Adam(our_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_model(our_model , data_loader, optimizer, criterion, epochs=1000)
    torch.save(our_model.state_dict(), name)


if __name__== "__main__":
    training_model("model2_DWT2",DWT)