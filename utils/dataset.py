import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(X, y, batch_size=64):
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = torch.tensor(X[:train_size], dtype=torch.float32)
    y_train = torch.tensor(y[:train_size], dtype=torch.float32)

    X_val = torch.tensor(X[train_size:train_size+val_size], dtype=torch.float32)
    y_val = torch.tensor(y[train_size:train_size+val_size], dtype=torch.float32)

    X_test = torch.tensor(X[train_size+val_size:], dtype=torch.float32)
    y_test = torch.tensor(y[train_size+val_size:], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader