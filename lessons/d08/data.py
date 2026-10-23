import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# Generate data, use random_seed in train file, not here
def generate_data():
    data_1 = torch.zeros((32, 1, 16, 16), dtype = torch.float32)
    data_1[:, :, :, 6:10] = 1
    data_1 = data_1 + torch.randn_like(data_1) * 0.2
    data_2 = torch.zeros((32, 1, 16, 16), dtype = torch.float32)
    data_2[:, :, 6:10, :] = 1
    data_2 = data_2 + torch.randn_like(data_2) * 0.2
    data_3 = torch.zeros((32, 1, 16, 16), dtype = torch.float32)
    data_3[:, :, :, 6:10] = 1
    data_3[:, :, 6:10, :] = 1
    data_3 = data_3 + torch.randn_like(data_3) * 0.2

    target_1 = torch.full((32,), 0, dtype = torch.long)
    target_2 = torch.full((32,), 1, dtype = torch.long)
    target_3 = torch.full((32,), 2, dtype = torch.long)

    indices = torch.randperm(32)
    train_indice, val_indice, test_indice = indices[:16], indices[16:24], indices[24:]

    train_data = torch.cat([data_1[train_indice], data_2[train_indice], data_3[train_indice]], dim = 0)
    train_target = torch.cat([target_1[train_indice], target_2[train_indice], target_3[train_indice]], dim = 0)
    train_ids = torch.arange(48, dtype = torch.long) # Generate train_data's index

    val_data = torch.cat([data_1[val_indice], data_2[val_indice], data_3[val_indice]], dim = 0)
    val_target = torch.cat([target_1[val_indice], target_2[val_indice], target_3[val_indice]], dim = 0)
    test_data = torch.cat([data_1[test_indice], data_2[test_indice], data_3[test_indice]], dim = 0)
    test_target = torch.cat([target_1[test_indice], target_2[test_indice], target_3[test_indice]], dim = 0)

    data = {
        "train_data": train_data,
        "train_target": train_target,
        "train_ids": train_ids,
        "val_data": val_data,
        "val_target": val_target,
        "test_data": test_data,
        "test_target": test_target
    }

    return data

# Load data from data_path
def load_data(path):
    data = torch.load(path, map_location = "cpu", weights_only=True)

    return data

# Create data_loader for training
def create_train_loader(data, generator, batch_size):
    train_data = data["train_data"]
    train_target = data["train_target"]
    train_ids = data["train_ids"]
    
    train_dataset = TensorDataset(train_data, train_target, train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle = True,
        num_workers = 0,
        generator = generator
    )

    return train_loader

# def create_val_loader(val_data, val_target, val_ids, generator):
#     val_dataset = TensorDataset(val_data, val_target, val_ids)
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size = 16,
#         shuffle = True,
#         num_workers = 0,
#         generator = generator
#     )

#     return val_loader

# def create_test_loader(test_data, test_target, test_ids, generator):
#     test_dataset = TensorDataset(test_data, test_target, test_ids)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size = 16,
#         shuffle = True,
#         num_workers = 0,
#         generator = generator
#     )

#     return test_loader
    
