import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

seed = torch.manual_seed(42)

def generate_data():
    image_1 = torch.zeros((32, 1, 16, 16), dtype = torch.float32)
    image_1[:, :, :, 6:10] = 1
    image_1 = image_1 + torch.randn_like(image_1) * 0.2
    image_2 = torch.zeros((32, 1, 16, 16), dtype = torch.float32)
    image_2[:, :, 6:10, :] = 1
    image_2 = image_2 + torch.randn_like(image_2) * 0.2
    image_3 = torch.zeros((32, 1, 16, 16), dtype = torch.float32)
    image_3[:, :, :, 6:10] = 1
    image_3[:, :, 6:10, :] = 1
    image_3 = image_3 + torch.randn_like(image_3) * 0.2

    label_1 = torch.full((32,), 0, dtype = torch.long)
    label_2 = torch.full((32,), 1, dtype = torch.long)
    label_3 = torch.full((32,), 2, dtype = torch.long)

    indices = torch.randperm(32)
    train_indice, val_indice, test_indice = indices[:16], indices[16:24], indices[24:]

    train_images = torch.cat([image_1[train_indice], image_2[train_indice], image_3[train_indice]], dim = 0)
    train_labels = torch.cat([label_1[train_indice], label_2[train_indice], label_3[train_indice]], dim = 0)
    train_ids = torch.arange(48, dtype = torch.long)
    val_images = torch.cat([image_1[val_indice], image_2[val_indice], image_3[val_indice]], dim = 0)
    val_labels = torch.cat([label_1[val_indice], label_2[val_indice], label_3[val_indice]], dim = 0)
    test_images = torch.cat([image_1[test_indice], image_2[test_indice], image_3[test_indice]], dim = 0)
    test_labels = torch.cat([label_1[test_indice], label_2[test_indice], label_3[test_indice]], dim = 0)

    data = {
        "train_images": train_images,
        "train_labels": train_labels,
        "train_ids": train_ids,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels
    }

    return data

def load_data(path):
    data = torch.load(path, map_locatrion = "cpu", weights_only=True)

    return data

def create_train_loader(train_images, train_labels, train_ids, generator):
    train_dataset = TensorDataset(train_images, train_labels, train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size = 16,
        shuffle = True,
        num_workers = 0,
        generator = generator
    )

    return train_loader

def create_val_loader(val_images, val_labels, val_ids, generator):
    val_dataset = TensorDataset(val_images, val_labels, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size = 16,
        shuffle = True,
        num_workers = 0,
        generator = generator
    )

    return val_loader

def create_test_loader(test_images, test_labels, test_ids, generator):
    test_dataset = TensorDataset(test_images, test_labels, test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size = 16,
        shuffle = True,
        num_workers = 0,
        generator = generator
    )

    return test_loader
    




