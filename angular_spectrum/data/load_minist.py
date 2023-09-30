import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np



def get_minist(device='cuda'):
    # keep it simple: no validate
    transform = transforms.Compose((
        transforms.ToTensor(), 
        
        )
    )

    training_set = torchvision.datasets.MNIST(
        root='minist', 
        train=True,
        download=True, 
        transform=transform
    )
    
    test_set = torchvision.datasets.MNIST(
        root='minist', 
        train=False,
        download=True, 
        transform=transform
    )
    return (
        training_set.data.to(device) / 255, 
        test_set.data.to(device) / 255, 
        training_set.targets.to(device), 
        test_set.targets.to(device), 
    )


