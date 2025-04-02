import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
from torchvision import datasets, transforms
from dataset_utils import train_model, evaluate_model

# Load class names
with open('flower_classes.txt') as f:
    class_names = [line.strip() for line in f]

if __name__ == "__main__":
    # Dataset transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load data
    train_data = datasets.Flowers102(
        root='./data',
        split='train',
        transform=train_transform,
        download=True
    )
    
    # Train model
    model = train_model(train_data, num_epochs=15)
    
    # Evaluate
    test_data = datasets.Flowers102(
        root='./data',
        split='test',
        transform=train_transform
    )
    evaluate_model(model, test_data)