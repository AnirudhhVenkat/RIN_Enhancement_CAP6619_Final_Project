import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from torchvision import transforms
from tqdm import tqdm
import os
import random
from resnet50 import get_radimagenet_resnet50

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class RadiologyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transforms.Compose([transforms.ToTensor(),])
        
        #label mapping because the labels are strings
        self.unique_labels = sorted(dataframe['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.num_classes = len(self.unique_labels)
        
        print(f"Found {self.num_classes} unique classes in subset:")
        for label, idx in self.label_to_idx.items():
            print(f"  {label} -> {idx}")

    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        filename = self.dataframe.iloc[idx]['filename']
        label_str = self.dataframe.iloc[idx]['label']
        label_idx = self.label_to_idx[label_str]  # Convert string label to integer index
        
        # Read and preprocess image
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))  # Resize to 224x224 as per paper
        
        # Convert to float and normalize to [0,1]
        image = image.astype(np.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label_idx, dtype=torch.long)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait before early stopping
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1)
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation - Loss: {avg_val_loss:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'radimagenet_model_best.pth')
            print(f'New best model saved with validation loss: {avg_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs')
            
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    # Save final model weights
    torch.save(model.state_dict(), 'pretrained_resnet50_model_final.pth')
    print("Final model weights saved to pretrained_resnet50_model_final.pth")

def main():
    # Set random seeds for reproducibility
    set_seed(42)
    
    # Load pre-split data
    train_df = pd.read_csv('RadiologyAI_train.csv')
    val_df = pd.read_csv('RadiologyAI_val.csv')
    test_df = pd.read_csv('RadiologyAI_test.csv')
    
    # Sample 0.2% of data from each split
    sample_size_train = int(len(train_df) * 0.015)
    sample_size_val = int(len(val_df) * 0.015)
    sample_size_test = int(len(test_df) * 0.015)
    
    train_df = train_df.sample(n=sample_size_train, random_state=42)
    val_df = val_df.sample(n=sample_size_val, random_state=42)
    test_df = test_df.sample(n=sample_size_test, random_state=42)
    
    print(f"Training set size: {len(train_df)} (1.5% of original)")
    print(f"Validation set size: {len(val_df)} (1.5% of original)")
    print(f"Test set size: {len(test_df)} (1.5% of original)")
    
    # Create datasets
    train_dataset = RadiologyDataset(train_df)
    val_dataset = RadiologyDataset(val_df)
    test_dataset = RadiologyDataset(test_df)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model with 165 classes (full RadImageNet)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_radimagenet_resnet50(num_classes=165).to(device)  # Always use 165 classes
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 100
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    
    # Final evaluation on test set
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'Final Test Loss: {avg_test_loss:.4f}')

if __name__ == "__main__":
    main() 