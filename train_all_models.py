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
from densenet121 import get_radimagenet_densenet121
from inception_v3 import get_radimagenet_inception_v3
from inception_resnet_v2 import get_radimagenet_inception_resnet_v2
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle

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

def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Get model architecture based on name"""
    if model_name == 'resnet50':
        return get_radimagenet_resnet50(num_classes=num_classes)
    elif model_name == 'densenet121':
        return get_radimagenet_densenet121(num_classes=num_classes)
    elif model_name == 'inception_v3':
        return get_radimagenet_inception_v3(num_classes=num_classes)
    elif model_name == 'inception_resnet_v2':
        return get_radimagenet_inception_resnet_v2(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                num_epochs: int, 
                device: torch.device,
                model_name: str,
                early_stopping_patience: int = 5) -> Dict[str, Any]:
    """Train a model and return training history"""
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0
    }
    
    # Create directory for saving models if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
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
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
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
        history['val_loss'].append(avg_val_loss)
        print(f'Validation - Loss: {avg_val_loss:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            history['best_epoch'] = epoch
            
            # Save best model with additional information
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'model_name': model_name,
                'num_classes': model.fc.out_features if hasattr(model, 'fc') else model.classifier.out_features
            }
            torch.save(save_dict, f'saved_models/{model_name}_best.pth')
            print(f'New best model saved with validation loss: {avg_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs')
            
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    # Save final model with additional information
    final_save_dict = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1],
        'model_name': model_name,
        'num_classes': model.fc.out_features if hasattr(model, 'fc') else model.classifier.out_features,
        'training_history': history
    }
    torch.save(final_save_dict, f'saved_models/{model_name}_final.pth')
    print(f"Final model weights and training history saved to saved_models/{model_name}_final.pth")
    
    return history

def plot_and_save_roc_curves(y_true, y_score, num_classes, model_name, phase='test'):
    """Plot and save ROC curves for each class"""
    # Create directory for ROC curves if it doesn't exist
    os.makedirs('roc_curves', exist_ok=True)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'yellow'])
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {model_name} ({phase} set)')
    plt.legend(loc="lower right")
    
    # Save the figure
    plt.savefig(f'roc_curves/{model_name}_{phase}_roc.png')
    plt.close()
    
    return roc_auc

def evaluate_model(model: nn.Module, 
                  data_loader: DataLoader, 
                  criterion: nn.Module, 
                  device: torch.device,
                  model_name: str,
                  phase: str = 'test') -> Dict[str, Any]:
    """Evaluate model and return metrics including ROC curves"""
    model.eval()
    all_labels = []
    all_outputs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    
    # Convert labels to one-hot encoding
    num_classes = all_outputs.shape[1]
    y_true = np.zeros((len(all_labels), num_classes))
    y_true[np.arange(len(all_labels)), all_labels] = 1
    
    # Calculate and save ROC curves
    roc_auc = plot_and_save_roc_curves(y_true, all_outputs, num_classes, model_name, phase)
    
    # Calculate accuracy
    predicted = np.argmax(all_outputs, axis=1)
    accuracy = np.mean(predicted == all_labels)
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

def main():
    # Set random seeds for reproducibility
    set_seed(42)
    
    # Load pre-split data
    train_df = pd.read_csv('RadiologyAI_train.csv')
    val_df = pd.read_csv('RadiologyAI_val.csv')
    test_df = pd.read_csv('RadiologyAI_test.csv')
    
    # Sample 1.5% of data from each split
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
    
    # Fixed parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Models to train
    model_names = ['resnet50', 'densenet121', 'inception_v3', 'inception_resnet_v2']
    num_classes = train_dataset.num_classes
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training parameters
    early_stopping_patience = 5
    
    # Train each model
    for model_name in model_names:
        print(f"\nTraining {model_name}...")
        
        # Initialize model
        model = get_model(model_name, num_classes).to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
            device=device,
            model_name=model_name,
            early_stopping_patience=early_stopping_patience
        )
        
        # Evaluate on test set and save ROC curves
        test_metrics = evaluate_model(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            model_name=model_name,
            phase='test'
        )
        
        print(f'{model_name} - Test Loss: {test_metrics["loss"]:.4f}, Accuracy: {test_metrics["accuracy"]:.2%}')
        print(f'Average ROC AUC: {np.mean(list(test_metrics["roc_auc"].values())):.4f}')

if __name__ == "__main__":
    main() 