import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchsummary import summary

def get_radimagenet_resnet50(num_classes=165, pretrained=False):
    """
    Get ResNet50 model with RadImageNet modifications:
    1. Global average pooling
    2. Dropout layer (0.5)
    3. Softmax activation
    4. 165 output classes
    """
    # Load base ResNet50 model
    model = resnet50(pretrained=False)  # Changed to False to use random initialization
    
    # Modify the final layer for RadImageNet
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes),
        nn.Softmax(dim=1)
    )
    
    return model


if __name__ == "__main__":
    # Create model
    model = get_radimagenet_resnet50(num_classes=165)
    

    