import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def get_radimagenet_inception_resnet_v2(num_classes=165):
    """
    Creates an Inception-ResNet-v2 model modified for RadImageNet.
    
    Modifications:
    1. Random weight initialization
    2. Global average pooling
    3. Dropout (0.5)
    4. Softmax output layer
    5. 165 output classes
    
    Args:
        num_classes (int): Number of output classes (default: 165 for RadImageNet)
    
    Returns:
        torch.nn.Module: Modified Inception-ResNet-v2 model
    """
    # Load the model with random weights using timm
    model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=num_classes)
    
    # Get the number of features in the last layer
    num_ftrs = model.classif.in_features
    
    # Replace the final classification layer with our custom structure
    model.classif = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes),
        nn.Softmax(dim=1)
    )
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    return model


if __name__ == "__main__":
    # Create and test the model
    model = get_radimagenet_inception_resnet_v2()
    print(model)
    
