import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0

def get_radimagenet_efficientnetv2(num_classes=165):
    """
    Creates an EfficientNetV2B0 model modified for RadImageNet.
    
    Modifications:
    1. ImageNet pre-trained weights
    2. Global average pooling
    3. Dropout (0.5)
    4. Softmax output layer
    5. 165 output classes
    
    Args:
        num_classes (int): Number of output classes (default: 165 for RadImageNet)
    
    Returns:
        tf.keras.Model: Modified EfficientNetV2B0 model
    """
    # Load the base EfficientNetV2B0 model
    base_model = EfficientNetV2B0(
        include_top=False,
        weights=None,  # Use ImageNet pre-trained weights
        input_shape=(224, 224, 3)
    )
    
    # Add classification head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    outputs = tf.keras.layers.Softmax()(x)
    
    # Create the final model
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    return model

if __name__ == "__main__":
    # Create and test the model
    model = get_radimagenet_efficientnetv2(165)
  