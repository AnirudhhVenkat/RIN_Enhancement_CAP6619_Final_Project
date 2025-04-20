import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2

def get_radimagenet_inception_resnet_v2(num_classes=165):
    """
    Creates an InceptionResNetV2 model modified for RadImageNet.
    
    Modifications:
    1. Random weight initialization
    2. Global average pooling
    3. Dropout (0.5)
    4. Softmax output layer
    5. 165 output classes
    
    Args:
        num_classes (int): Number of output classes (default: 165 for RadImageNet)
    
    Returns:
        tf.keras.Model: Modified InceptionResNetV2 model
    """
    # Load the base InceptionResNetV2 model
    base_model = InceptionResNetV2(
        include_top=False,
        weights=None,  # Use random initialization
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
    model = get_radimagenet_inception_resnet_v2(165)
    print(model)
    
