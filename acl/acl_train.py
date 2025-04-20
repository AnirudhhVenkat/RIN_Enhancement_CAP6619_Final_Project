#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, InceptionV3, DenseNet121
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint,Callback
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import argparse
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import mixed_precision
from tensorflow.data import Dataset

# Configure TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# Disable XLA compilation
tf.config.optimizer.set_jit(False)

# Check for GPU devices (MPS on macOS)
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("\nGPU/MPS device found! Using GPU for acceleration.")
    print(f"GPU device details: {gpu_devices[0]}")
    # Set memory growth to prevent memory issues
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("\nNo GPU device found. Using CPU.")

# Create a MirroredStrategy for GPU
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

### Set up training image size, batch size and number of epochs
image_size = 256
batch_size = 512
num_epoches = 30

### Creat model
def get_compiled_model(model_name, database, structure, lr):
    if not model_name in ['IRV2', 'ResNet50', 'DenseNet121', 'InceptionV3']:
        raise Exception('Pre-trained network not exists. Please choose IRV2/ResNet50/DenseNet121/InceptionV3 instead')
    else:
        if model_name == 'IRV2':
            if database == 'RadImageNet':
                model_dir = "RadImageNet_models/RadImageNet-IRV2-notop.h5"
                base_model = InceptionResNetV2(weights=None, input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
                base_model.load_weights(model_dir)
            else:
                base_model = InceptionResNetV2(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
        if model_name == 'ResNet50':
            if database == 'RadImageNet':
                model_dir = "RadImageNet_models/RadImageNet-ResNet50-notop.h5"
                base_model = ResNet50(weights=None, input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
                base_model.load_weights(model_dir)
            else:
                base_model = ResNet50(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
        if model_name == 'DenseNet121':
            if database == 'RadImageNet':
                model_dir = "RadImageNet_models/RadImageNet-DenseNet121-notop.h5"
                base_model = DenseNet121(weights=None, input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
                base_model.load_weights(model_dir)
            else:
                base_model = DenseNet121(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
        if model_name == 'InceptionV3':
            if database == 'RadImageNet':
                model_dir = "RadImageNet_models/RadImageNet-InceptionV3-notop.h5"
                base_model = InceptionV3(weights=None, input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
                base_model.load_weights(model_dir)
            else:
                base_model = InceptionV3(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False, pooling='avg')
    
    # Set layer trainability based on structure
    if structure == 'freezeall':
        for layer in base_model.layers:
            layer.trainable = False
    elif structure == 'unfreezeall':
        pass  # All layers are trainable
    elif structure == 'unfreezetop10':
        for layer in base_model.layers[:-10]:
            layer.trainable = False
    
    # Add classification layers
    y = base_model.output
    y = Dropout(0.5)(y)  # Match paper's dropout rate
    predictions = Dense(1, activation='sigmoid')(y)  # Changed from 2 to 1 for binary classification
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Set learning rate based on structure
    if structure == 'freezeall' or structure == 'unfreezetop10':
        lr = 0.01 if lr == 0.01 else 0.001
    else:  # unfreezeall
        lr = 0.001 if lr == 0.001 else 0.0001
    
    # Use legacy optimizer for M1/M2 Macs
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    
    # Compile model with legacy optimizer
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
  
def run_model(model_name, database, structure, lr):
    ### set model
    with strategy.scope():
        model = get_compiled_model(model_name, database, structure, lr)
    
    # Run only fold 1
    i = 0  # Only use fold 1
    print(f"\nRunning fold {i+1}")
    
    # Read CSV files
    df_train = pd.read_csv(f'acl/dataframe/train_fold{i+1}.csv')
    df_val = pd.read_csv(f'acl/dataframe/val_fold{i+1}.csv')
    
    # Sample 10% of the data
    df_train = df_train.sample(frac=0.1, random_state=42)
    df_val = df_val.sample(frac=0.1, random_state=42)
    
    # Convert relative paths to absolute paths
    df_train['filename'] = df_train['filename'].apply(lambda x: os.path.join(os.getcwd(), "acl", x))
    df_val['filename'] = df_val['filename'].apply(lambda x: os.path.join(os.getcwd(), "acl", x))
    
    # Check for missing images
    missing_train = [f for f in df_train['filename'] if not os.path.exists(f)]
    missing_val = [f for f in df_val['filename'] if not os.path.exists(f)]
    
    if missing_train:
        print(f"Warning: {len(missing_train)} training images not found")
        df_train = df_train[~df_train['filename'].isin(missing_train)]
    if missing_val:
        print(f"Warning: {len(missing_val)} validation images not found")
        df_val = df_val[~df_val['filename'].isin(missing_val)]
    
    if len(df_train) == 0 or len(df_val) == 0:
        print(f"Skipping fold {i+1} due to missing images")
        return None, None, None, None
    
    # Set up data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col='filename',
        y_col='acl_label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col='filename',
        y_col='acl_label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    ### Set train steps and validation steps
    train_steps = len(df_train) // batch_size
    val_steps = len(df_val) // batch_size
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=num_epoches,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[early_stopping]
    )
    
    # Calculate metrics
    y_pred = model.predict(validation_generator, steps=len(validation_generator))
    y_true = validation_generator.labels
    
    # Calculate ROC curve and AUROC for validation
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auroc = auc(fpr, tpr)
    
    # Load and evaluate on test fold
    df_test = pd.read_csv('acl/dataframe/test_fold1.csv')
    df_test['filename'] = df_test['filename'].apply(lambda x: os.path.join(os.getcwd(), "acl", x))
    
    # Sample 10% of the test data
    df_test = df_test.sample(frac=0.1, random_state=42)
    
    # Check for missing test images
    missing_test = [f for f in df_test['filename'] if not os.path.exists(f)]
    if missing_test:
        print(f"Warning: {len(missing_test)} test images not found")
        df_test = df_test[~df_test['filename'].isin(missing_test)]
    
    if len(df_test) == 0:
        print("Error: No valid test images found")
        return None, None, None, None
    
    # Create test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        x_col='filename',
        y_col='acl_label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    # Evaluate on test set
    test_y_pred = model.predict(test_generator, steps=len(test_generator))
    test_y_true = test_generator.labels
    test_fpr, test_tpr, _ = roc_curve(test_y_true, test_y_pred)
    test_auroc = auc(test_fpr, test_tpr)
    
    return test_auroc, [], test_fpr.tolist(), test_tpr.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ACL model')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (IRV2, ResNet50, DenseNet121, InceptionV3)')
    parser.add_argument('--database', type=str, required=True, help='Database (RadImageNet or ImageNet)')
    parser.add_argument('--structure', type=str, required=True, help='Structure (freezeall, unfreezeall, unfreezetop10)')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    args = parser.parse_args()

    # Run the model with the provided arguments
    test_auroc, subject_aurocs, fpr, tpr = run_model(
        args.model_name,
        args.database,
        args.structure,
        args.lr
    )

    # Print results in format expected by run_all_experiments.py
    print(f"Test AUROC: {test_auroc}")
    print(f"Subject_AUROCs: {subject_aurocs}")
    print(f"FPR: {fpr}")
    print(f"TPR: {tpr}")
