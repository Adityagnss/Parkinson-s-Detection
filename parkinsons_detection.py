import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data loading and preprocessing
def load_and_preprocess_data(file_path, sequence_length=50):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Extract relevant features (MFCCs, Jitter, Shimmer)
    jitter_cols = ['Jitter_rel', 'Jitter_abs', 'Jitter_RAP', 'Jitter_PPQ']
    shimmer_cols = ['Shim_loc', 'Shim_dB', 'Shim_APQ3', 'Shim_APQ5', 'Shi_APQ11']
    mfcc_cols = [col for col in df.columns if col.startswith('MFCC')]
    
    # Combine selected features
    selected_features = jitter_cols + shimmer_cols + mfcc_cols
    X = df[selected_features].values
    y = df['Status'].values
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape data for sequence input (batch_size, time_steps, features)
    num_samples = len(X) - sequence_length + 1
    sequences = np.zeros((num_samples, sequence_length, X.shape[1]))
    
    for i in range(num_samples):
        sequences[i] = X[i:i + sequence_length]
    
    # Adjust labels accordingly
    y = y[sequence_length-1:]
    
    return sequences, y

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(key_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_model(input_shape):
    """Create the model architecture"""
    inputs = layers.Input(shape=input_shape[1:], name='input_layer')
    
    # First Conv1D block
    x = layers.Conv1D(filters=128, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Second Conv1D block with residual connection
    conv_out = layers.Conv1D(filters=128, kernel_size=3, padding='same')(x)
    conv_out = layers.BatchNormalization()(conv_out)
    conv_out = layers.Activation('relu')(conv_out)
    conv_out = layers.Dropout(0.2)(conv_out)
    
    # Add residual connection
    x = layers.Add()([conv_out, x])
    
    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True)
    )(x)
    x = layers.Dropout(0.3)(x)
    
    # Transformer block
    x = TransformerBlock(num_heads=8, key_dim=128, ff_dim=128)(x)
    
    # Global pooling and dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def plot_training_history(history, save_path):
    """Plot and save training history metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def train_model(file_path):
    """Train the model using k-fold cross validation"""
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)
    
    # Initialize k-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"model_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model for each fold
    fold_no = 1
    for train_idx, val_idx in kfold.split(X, y):
        print(f"\nTraining Fold {fold_no}/5")
        
        # Create fold directory
        fold_dir = os.path.join(output_dir, f"fold_{fold_no}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create and compile model
        model = create_model(X_train.shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Create learning rate scheduler
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Save model architecture and weights
        model_path = os.path.join(fold_dir, 'model.h5')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[reduce_lr, early_stop, checkpoint],
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss, val_acc = model.evaluate(X_val, y_val)
        print(f"\nValidation accuracy for fold {fold_no}: {val_acc:.4f}")
        
        fold_no += 1
    
    print(f"\nModel outputs saved to: {output_dir}")
    return model, history, output_dir

def predict_parkinsons(model_path, input_data_path):
    """
    Make predictions using the trained model
    Args:
        model_path: Path to the saved model
        input_data_path: Path to the input data CSV file
    """
    # Load and preprocess the data
    X, _ = load_and_preprocess_data(input_data_path)
    
    # Define custom objects
    custom_objects = {'TransformerBlock': TransformerBlock}
    
    # Load the model with custom objects
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Convert predictions to binary (0 or 1)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Calculate the percentage of positive predictions
    positive_percentage = (binary_predictions.sum() / len(binary_predictions)) * 100
    
    print(f"\nPrediction Results:")
    print(f"Total samples: {len(binary_predictions)}")
    print(f"Positive predictions (Parkinson's): {binary_predictions.sum()}")
    print(f"Negative predictions (Healthy): {len(binary_predictions) - binary_predictions.sum()}")
    print(f"Percentage of positive predictions: {positive_percentage:.2f}%")
    
    return binary_predictions, predictions

if __name__ == "__main__":
    file_path = "/Users/gadeynagasrisaiaditya/Desktop/Aditya/acm project/coding1/ReplicatedAcousticFeatures-ParkinsonDatabase 3.csv"
    
    # Train the model
    model, history, output_dir = train_model(file_path)
    print(f"\nModel outputs saved to: {output_dir}")
    
    # Make predictions using the best model from the last fold
    model_path = os.path.join(output_dir, "fold_5", "model.h5")
    binary_predictions, raw_predictions = predict_parkinsons(model_path, file_path)
