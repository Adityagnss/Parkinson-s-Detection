import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
import librosa
import soundfile as sf
import tempfile

# Pre-computed statistics from training data
FEATURE_MEANS = {
    'Jitter_rel': 0.00632,
    'Jitter_abs': 0.00004,
    'Jitter_RAP': 0.00323,
    'Jitter_PPQ': 0.00339,
    'Shim_loc': 0.02792,
    'Shim_dB': 0.26314,
    'Shim_APQ3': 0.01337,
    'Shim_APQ5': 0.01635,
    'Shi_APQ11': 0.02041
}

FEATURE_STDS = {
    'Jitter_rel': 0.00564,
    'Jitter_abs': 0.00003,
    'Jitter_RAP': 0.00278,
    'Jitter_PPQ': 0.00291,
    'Shim_loc': 0.01992,
    'Shim_dB': 0.19124,
    'Shim_APQ3': 0.00951,
    'Shim_APQ5': 0.01170,
    'Shi_APQ11': 0.01467
}

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        
        # Initialize layers in build method
        self.attention = None
        self.ffn_layer1 = None
        self.ffn_layer2 = None
        self.layer_norm1 = None
        self.layer_norm2 = None
        self.dropout1 = None
        self.dropout2 = None

    def build(self, input_shape):
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim)
        self.ffn_layer1 = tf.keras.layers.Dense(self.ff_dim, activation="relu")
        self.ffn_layer2 = tf.keras.layers.Dense(input_shape[-1])
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        super().build(input_shape)

    def call(self, inputs, training=None):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)
        
        ffn_output = self.ffn_layer1(out1)
        ffn_output = self.ffn_layer2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layer_norm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
        })
        return config

# Register the custom layer
tf.keras.utils.get_custom_objects()["TransformerBlock"] = TransformerBlock

def create_model(input_shape=(50, 22)):
    """Create the model architecture"""
    inputs = tf.keras.layers.Input(shape=input_shape, name='input_layer')
    
    # First Conv1D block
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Second Conv1D block with residual connection
    conv = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(x)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)
    conv = tf.keras.layers.Dropout(0.2)(conv)
    
    # Add residual connection
    x = tf.keras.layers.Add()([conv, x])
    
    # Bidirectional LSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Transformer block
    transformer_block = TransformerBlock(
        num_heads=8,
        key_dim=128,
        ff_dim=128
    )
    x = transformer_block(x)
    
    # Global pooling and dense layers
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model():
    """Load the trained model"""
    try:
        # Use cached model if available
        if 'model' in st.session_state:
            return st.session_state.model
            
        # Get absolute path to current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Find the most recent model directory
        model_dirs = [d for d in os.listdir(current_dir) if d.startswith('model_output_')]
        if not model_dirs:
            raise FileNotFoundError("No model directories found")
            
        latest_dir = max(model_dirs)  # Get the most recent directory
        model_path = os.path.join(current_dir, latest_dir, 'fold_3', 'model.h5')
        
        st.write(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Load model with custom objects
        custom_objects = {'TransformerBlock': TransformerBlock}
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path)
            
        # Cache the model
        st.session_state.model = model
        st.success("Model loaded successfully!")
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Current directory: {current_dir}")
        st.error(f"Available directories: {os.listdir(current_dir)}")
        raise Exception(f"Could not load model: {str(e)}")

def extract_features_from_audio(audio_file):
    """Extract voice features from audio file"""
    try:
        st.write("Loading audio file...")
        # Load audio file
        y, sr = librosa.load(audio_file, duration=5)  # Load 5 seconds for consistency
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Split into frames
        frame_length = int(sr * 0.04)  # 40ms frames
        hop_length = int(sr * 0.02)    # 20ms hop
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        
        features_list = []
        for frame in frames.T:
            if np.std(frame) < 0.01:  # Skip silent frames
                continue
            
            # Extract pitch
            f0, voiced_flag, voiced_probs = librosa.pyin(frame, 
                                                        fmin=librosa.note_to_hz('C2'), 
                                                        fmax=librosa.note_to_hz('C7'),
                                                        sr=sr,
                                                        frame_length=frame_length)
            
            if voiced_flag is None or not any(voiced_flag):
                continue
                
            f0 = f0[voiced_flag]
            
            if len(f0) < 2:  # Need at least 2 points for jitter
                continue
            
            # Calculate periods
            periods = 1.0 / f0
            
            # Jitter features
            jitter_abs = np.mean(np.abs(np.diff(periods)))
            jitter_rel = jitter_abs / np.mean(periods)
            
            # RAP (Relative Average Perturbation)
            if len(periods) >= 3:
                rap = np.mean(np.abs(periods[1:-1] - np.mean([periods[:-2], periods[2:]], axis=0)))
                jitter_rap = rap / np.mean(periods)
            else:
                jitter_rap = 0
            
            # PPQ5 (Five-point Period Perturbation Quotient)
            if len(periods) >= 5:
                ppq = np.mean(np.abs(periods[2:-2] - np.mean([periods[:-4], periods[1:-3],
                                                             periods[3:-1], periods[4:]], axis=0)))
                jitter_ppq = ppq / np.mean(periods)
            else:
                jitter_ppq = 0
            
            # RMS energy for shimmer
            rms = librosa.feature.rms(y=frame, frame_length=frame_length, hop_length=frame_length)[0]
            
            if len(rms) >= 2:
                # Local shimmer
                shim_loc = np.mean(np.abs(np.diff(rms))) / np.mean(rms)
                shim_db = 20 * np.log10(shim_loc + 1e-10)
                
                # APQ3
                if len(rms) >= 3:
                    apq3 = np.mean(np.abs(rms[1:-1] - np.mean([rms[:-2], rms[2:]], axis=0))) / np.mean(rms)
                else:
                    apq3 = 0
                
                # APQ5
                if len(rms) >= 5:
                    apq5 = np.mean(np.abs(rms[2:-2] - np.mean([rms[:-4], rms[1:-3],
                                                              rms[3:-1], rms[4:]], axis=0))) / np.mean(rms)
                else:
                    apq5 = 0
                
                # APQ11
                if len(rms) >= 11:
                    apq11 = np.mean(np.abs(rms[5:-5] - np.mean([rms[:-10], rms[1:-9], rms[2:-8],
                                                               rms[3:-7], rms[4:-6], rms[6:-4],
                                                               rms[7:-3], rms[8:-2], rms[9:-1],
                                                               rms[10:]], axis=0))) / np.mean(rms)
                else:
                    apq11 = 0
            else:
                shim_loc = shim_db = apq3 = apq5 = apq11 = 0
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            
            # Combine features
            frame_features = [
                jitter_rel,    # Jitter_rel
                jitter_abs,    # Jitter_abs
                jitter_rap,    # Jitter_RAP
                jitter_ppq,    # Jitter_PPQ
                shim_loc,      # Shim_loc
                shim_db,       # Shim_dB
                apq3,          # Shim_APQ3
                apq5,          # Shim_APQ5
                apq11          # Shi_APQ11
            ] + list(mfcc_means)
            
            features_list.append(frame_features)
        
        if not features_list:
            raise ValueError("No valid voice segments found in the audio")
        
        # Calculate mean features
        features_array = np.array(features_list)
        mean_features = np.mean(features_array, axis=0)
        
        # Scale features using pre-computed statistics
        scaled_features = []
        feature_names = [
            'Jitter_rel', 'Jitter_abs', 'Jitter_RAP', 'Jitter_PPQ',
            'Shim_loc', 'Shim_dB', 'Shim_APQ3', 'Shim_APQ5', 'Shi_APQ11'
        ]
        
        # Debug original values
        st.write("\nOriginal feature values:")
        for name, value in zip(feature_names, mean_features[:9]):
            st.write(f"{name}: {value:.6f}")
        
        # Scale voice features
        for i, name in enumerate(feature_names):
            value = mean_features[i]
            mean = FEATURE_MEANS[name]
            std = FEATURE_STDS[name]
            scaled_value = (value - mean) / (std if std != 0 else 1)
            scaled_features.append(scaled_value)
        
        # Debug scaled values
        st.write("\nScaled feature values:")
        for name, value in zip(feature_names, scaled_features):
            st.write(f"{name}: {value:.6f}")
        
        # Scale MFCCs using standard scaling
        mfcc_features = mean_features[9:]
        mfcc_mean = np.mean(mfcc_features)
        mfcc_std = np.std(mfcc_features)
        scaled_mfccs = (mfcc_features - mfcc_mean) / (mfcc_std if mfcc_std != 0 else 1)
        scaled_features.extend(scaled_mfccs)
        
        return scaled_features
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        import traceback
        st.error("Traceback: " + traceback.format_exc())
        raise

def preprocess_features(features):
    """Preprocess the input features"""
    try:
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        # Create a sequence of 50 timesteps by repeating the features
        # The model expects shape (batch_size, timesteps=50, features=22)
        features = np.tile(features, (50, 1))  # Repeat features 50 times
        features = features.reshape(1, 50, -1)  # Add batch dimension
        
        st.write("Debug: Preprocessed features shape:", features.shape)
        st.write("Debug: Feature values (first timestep):", features[0, 0])
        
        return features
    except Exception as e:
        st.error(f"Error preprocessing features: {str(e)}")
        raise

def predict_parkinsons(_model, features):
    """Make prediction using the model"""
    try:
        st.write("Starting prediction...")
        
        # Preprocess features
        processed_features = preprocess_features(features)
        
        # Make prediction
        with tf.device('/CPU:0'):  # Use CPU to avoid Metal plugin issues
            prediction = _model.predict(processed_features, verbose=0)
            st.write("Raw prediction shape:", prediction.shape)
            st.write("Raw prediction values:", prediction)
        
        # Get probability (average across timesteps if needed)
        if len(prediction.shape) > 2:
            probability = float(np.mean(prediction))
        else:
            probability = float(prediction[0, 0])
            
        st.write("Final probability:", probability)
        
        return probability
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        raise

def analyze_voice(audio_file):
    """Analyze voice recording and make prediction"""
    try:
        # Extract features
        features = extract_features_from_audio(audio_file)
        
        # Load model
        model = load_model()
        
        # Make prediction
        probability = predict_parkinsons(model, features)
        
        # Clear previous results
        if 'last_probability' in st.session_state:
            del st.session_state.last_probability
        
        # Store new results
        st.session_state.last_probability = probability
        
        # Display results
        display_results(probability)
        
    except Exception as e:
        st.error(f"Error analyzing voice: {str(e)}")
        raise

def display_results(probability):
    """Display analysis results"""
    try:
        st.subheader("Analysis Result")
        
        # Calculate risk level
        if probability < 0.4:
            risk = "Low Risk"
            color = "green"
        elif probability < 0.7:
            risk = "Moderate Risk"
            color = "orange"
        else:
            risk = "High Risk"
            color = "red"
        
        # Display risk level and probability
        st.markdown(f"<h3 style='color: {color}'>{risk}</h3>", unsafe_allow_html=True)
        st.write(f"{probability*100:.1f}%")
        st.write("Probability of Parkinson's Indicators")
        
        # Add explanation
        with st.expander("See Analysis Details"):
            st.write("""
            This analysis is based on various voice characteristics including:
            - Jitter (variations in pitch)
            - Shimmer (variations in amplitude)
            - Voice breaks and tremors
            - Harmonic characteristics
            
            A higher percentage indicates stronger presence of voice patterns 
            associated with Parkinson's disease. However, this is not a diagnosis 
            and should be discussed with a healthcare provider.
            """)
            
            # Show feature importance if available
            if 'last_features' in st.session_state:
                st.write("\nFeature Importance:")
                features = st.session_state.last_features
                for name, value in zip(['Jitter', 'Shimmer', 'Harmonicity'], 
                                     [features[0], features[4], features[-1]]):
                    importance = abs(value)
                    st.progress(min(importance, 1.0), 
                              text=f"{name}: {importance:.2f}")
        
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        raise

def main():
    st.title("Parkinson's Disease Detection")
    st.write("Upload a voice recording to check for Parkinson's Disease indicators")
    
    st.write("\nPlease record your voice saying 'ahhh' for at least 5 seconds")
    
    # File uploader
    audio_file = st.file_uploader("Upload Voice Recording (WAV or MP3, up to 60 seconds)", 
                                 type=['wav', 'mp3'])
    
    if audio_file is not None:
        try:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_path = tmp_file.name
            
            st.write("Analyzing voice features...")
            
            # Analyze voice
            analyze_voice(tmp_path)
            
            # Cleanup
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error("Error processing audio file. Please try again.")
            st.error(f"Error details: {str(e)}")
            raise

if __name__ == "__main__":
    main()
